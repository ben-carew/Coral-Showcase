"""Differentiable likelihood backend (R39), REDACTED for this showcase.

In Coral this file (likelihood_diff.py, ~1500 lines) writes the single-ring and N-ring
likelihood once against a small array-backend shim, so the same code runs under JAX (jit) and
PyTorch (torch.compile) and can be differentiated with respect to the track parameters and,
because every tuning table is an argument, the model's own parameters.

Kept here (my own infrastructure, independent of the physics model):
  * the JAX / PyTorch backend shim;
  * the learned pieces that plug into the light model (MLP light correction, learned per-PMT
    term weights);
  * padded PMT-set buckets, per-kind compilation (value / value+grad / Hessian), and
    serialised ahead-of-time executables that skip re-tracing in every cluster job;
  * scipy's Nelder-Mead re-implemented as ONE on-device lax.while_loop program.

Removed: the physics model itself (predicted light, charge and time pdfs, scattering,
N-ring sums) and the table loaders, which transcribe fiTQun (HK/SK collaboration code).
The four model entry points are stubs below; their signatures are the real ones.
"""
import hashlib
import os
import pickle
from typing import NamedTuple

import numpy as np

LOG_OFFSET = 1.e-20
#: the learned light correction's arrays (an .npz in [tuning_files.light_corrections])
LLC_KEYS = ("W1", "b1", "W2", "b2", "w3", "b3", "mu", "sd")
LLC_FEATURES = ("cos_theta0", "log10_r0_10m", "eta0", "logp", "dwall_10m", "towall_10m")
#: a correction file with kind = term_weights holds learned per-PMT term weights instead: the charge
#: and time terms of each PMT x 2 sigmoid(MLP(features)) (two outputs; 1 at a zero last layer)
LTW_FEATURES = ("cos_theta0", "log10_r0_10m", "eta0", "log10_mu")
BIG = 1.e30
# Padded PMT-set sizes grow geometrically (a likelihood compiles once per size). A GPU is
# latency-bound, so padding is nearly free there and coarse buckets mean few compilations;
# on a CPU padding costs time. CORAL_BUCKET_RATIO / CORAL_BUCKET_MIN override.
BUCKETS = {"gpu": (4.0, 2048), "cpu": (2.0 ** 0.5, 64)}


class Cfg(NamedTuple):
    """Static part of a likelihood: changing a field recompiles. k = every constant of the
    particle's tables and of the run (sorted (name, value) pairs), so a fit step changes
    only the dynamic scalars S (qe, unhit weight, dispersion, charge weight, unhit calib)."""
    r0_interp: bool
    eps_interp: bool
    indirect_in_charge: int     # 0 direct only, 1 every PMT, 2 hit PMTs only
    use_tpdf: bool
    ngpars: int                 # 4 crystal ball, 2 gaussian
    tail_fixed: bool            # crystal-ball tail held at (t_alpha, t_n)
    geom: int                   # 0 none, 1 two 1-D curves, 2 + (towall, p) table
    nsect_steps: int            # bisection steps over a cell's section boundaries
    npars: int
    llc: bool                   # learned light correction on the direct photons (M["llc"])
    ltw: bool                   # learned per-PMT weights of the charge and time terms (M["ltw"])
    k: tuple
    wall_trunc: bool = False    # emission stops where the track leaves the tank (R59)


# ---------------------------------------------------------------- backends

class _Jax:
    name = "jax"

    def __init__(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        # compiled programs persist on node-local disk (never on a shared filesystem)
        cache = os.environ.get("CORAL_JAX_CACHE") or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"coral_jax_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        # XLA's own GPU caches would put this job's cache path into the compile options, and so
        # into every key: a warm-up's cache would never match another job's programs
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
        import jax.numpy as jnp
        import jax.scipy.special as jss
        self.jax, self.np, self.erf = jax, jnp, jss.erf
        self.f64, self.f32, self.i64 = jnp.float64, jnp.float32, jnp.int64

    def asarray(self, a, dtype=None):
        return self.np.asarray(a, dtype=dtype)

    def itrunc(self, x):
        return self.np.trunc(x).astype(self.i64)

    def clip(self, x, lo, hi):
        return self.np.clip(x, lo, hi)

    def cmax(self, x, c):
        return self.np.maximum(x, c)

    def cmin(self, x, c):
        return self.np.minimum(x, c)

    def searchsorted_right(self, xp, x):
        return self.np.searchsorted(xp, x, side="right")

    def tofloat(self, x, dtype):
        return x.astype(dtype)

    def einsum(self, s, *a):
        return self.np.einsum(s, *a)

    def argsort0(self, x):
        return self.np.argsort(x, axis=0, stable=True)

    def rows(self, a, idx):
        return a[idx]

    def at(self, a, i):
        return a[i]

    def take0(self, a, idx):
        return self.np.take_along_axis(a, idx, axis=0)


class _Torch:
    name = "torch"

    def __init__(self):
        import torch
        self.device = torch.device(os.environ.get("CORAL_TORCH_DEVICE", "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
        self.torch = torch
        self.np = torch
        self.erf = torch.special.erf
        self.f64, self.f32, self.i64 = torch.float64, torch.float32, torch.int64

    def asarray(self, a, dtype=None):
        if isinstance(a, self.torch.Tensor):
            return (a if dtype is None else a.to(dtype)).to(self.device)
        a = np.asarray(a)
        if dtype is None:
            dtype = self.f64 if a.dtype.kind == "f" else (self.torch.bool if a.dtype.kind == "b" else self.i64)
        if a.dtype.kind == "f" and dtype == self.f32:
            a = a.astype(np.float32)
        return self.torch.as_tensor(a, dtype=dtype, device=self.device)

    def itrunc(self, x):
        return self.torch.trunc(x).long()

    def clip(self, x, lo, hi):
        T = self.torch.Tensor
        if isinstance(lo, T) or isinstance(hi, T):
            lo = lo if isinstance(lo, T) else self.torch.full_like(x, lo)
            hi = hi if isinstance(hi, T) else self.torch.full_like(x, hi)
            return self.torch.minimum(self.torch.maximum(x, lo), hi)
        return self.torch.clamp(x, lo, hi)

    def cmax(self, x, c):
        return self.torch.clamp(x, min=c)

    def cmin(self, x, c):
        return self.torch.clamp(x, max=c)

    def searchsorted_right(self, xp, x):
        return self.torch.searchsorted(xp, x.reshape(1), right=True)[0]

    def tofloat(self, x, dtype):
        return x.to(dtype)

    def einsum(self, s, *a):
        return self.torch.einsum(s, *a)

    def argsort0(self, x):
        return self.torch.argsort(x, dim=0, stable=True)

    def rows(self, a, idx):
        return a.index_select(0, idx)

    def at(self, a, i):
        """a[i] for a 0-d index tensor, without reading it back to Python (torch.compile)."""
        return a.index_select(0, i.reshape(1))[0]

    def take0(self, a, idx):
        return self.torch.gather(a, 0, idx)


_BACKENDS = {}


def backend(name):
    if name not in _BACKENDS:
        _BACKENDS[name] = _Jax() if name == "jax" else _Torch()
    return _BACKENDS[name]



# ---------------------------------------------------------------- learned model components

def _mlp(B, f, L):
    """The learned light correction g(features): a 2-hidden-layer tanh MLP on the
    normalised features f (F, N) -> (N,); exp(g) multiplies the direct photons."""
    xp = B.np
    z = [(f[i] - L["mu"][i]) / L["sd"][i] for i in range(f.shape[0] if hasattr(f, "shape") else len(f))]
    Z = xp.stack(z, -1)                                   # (N, F)
    h1 = xp.tanh(Z @ L["W1"] + L["b1"])
    h2 = xp.tanh(h1 @ L["W2"] + L["b2"])
    return h2 @ L["w3"] + L["b3"]


def _term_weights(B, M, v, pd, G, mu):
    """Learned weights (charge terms, time term) of a PMT set: 2 sigmoid(MLP(LTW_FEATURES))."""
    xp = B.np
    r0x, r0y, r0z = v[0] - G["px"], v[1] - G["py"], v[2] - G["pz"]
    m = xp.sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
    th0 = -(r0x * pd[0] + r0y * pd[1] + r0z * pd[2]) / m
    eta0 = (r0x * G["ox"] + r0y * G["oy"] + r0z * G["oz"]) / m
    g = _mlp(B, [th0, xp.log10(m / 1000.0), eta0, xp.log10(B.cmax(mu, 1e-3))], M["ltw"])
    w = 2.0 / (1.0 + xp.exp(-g))
    return w[:, 0], w[:, 1]




# ---------------------------------------------------------------- model entry points (REDACTED)
# Each returns the negative log-likelihood summed over a PMT set (or the per-PMT terms) at the
# flat parameter vector X, given the PMT set P, the model tables M, the dynamic scalars S and
# the static configuration cfg. The bodies transcribe fiTQun's model and are not shown.

def _redacted(*_a, **_k):
    raise NotImplementedError("physics model redacted: transcribes fiTQun (HK/SK collaboration code)")


hit_value = unhit_value = hit_terms = unhit_terms = _redacted
nring_hit_value = nring_unhit_value = _redacted


# ---------------------------------------------------------------- PMT sets

def _bucket_policy():
    kind = "cpu"
    if "jax" in _BACKENDS:
        kind = "gpu" if _BACKENDS["jax"].jax.default_backend() == "gpu" else "cpu"
    elif "torch" in _BACKENDS:
        kind = "gpu" if _BACKENDS["torch"].device.type == "cuda" else "cpu"
    ratio, low = BUCKETS[kind]
    return (float(os.environ.get("CORAL_BUCKET_RATIO", ratio)), int(os.environ.get("CORAL_BUCKET_MIN", low)))


def _bucket(n):
    if n <= 0:
        return 0
    ratio, size = _bucket_policy()
    while size < n:
        size = int(math.ceil(size * ratio))
    return size


def bucket_sizes(n_max):
    """Every padded size up to the one holding n_max PMTs."""
    sizes, n = [], 1
    while not sizes or sizes[-1] < n_max:
        b = _bucket(n)
        sizes.append(b)
        n = b + 1
    return sizes


def _pmt_set(B, pos, ori, loc, b5, w5, az, pad=True, weights=None):
    """One PMT set as backend arrays, padded to its size bucket (by repeating its first PMT,
    masked out) so a likelihood compiles once per bucket; w = per-PMT weight (unhit sets)."""
    n = len(pos)
    size = _bucket(n) if pad else n
    idx = np.concatenate([np.arange(n), np.zeros(size - n, dtype=np.int64)])
    mask = np.zeros(size)
    mask[:n] = 1.0
    wts = np.zeros(size)
    wts[:n] = 1.0 if weights is None else weights
    wall = np.array([2, 0, 1])[loc[idx]] if n else np.zeros(0, np.int64)   # Loc 0 top -> table 2, 1 side -> 0, 2 bottom -> 1
    a = B.asarray
    return dict(px=a(pos[idx, 0]), py=a(pos[idx, 1]), pz=a(pos[idx, 2]),
                ox=a(ori[idx, 0]), oy=a(ori[idx, 1]), oz=a(ori[idx, 2]),
                wall=a(wall.astype(np.int64)), b5=a(b5[idx].astype(np.int64)), w5=a(w5[idx]), az=a(az[idx]),
                mask=a(mask), w=a(wts)), idx, n


# _hit_rows (the hits' charge-pdf rows) is redacted: it reads fiTQun's charge-pdf tables.


# ---------------------------------------------------------------- compiled entry points

_COMPILED = {}
_FUNCS = {"hit": hit_value, "unhit": unhit_value, "hit_terms": hit_terms, "unhit_terms": unhit_terms,
          "nring_hit": nring_hit_value, "nring_unhit": nring_unhit_value}


def _compiled(B, kind):
    """kind: <fn> (value), <fn>:vg (value and gradient in X), <fn>:hess (Hessian in X) for fn
    in _FUNCS, each called (X, PMT set, M, S, cfg); one compiled function per backend and kind,
    specialised by jit / torch.compile per static cfg and padded set size."""
    key = (B.name, kind)
    if key in _COMPILED:
        return _COMPILED[key]
    name, _, mode = kind.partition(":")
    base = _FUNCS[name]
    fn = lambda X, P, M, S, cfg: base(B, X, P, M, S, cfg)
    if B.name == "jax":
        jax = B.jax
        if mode == "vg":
            f = jax.jit(jax.value_and_grad(fn), static_argnames="cfg")
        elif mode == "hess":
            f = jax.jit(jax.hessian(fn), static_argnames="cfg")
        else:
            f = jax.jit(fn, static_argnames="cfg")
    else:
        torch = B.torch
        per_cfg = {}
        for k in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k, 4096)

        def cfn(X, P, M, S, cfg):
            # one compiled closure per static config: its ints stay constants
            if cfg not in per_cfg:
                g = lambda X_, P_, M_, S_: fn(X_, P_, M_, S_, cfg)
                per_cfg[cfg] = (torch.compile(g, dynamic=False)
                                if os.environ.get("CORAL_TORCH_COMPILE", "1") == "1" else g)
            return per_cfg[cfg](X, P, M, S)
        if mode == "vg":
            def f(X, P, M, S, cfg):
                Xg = X.detach().clone().requires_grad_(True)
                v = cfn(Xg, P, M, S, cfg)
                (g,) = torch.autograd.grad(v, Xg)
                return v.detach(), g
        elif mode == "hess":
            def f(X, P, M, S, cfg):
                return torch.autograd.functional.hessian(lambda Z: fn(Z, P, M, S, cfg), X)
        else:
            def f(X, P, M, S, cfg):
                with torch.no_grad():
                    return cfn(X, P, M, S, cfg)
    _COMPILED[key] = f
    return f


_AOT = {}


def _aot_dir():
    d = os.environ.get("CORAL_JAX_AOT")
    if d:
        return d
    c = os.environ.get("CORAL_JAX_CACHE")
    return os.path.join(c, "aot") if c else None


def _resolve(B, jitted, name, dyn, static=None):
    """The compiled executable of jitted for these dynamic arguments (and static cfg): from
    this process, else deserialised from the store (no tracing, no lowering), else compiled
    and stored. JAX re-traces and re-lowers a program in every process even when its
    persistent cache holds the binary; a serialised executable skips both."""
    if B.name != "jax":
        return (lambda *a: jitted(*a, cfg=static)) if static is not None else jitted
    jax = B.jax
    leaves, tree = jax.tree_util.tree_flatten(dyn)
    sig = (name, repr(static), str(tree), tuple((tuple(x.shape), str(x.dtype)) for x in leaves),
           jax.__version__, jax.devices()[0].device_kind)
    key = hashlib.sha256(repr(sig).encode()).hexdigest()[:40]
    f = _AOT.get(key)
    if f is not None:
        return f
    from jax.experimental.serialize_executable import serialize, deserialize_and_load
    d = _aot_dir()
    path = os.path.join(d, key + ".pkl") if d else None
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                f = deserialize_and_load(*pickle.load(fh))
        except Exception:
            f = None
    if f is None:
        f = (jitted.lower(*dyn, cfg=static) if static is not None else jitted.lower(*dyn)).compile()
        if path:
            os.makedirs(d, exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump(serialize(f), fh)
            os.replace(tmp, path)
    _AOT[key] = f
    return f


def _numpy(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def _scalars_dyn(B, **kw):
    """Per-likelihood scalars as backend arrays (no per-call conversion)."""
    return {k: B.asarray(np.asarray(v, dtype=np.float64)) for k, v in kw.items()}


class _Sum:
    """A likelihood made of compiled parts (hit set, unhit set), each called only when its
    PMT set is not empty: value, value and gradient, Hessian. Each part's executable is
    resolved once per likelihood; the parts are added on the device and read back once."""

    def __init__(self, B, parts, n_x):
        self.B, self.parts, self.n_x = B, parts, n_x
        self.f = {m: [None] * len(parts) for m in ("", ":vg", ":hess")}

    def X(self, v):
        return self.B.asarray(np.asarray(v, dtype=np.float64))

    def fn(self, m, i):
        f = self.f[m][i]
        if f is None:
            k, P, M, S, c = self.parts[i]
            f = _resolve(self.B, _compiled(self.B, k + m), k + m, (self.X(np.zeros(self.n_x)), P, M, S), c)
            self.f[m][i] = f
        return f

    def _sum(self, m, v):
        X = self.X(v)
        out = None
        for i, (_, P, M, S, c) in enumerate(self.parts):
            r = self.fn(m, i)(X, P, M, S)
            out = r if out is None else (tuple(a + b for a, b in zip(out, r)) if isinstance(r, tuple) else out + r)
        return out

    def value(self, v):
        return float(self._sum("", v)) if self.parts else 0.0

    def value_and_grad(self, v):
        if not self.parts:
            return 0.0, np.zeros(self.n_x)
        val, grad = self._sum(":vg", v)
        return float(val), _numpy(grad).astype(np.float64)

    def hessian(self, v):
        if not self.parts:
            return np.zeros((self.n_x, self.n_x))
        return _numpy(self._sum(":hess", v)).astype(np.float64)



# ---------------------------------------------------------------- on-device Nelder-Mead

_NM = {}


def _nm_program(B, structure, idx, adaptive):
    """scipy.optimize's Nelder-Mead (_minimize_neldermead, bounds = clipping, xatol / fatol /
    maxiter) as one compiled program: a lax.while_loop whose every pass evaluates exactly one
    pending point (initial vertex, reflection, expansion, contraction, inside contraction or
    shrink vertex) in scipy's order, so a fit step is one dispatch and its evaluations never
    leave the device. structure = ((kind, cfg), ...) of the summed parts; idx = the fitted
    entries of the full parameter vector."""
    key = (B.name, structure, idx, adaptive)
    if key in _NM:
        return _NM[key]
    jax, jnp = B.jax, B.np
    lax = jax.lax
    N = len(idx)
    ia = np.asarray(idx)
    if adaptive:
        rho, chi, psi, sigma = 1.0, 1.0 + 2.0 / N, 0.75 - 1.0 / (2.0 * N), 1.0 - 1.0 / N
    else:
        rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    INIT, REFL, EXPN, CONT, CCON, SHRK = range(6)

    def objective(xf, xbase, datas):
        X = xbase.at[ia].set(xf)
        tot = 0.0
        for (kind, cfg), (P, M, S) in zip(structure, datas):
            tot = tot + _FUNCS[kind](B, X, P, M, S, cfg)
        return tot

    def run(sim0, xbase, lb, ub, xatol, fatol, maxiter, datas):
        sim = jnp.clip(jnp.where(sim0 > ub, 2.0 * ub - sim0, sim0), lb, ub)
        clip = lambda x: jnp.clip(x, lb, ub)

        def begin(st):
            # after a sort: scipy's loop test, then the convergence test, then the reflection
            sim, fsim = st["sim"], st["fsim"]
            conv = ((jnp.max(jnp.abs(sim[1:] - sim[0])) <= xatol)
                    & (jnp.max(jnp.abs(fsim[0] - fsim[1:])) <= fatol))
            stop = conv | (st["it"] >= maxiter)
            xbar = jnp.sum(sim[:-1], axis=0) / N
            xr = clip((1.0 + rho) * xbar - rho * sim[-1])
            return dict(st, done=stop, conv=conv, xbar=xbar, pend=xr, phase=jnp.int32(REFL))

        def end_iter(st):
            ind = jnp.argsort(st["fsim"], stable=True)
            st = dict(st, sim=st["sim"][ind], fsim=st["fsim"][ind], it=st["it"] + 1)
            return begin(st)

        def accept(st, x, f):
            return dict(st, sim=st["sim"].at[-1].set(x), fsim=st["fsim"].at[-1].set(f))

        def shrink(st):
            s0 = st["sim"][0]
            new = clip(s0 + sigma * (st["sim"] - s0)).at[0].set(s0)
            return dict(st, sim=new, k=jnp.int32(1), pend=new[1], phase=jnp.int32(SHRK))

        def ph_init(st, f):
            fsim = st["fsim"].at[st["k"]].set(f)
            k = st["k"] + 1
            more = k <= N
            st = dict(st, fsim=fsim, k=k, pend=st["sim"][jnp.minimum(k, N)])
            ind = jnp.argsort(fsim, stable=True)
            sorted_ = dict(st, sim=st["sim"][ind], fsim=fsim[ind])
            return lax.cond(more, lambda s: s, lambda s: begin(sorted_), st)

        def ph_refl(st, f):
            sim, fsim, xbar = st["sim"], st["fsim"], st["xbar"]
            st = dict(st, xr=st["pend"], fxr=f)
            xe = clip((1.0 + rho * chi) * xbar - rho * chi * sim[-1])
            xc = clip((1.0 + psi * rho) * xbar - psi * rho * sim[-1])
            xcc = clip((1.0 - psi) * xbar + psi * sim[-1])
            case = jnp.where(f < fsim[0], 0, jnp.where(f < fsim[-2], 1, jnp.where(f < fsim[-1], 2, 3)))
            return lax.switch(case, [
                lambda s: dict(s, pend=xe, phase=jnp.int32(EXPN)),
                lambda s: end_iter(accept(s, s["xr"], s["fxr"])),
                lambda s: dict(s, pend=xc, phase=jnp.int32(CONT)),
                lambda s: dict(s, pend=xcc, phase=jnp.int32(CCON))], st)

        def ph_expn(st, f):
            better = f < st["fxr"]
            x = jnp.where(better, st["pend"], st["xr"])
            return end_iter(accept(st, x, jnp.where(better, f, st["fxr"])))

        def ph_cont(st, f):
            return lax.cond(f <= st["fxr"], lambda s: end_iter(accept(s, s["pend"], f)), shrink, st)

        def ph_ccon(st, f):
            return lax.cond(f < st["fsim"][-1], lambda s: end_iter(accept(s, s["pend"], f)), shrink, st)

        def ph_shrk(st, f):
            fsim = st["fsim"].at[st["k"]].set(f)
            k = st["k"] + 1
            st = dict(st, fsim=fsim, k=k, pend=st["sim"][jnp.minimum(k, N)])
            return lax.cond(k <= N, lambda s: s, end_iter, st)

        def body(st):
            f = objective(st["pend"], xbase, datas)
            st = dict(st, fc=st["fc"] + 1)
            return lax.switch(st["phase"], [ph_init, ph_refl, ph_expn, ph_cont, ph_ccon, ph_shrk], st, f)

        z = jnp.zeros(N, dtype=jnp.float64)
        st0 = dict(sim=sim, fsim=jnp.full((N + 1,), jnp.inf), phase=jnp.int32(INIT), k=jnp.int32(0),
                   pend=sim[0], xr=z, fxr=jnp.float64(jnp.inf), xbar=z, it=jnp.int32(1), fc=jnp.int32(0),
                   done=jnp.bool_(False), conv=jnp.bool_(False))
        st = lax.while_loop(lambda s: ~s["done"], body, st0)
        return st["sim"][0], st["fsim"][0], st["it"], st["fc"], st["conv"]

    prog = jax.jit(run)
    _NM[key] = prog
    return prog


def device_nelder_mead(parts, x_full, idx, sim0, bounds, xatol, fatol, maxiter, adaptive=True):
    """Minimise the summed parts over the entries idx of x_full on the device, from the
    initial simplex sim0 (N+1, N); bounds = per fitted entry (lo, hi) or None.
    Returns (x (N,), fun, nit, nfev, success)."""
    B = backend("jax")
    structure = tuple((k, c) for k, P, M, S, c in parts)
    datas = tuple((P, M, S) for k, P, M, S, c in parts)
    lb = np.array([-np.inf if b is None or b[0] is None else b[0] for b in bounds], dtype=np.float64)
    ub = np.array([np.inf if b is None or b[1] is None else b[1] for b in bounds], dtype=np.float64)
    prog = _nm_program(B, structure, tuple(int(i) for i in idx), bool(adaptive))
    a = B.asarray
    args = (a(np.asarray(sim0, dtype=np.float64)), a(np.asarray(x_full, dtype=np.float64)), a(lb), a(ub),
            a(np.float64(xatol)), a(np.float64(fatol)), a(np.int32(maxiter)), datas)
    name = f"nm{structure!r}{tuple(int(i) for i in idx)}{bool(adaptive)}"
    x, f, it, fc, conv = _resolve(B, prog, name, args)(*args)
    it, fc = int(it), int(fc)
    return np.asarray(x, dtype=np.float64), float(f), it, fc, bool(it < maxiter)

