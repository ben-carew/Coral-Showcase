"""L-BFGS-B fit of part of a flat parameter vector on the likelihood's exact gradient: the
alternative to the multi-ring and pi0 simplex fits ([likelihood] multi_ring_minimiser = lbfgs),
or the polish after a loose simplex (= hybrid), JAX backend."""
import numpy as np
from scipy.optimize import minimize


def lbfgs_fit(value_and_grad, to_full, pull, x0, bounds, steps, gtol=1e-2, maxiter=500, precondition=None):
    """Minimise value(to_full(v)) over v from x0, in units of the simplex steps.

    value_and_grad(x_full) -> (f, df/dx_full); to_full(v) -> x_full; pull(g_full, v) -> df/dv
    (the fitted entries of g_full, plus any chain-rule term of a derived parameter).
    precondition: None | "diag" (each entry scaled by its curvature, 1/sqrt(H_ii)) | "full"
    (whitened by H^-1/2, unbounded, clipped to the bounds at the end), H from central differences
    of the gradient at x0 (2N gradient calls). Returns (success, f, v, nfev) at the lowest finite
    point evaluated."""
    x0 = np.asarray(x0, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    n = len(x0)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    best = [np.inf, x0]
    nfev0 = 0
    P = np.eye(n)
    if precondition in ("diag", "full"):
        gs = lambda v: np.asarray(pull(np.asarray(value_and_grad(to_full(v))[1], dtype=np.float64), v)) * s
        H = np.zeros((n, n))
        h = 1e-3
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            H[:, i] = (gs(np.clip(x0 + s * e, lo, hi)) - gs(np.clip(x0 - s * e, lo, hi))) / (2 * h)
        nfev0 = 2 * n
        H = 0.5 * (H + H.T)
        if np.all(np.isfinite(H)):
            if precondition == "diag":
                P = np.diag(1.0 / np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-6)))
            else:
                w, Q = np.linalg.eigh(H)
                w = np.maximum(np.abs(w), 1e-6 * np.max(np.abs(w)))
                P = Q / np.sqrt(w)[None, :]
        else:
            precondition = None

    def f(u):
        v = x0 + s * (P @ u)
        if precondition == "full":
            v = np.clip(v, lo, hi)
        val, g = value_and_grad(to_full(v))
        if not np.isfinite(val):
            return 1e30, np.zeros_like(u)
        if val < best[0]:
            best[0], best[1] = float(val), v
        return float(val), P.T @ (np.asarray(pull(np.asarray(g, dtype=np.float64), v)) * s)

    if precondition == "full":
        ub = None
    else:
        d = np.diag(P)
        ub = [(None if lo_ is None else (lo_ - x0[i]) / (s[i] * d[i]), None if hi_ is None else (hi_ - x0[i]) / (s[i] * d[i]))
              for i, (lo_, hi_) in enumerate(bounds)]
    r = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", bounds=ub,
                 options=dict(maxiter=maxiter, gtol=gtol, ftol=0.0))
    return bool(r.success), best[0], best[1], int(r.nfev) + nfev0


def newton_fit(value, value_and_grad, hessian, to_full, jac, x0, bounds, steps, maxiter=20):
    """Damped Newton (Levenberg-Marquardt, damping in proportion to each curvature, on the exact
    Hessian) from x0, in units of the simplex
    steps: the polish of a simplex that has found the basin. jac(v) -> d x_full / d v (the
    selection of the fitted entries, plus any derived parameter's row); a step is kept only if
    the value falls. Returns (success, f, v, nfev) with nfev counting value, gradient and
    Hessian calls."""
    v = np.asarray(x0, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    f, g = value_and_grad(to_full(v))
    nfev = 1
    if not np.isfinite(f):
        return False, np.inf, v, nfev
    lam, ok = None, False
    for _ in range(maxiter):
        J = jac(v)
        H = J.T @ np.asarray(hessian(to_full(v)), dtype=np.float64) @ J
        gv = J.T @ np.asarray(g, dtype=np.float64)
        nfev += 1
        Hs, gs = s[:, None] * H * s[None, :], s * gv
        if not (np.all(np.isfinite(Hs)) and np.all(np.isfinite(gs))):
            break
        try:                       # converged when the undamped Newton decrement 1/2 g H^-1 g is small
            L = np.linalg.cholesky(Hs)
            w = np.linalg.solve(L, gs)
            if 0.5 * float(w @ w) < 1e-3:
                ok = True
                break
        except np.linalg.LinAlgError:
            pass                   # not positive definite here: keep stepping
        # Marquardt: damp each direction by its own curvature (the time and the momentum curvatures
        # differ by ~1e5 in step units, a uniform damping would freeze the soft directions)
        D = np.diag(np.maximum(np.abs(np.diag(Hs)), 1e-9))
        lam = 1e-3 if lam is None else lam
        moved = False
        for _ in range(6):
            try:
                du = -np.linalg.solve(Hs + lam * D, gs)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            vn = np.clip(v + s * du, lo, hi)
            fn = value(to_full(vn))
            nfev += 1
            if np.isfinite(fn) and fn < f:
                moved, lam = True, lam / 10.0
                break
            lam *= 10.0
        if not moved:
            break                  # no descent at any damping
        v = vn
        f, g = value_and_grad(to_full(v))
        nfev += 1
    return ok, float(f), v, nfev
