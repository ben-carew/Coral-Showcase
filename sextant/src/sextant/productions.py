"""Where the WCSim samples live, and how their events are numbered.

One registry, used by the runners to find input files and by the truth builder
to label them. Adding a sample means adding an entry here and nothing else.

## Event numbering

Every table in Sextant is keyed on a single global integer `event`, formed as

    event = file_index * stride + entry_index

so that a result merged from many files can be joined to truth without carrying
the filename around. The stride must exceed the number of events in any one
file or ids from adjacent files collide silently -- two different events sharing
one id, each overwriting the other in a merge, with nothing in the output to say
it happened.

The particle-gun productions hold exactly 100 events per file and the existing
Coral/fiTQun scripts hardcode a stride of 100, which is correct for them. **The
beam production holds 101**, so the same stride would alias event 100 of file 0
onto event 0 of file 1. Its stride is therefore 1000. `stride` and
`events_per_file` are separate fields for exactly this reason: the first is a
numbering choice that must never change once results exist, the second is a
property of the files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

#: Root of the shared beta-production area on CC-IN2P3. Note the "sensitvity"
#: typo -- it is in the real path, not a mistake here.
BETA_PRODUCTION = Path(
    "/sps/hyperk/common/hyperk.org/beta-production/prod_sensitvity_1"
)
PGUN_BASE = BETA_PRODUCTION / "pgun"
BEAM_BASE = (BETA_PRODUCTION / "beam"
             / "neut5.6.4_scale_t2k_vectors_csca.1.0"
             / "nuance_cnua.1.0.1"
             / "wcsim_v1.12.35_cwcs.1.0.1.0")

PGUN_DETECTOR = "HyperK_HybridmPMT_WithOD_Realistic"


@dataclass(frozen=True)
class Sample:
    """One reconstructable sample: a directory of WCSim files plus its labels."""

    name: str
    input_dir: Path
    #: Regex with a single capture group giving the file index, matched against
    #: the file *name*. Beam files are not zero-padded to a fixed width
    #: (..._0000_rwcs.root next to ..._14542_rwcs.root), so the index cannot be
    #: taken from a fixed slice.
    file_pattern: str
    #: Template used to build the Nth input filename, for runners that address
    #: files by index rather than globbing.
    file_template: str | None = None
    kind: str = "pgun"                  # 'pgun' | 'beam'
    #: True particle for a gun sample; None for beam (it varies per event).
    particle: str | None = None
    #: Beam: the charged-current lepton's flavour as a fit hypothesis (e- / mu-),
    #: which is what PID is scored against on CC single-ring events.
    lepton: str | None = None
    #: Official reconstruction shipped with the production, if any.
    reference_reco_dir: Path | None = None
    reference_reco_pattern: str | None = None
    notes: str = ""

    def file_index(self, path) -> int:
        """The file index of an input file *or* of its reconstruction.

        A sample's WCSim input and the official reconstruction of it are named
        differently (`..._0005_rwcs.root` against `..._0005_rfqn.root`) but
        carry the same index, and both need to map to the same global event ids
        -- so both patterns are tried rather than only the input one.
        """
        name = Path(path).name
        for pattern in (self.file_pattern, self.reference_reco_pattern):
            if not pattern:
                continue
            m = re.search(pattern, name)
            if m:
                return int(m.group(1))
        raise ValueError(f"{name} matches neither the input pattern "
                         f"({self.file_pattern}) nor the reconstruction pattern "
                         f"({self.reference_reco_pattern}) of sample "
                         f"'{self.name}'")

    def input_file(self, index: int) -> Path:
        if self.file_template is None:
            raise ValueError(f"sample '{self.name}' has no file_template; "
                             "list the directory instead")
        return self.input_dir / self.file_template.format(index=index)


@dataclass(frozen=True)
class Production:
    name: str
    events_per_file: int
    stride: int
    samples: dict = field(default_factory=dict)
    wcsim_version: str = ""
    notes: str = ""

    def sample(self, name: str) -> Sample:
        if name not in self.samples:
            raise KeyError(f"sample '{name}' not in production '{self.name}' "
                           f"(have: {', '.join(sorted(self.samples))})")
        return self.samples[name]

    def event_id(self, file_index, entry_index):
        return file_index * self.stride + entry_index

    def split_event_id(self, event):
        return divmod(int(event), self.stride)


def _pgun_sample(name, subdir, prefix, particle, run=0):
    """`run` is the production's 8-digit run block: each holds files 0000-0999."""
    block = f"{run:08d}"
    return Sample(
        name=name,
        input_dir=PGUN_BASE / "{prod_dir}" / PGUN_DETECTOR / subdir / "rwcs",
        file_pattern=rf"{re.escape(prefix)}_{block}-(\d+)_rwcs\.root",
        file_template=prefix + f"_{block}-" + "{index:04d}_rwcs.root",
        kind="pgun",
        particle=particle,
    )


def _resolve_pgun(production_dir: str, sample: Sample) -> Sample:
    """Substitute the production directory into a particle-gun sample path."""
    return Sample(**{**sample.__dict__,
                     "input_dir": Path(str(sample.input_dir).format(prod_dir=production_dir))})


# ---------------------------------------------------------------------------
# Particle-gun productions (single particle per event, 100 events/file)
# ---------------------------------------------------------------------------

_PGUN_V11222 = {
    "electron":      _pgun_sample("electron", "id_elpos", "pgun_hkfd_id_elpos", "e-"),
    "muon":          _pgun_sample("muon", "id_muneg", "pgun_hkfd_id_muneg", "mu-"),
    "muon-no-decay": _pgun_sample("muon-no-decay", "id_muneg_78off",
                                  "pgun_hkfd_id_muneg_78off", "mu-"),
    "pi0":           _pgun_sample("pi0", "id_pi0", "pgun_hkfd_id_pi0", "pi0"),
    "pion":          _pgun_sample("pion", "id_pipos", "pgun_hkfd_id_pipos", "pi+"),
}

_PGUN_V11229 = {
    "electron": _pgun_sample("electron", "id_elpos", "hkfd_id_elpos", "e-"),
    "muon":     _pgun_sample("muon", "id_muneg", "hkfd_id_muneg", "mu-"),
    "pi0":      _pgun_sample("pi0", "id_pi0", "hkfd_id_pi0", "pi0"),
    # "electron" is id_elpos = POSITRONS (truth Ipnu -11). True e-, and run block 1 of
    # e- / mu- (never used for any calibration):
    "elneg":    _pgun_sample("elneg", "id_elneg", "hkfd_id_elneg", "e-"),
    "elneg-r1": _pgun_sample("elneg-r1", "id_elneg", "hkfd_id_elneg", "e-", run=1),
    "muneg-r1": _pgun_sample("muneg-r1", "id_muneg", "hkfd_id_muneg", "mu-", run=1),
}

# ---------------------------------------------------------------------------
# Beam production (NEUT neutrino interactions, 101 events/file)
# ---------------------------------------------------------------------------

_BEAM_PREFIX = "t2ksk23c.fqv4r0.21bv2"


def _beam_sample(horn: str, channel: str) -> Sample:
    stem = f"{_BEAM_PREFIX}.{horn}.{channel}"
    return Sample(
        name=f"{horn}-{channel}",
        input_dir=BEAM_BASE / horn / "hkfd_v1" / channel / "rwcs",
        file_pattern=rf"{re.escape(stem)}_(\d+)_rwcs\.root",
        file_template=stem + "_{index:04d}_rwcs.root",
        kind="beam",
        particle=None,
        lepton="e-" if channel.startswith("nue") else "mu-",
        # The production ships an official fiTQun 6.4.4 reconstruction of the
        # same files. Consuming it directly gives a fiTQun reference arm with no
        # cluster time spent at all -- see io.fitqun.read_official_reco.
        reference_reco_dir=(BEAM_BASE / "fitqun_6.4.4_cfqn.1.0.1.0.0" / horn
                            / "hkfd_v1" / channel / "rfqn"),
        reference_reco_pattern=rf"{re.escape(stem)}_(\d+)_rfqn\.root",
        notes="NEUT beam MC; events carry a full final state, 1-2 WCSim triggers each.",
    )


_BEAM_CHANNELS = ("numu_x_numu", "numubar_x_numubar", "nue_x_nue", "nuebar_x_nuebar")

PRODUCTIONS = {
    "v1.12.22": Production(
        name="v1.12.22",
        events_per_file=100,
        stride=100,
        wcsim_version="v1.12.22",
        samples={k: _resolve_pgun("wcsim_v1.12.22_cwcs1.0", v)
                 for k, v in _PGUN_V11222.items()},
        notes="Particle guns; the only production with muon-no-decay, pi0 and pi+ (pion).",
    ),
    "v1.12.29": Production(
        name="v1.12.29",
        events_per_file=100,
        stride=100,
        wcsim_version="v1.12.29",
        samples={k: _resolve_pgun("wcsim_1.12.29_cwcs1.1", v)
                 for k, v in _PGUN_V11229.items()},
        notes="Particle guns, newer WCSim; electron, muon and pi0 (pi0 has no truth table yet).",
    ),
    "beam-fhc": Production(
        name="beam-fhc",
        events_per_file=101,
        # 1000, not 101: a round stride keeps event ids readable (file 37,
        # entry 12 -> 37012) and leaves room if a later beam production writes
        # more events per file.
        stride=1000,
        wcsim_version="v1.12.35",
        samples={c: _beam_sample("fhc", c) for c in _BEAM_CHANNELS},
        notes=("T2K-flux NEUT beam MC, forward horn current, HK FD geometry. "
               "Ships an official fiTQun 6.4.4 reconstruction alongside."),
    ),
}

DEFAULT_PRODUCTION = "v1.12.22"


def get(production: str) -> Production:
    if production not in PRODUCTIONS:
        raise KeyError(f"unknown production '{production}' "
                       f"(have: {', '.join(sorted(PRODUCTIONS))})")
    return PRODUCTIONS[production]


def resolve(production: str, sample: str) -> tuple:
    prod = get(production)
    return prod, prod.sample(sample)


def find_reference_reco(production: str, sample: str, limit=None, root=None, start=0) -> list:
    """The official reconstruction files shipped with a sample, index-sorted.

    Empty when the sample has none -- the particle-gun productions ship only
    WCSim output, so a fiTQun arm there has to be run rather than read.
    """
    prod, spec = resolve(production, sample)
    if spec.reference_reco_dir is None:
        return []
    directory = Path(spec.reference_reco_dir)
    if root is not None:
        directory = Path(root) / directory.relative_to(directory.anchor)
    if not directory.is_dir():
        return []
    pattern = re.compile(spec.reference_reco_pattern)
    found = []
    for path in directory.iterdir():
        m = pattern.match(path.name)
        if m and int(m.group(1)) >= start:
            found.append((int(m.group(1)), path))
    found.sort()
    paths = [p for _, p in found]
    return paths[:limit] if limit else paths


def find_reference_reco_for_indices(production: str, sample: str, indices,
                                    root=None) -> list:
    """Official reconstruction files for specific input-file indices.

    The shipped reconstruction is usually a *subset* of the production -- for
    the beam channels only about a third of the inputs have one, and their
    indices are scattered. Taking "the first N reconstruction files" therefore
    selects a different part of the sample than the run covered, and the two
    arms end up describing different events entirely.
    """
    wanted = {int(i) for i in indices}
    return [p for p in find_reference_reco(production, sample, root=root)
            if _index_of(production, sample, p) in wanted]


def _index_of(production: str, sample: str, path):
    _, spec = resolve(production, sample)
    try:
        return spec.file_index(path)
    except ValueError:
        return -1


def find_inputs(production: str, sample: str, limit=None, root=None,
                start: int = 0) -> list:
    """Input files for a sample, sorted by file index.

    `root` reroots the production tree, for a mirror of the sample held
    somewhere other than the CC-IN2P3 shared area. `start` skips files below
    that index -- the held-out validation files (80-99) are not the first N.
    """
    prod, spec = resolve(production, sample)
    directory = Path(spec.input_dir)
    if root is not None:
        directory = Path(root) / directory.relative_to(directory.anchor)
    if not directory.is_dir():
        raise FileNotFoundError(f"input directory not found: {directory}")
    pattern = re.compile(spec.file_pattern)
    found = []
    for path in directory.iterdir():
        m = pattern.match(path.name)
        if m and int(m.group(1)) >= start:
            found.append((int(m.group(1)), path))
    found.sort()
    paths = [p for _, p in found]
    return paths[:limit] if limit else paths


def find_inputs_for_events(production: str, sample: str, events, root=None) -> list:
    """Just the input files holding the given global event ids, in file order.

    A probe or a targeted re-read wants twenty events, not the sample. Walking
    the whole production to find them costs hundreds to thousands of file opens
    for no reason -- the event id already says which file each one is in.
    """
    prod, spec = resolve(production, sample)
    indices = sorted({prod.split_event_id(int(e))[0] for e in events})
    paths = []
    for index in indices:
        try:
            path = spec.input_file(index)
        except ValueError:
            continue
        if root is not None:
            path = Path(root) / Path(path).relative_to(Path(path).anchor)
        if Path(path).exists():
            paths.append(Path(path))
    return paths


def describe() -> "list[dict]":
    """Flat rows for `sextant productions` and for the report's appendix."""
    rows = []
    for prod in PRODUCTIONS.values():
        for sample in prod.samples.values():
            rows.append({
                "production": prod.name,
                "sample": sample.name,
                "kind": sample.kind,
                "particle": sample.particle or "(beam)",
                "events_per_file": prod.events_per_file,
                "stride": prod.stride,
                "official_reco": bool(sample.reference_reco_dir),
                "input_dir": str(sample.input_dir),
            })
    return rows
