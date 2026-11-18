"""What "the same configuration" means across two different codes.

Coral and fiTQun expose overlapping but differently-named settings. A controlled
comparison needs the shared ones to actually match, and this module is the
single place that says which settings *are* shared.

### Shared settings versus calibration schemes

Most entries are genuinely shared: the detector configuration, the time window,
whether the time likelihood is on, whether scattered light is used, the
ring-counting cut coefficients. Forcing those to agree is what makes a
comparison a comparison.

A few are **not** independently settable, and forcing them is actively wrong.
The photon yield and the water attenuation length are fitted *together* against
the observed light, so a (quantum efficiency, attenuation length, scattered-
light) triple is a self-consistent calibration and its members cannot be mixed
with another's. fiTQun ships this as two override files that differ in exactly
those three (`wscat`: 0.221 / 7000 / on; `noscat`: 0.240 / 7500 / off), and
Coral's production tuning is a third (0.176 / 9500 / time-PDF only). Copying
fiTQun's quantum efficiency onto Coral while leaving Coral's attenuation length
reintroduces a large momentum bias -- a "matched" configuration that measures
nothing but the mismatch.

So calibration constants are grouped into named `CalibrationScheme`s. An
experiment declares which scheme each arm uses; the tool verifies each arm is
internally consistent with the scheme it claims, records both schemes in the
report, and refuses to silently interleave them. `force=True` on an individual
shared parameter is available and honoured -- but a calibration constant can
only be changed by changing the whole scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: How a mapped parameter behaves.
SHARED = "shared"            # genuinely the same knob; safe to force
CALIBRATION = "calibration"  # part of a self-consistent set; scheme-level only
INFORMATIONAL = "info"       # recorded and compared, no equivalent to force


@dataclass(frozen=True)
class Parameter:
    """One concept, and how each code spells it."""

    concept: str
    coral: str | None            # 'section.key' in Coral's config.ini
    fitqun: str | None           # fiTQun runtime-parameter name
    kind: str = SHARED
    cast: str = "str"            # 'str' | 'float' | 'int' | 'bool'
    note: str = ""

    def describe(self) -> dict:
        return {"concept": self.concept, "coral": self.coral or "—",
                "fitqun": self.fitqun or "—", "kind": self.kind,
                "note": self.note}


#: The equivalence table. Anything not listed here has no counterpart and is
#: reported as algorithm-specific rather than quietly ignored.
PARAMETERS = [
    # -- detector and sample -------------------------------------------------
    Parameter("detector_config", None, "fiTQun.WCSimConfig", INFORMATIONAL,
              note="Coral selects the detector implicitly through its tuning-file "
                   "paths; fiTQun names it. Compared, not forced."),
    Parameter("pmt_type", None, "fiTQun.WCSimPMTType", INFORMATIONAL,
              note="Likewise implicit in Coral's tuning files."),
    Parameter("dark_rate", None, "fiTQun.DarkRateWCSim", INFORMATIONAL, "float",
              note="Coral has no dark-rate term in the likelihood."),

    # -- calibration set (never mix across schemes) --------------------------
    Parameter("quantum_efficiency", "physics.quantum_efficiency",
              "fiTQun.QEEffWCSim", CALIBRATION, "float",
              note="Fitted jointly with the attenuation length and the "
                   "scattered-light treatment. Scheme-level only."),
    Parameter("attenuation_length", "physics.attenuation_length",
              "fiTQun.WaterAttenuationLengthWCSim", CALIBRATION, "float",
              note="Coral works in cm, fiTQun in cm; both are the water "
                   "attenuation length. Scheme-level only."),
    Parameter("scattered_light_in_charge", "scattering.indirect_in_charge",
              "fiTQun.UseScatteredLight", CALIBRATION, "bool",
              note="fiTQun puts scattered light in the predicted charge; Coral's "
                   "production tuning routes it to the time PDF only. Part of "
                   "the calibration set."),
    Parameter("qe_correction_electron", "qe_corrections.e-",
              "fiTQun.QECorrElectron1RFit", CALIBRATION, "float"),
    Parameter("qe_correction_muon", "qe_corrections.mu-",
              "fiTQun.QECorrMuon1RFit", CALIBRATION, "float"),
    Parameter("charge_to_momentum", "charge_conversion_factors.e-",
              "fiTQun.fConvqp", CALIBRATION, "float",
              note="Seed-scale conversion from total charge to momentum; "
                   "detector-dependent."),

    # -- genuinely shared switches ------------------------------------------
    Parameter("use_scattering_table", "in_out.use_scattering_table",
              "fiTQun.UseScatteredLight", SHARED, "bool",
              note="Whether the scattering table is loaded at all."),
    Parameter("interpolate_scattering_table", None, "fiTQun.InterpScatTable",
              INFORMATIONAL, "bool",
              note="fiTQun can fall back to nearest-neighbour lookup; Coral "
                   "always interpolates."),
    Parameter("use_time_likelihood", "likelihood.use_time_pdf",
              "fiTQun.UseTimeLikelihood", SHARED, "bool"),
    Parameter("total_charge_constraint", None,
              "fiTQun.TotalChargeConstraintWeight", INFORMATIONAL, "float",
              note="fiTQun's forward total-charge Gaussian term. Weight 0 in the "
                   "HK-2024 tunes; Coral has no equivalent."),
    Parameter("bias_correction", None, "fiTQun.DoBiasCorrection", INFORMATIONAL,
              "bool",
              note="Empirical post-fit correction. Must be OFF for an honest "
                   "algorithm comparison."),

    # -- which fits run ------------------------------------------------------
    Parameter("do_electron_fit", "fits.particles", "fiTQun.DoElectron1RFit",
              SHARED, "bool",
              note="Coral lists hypotheses in one key; the mapping tests for "
                   "membership rather than equality."),
    Parameter("do_muon_fit", "fits.particles", "fiTQun.DoMuon1RFit", SHARED,
              "bool"),
    Parameter("do_pi0_fit", "pi0.do_pi0", "fiTQun.DoPi0Fit", SHARED, "bool"),
    Parameter("do_multiring_fit", "multiring.do_multiring", "fiTQun.DoMRFit",
              SHARED, "bool"),
    Parameter("max_rings", "multiring.max_rings", "fiTQun.DoMRFit",
              INFORMATIONAL, "int",
              note="fiTQun encodes 'fit up to n rings' in the SAME flag that "
                   "switches the fit on (DoMRFit=1 means on, n>1 means cap at "
                   "n), so its value is not comparable with Coral's separate "
                   "max_rings. Reported, not compared -- treating it as shared "
                   "raised a spurious 'Coral=6 vs fiTQun=true' disagreement."),

    # -- ring counting -------------------------------------------------------
    Parameter("rccut_a0", "multiring.rccut_a0", "fiTQun.RCCuta0WCSim", SHARED,
              "float",
              note="Coral's values are retuned on this detector and deliberately "
                   "differ from the fiTQun defaults; matching them makes the "
                   "ring-counting comparison a comparison of one tuning."),
    Parameter("rccut_a1", "multiring.rccut_a1", "fiTQun.RCCuta1WCSim", SHARED,
              "float"),
    Parameter("rccut_1r2r_electron", "multiring.rccut_1r2r_e",
              "fiTQun.RCCut1Ea0WCSim", SHARED, "float"),
    Parameter("rccut_1r2r_muon", "multiring.rccut_1r2r_mu",
              "fiTQun.RCCut1Mua0WCSim", SHARED, "float"),

    # -- prefit --------------------------------------------------------------
    Parameter("prefit_stages", "fits.prefit_sigma", "fiTQun.vtxPreFitNfitr",
              INFORMATIONAL,
              note="Coral lists the per-stage sigmas, fiTQun the stage count "
                   "plus separate sigma parameters. Compared, not forced."),
]

BY_CONCEPT = {p.concept: p for p in PARAMETERS}

#: fiTQun runtime-parameter name -> the concept that owns it. Needed when
#: WRITING a fiTQun override file, because the value has to be rendered in
#: fiTQun's spelling: its parser reads ints, so a `bool` concept carrying the
#: string "false" has to land as 0, not as the word. Several concepts can share
#: one fiTQun key (max_rings and do_multiring_fit both map to DoMRFit), so the
#: first wins and the rest are reachable through BY_CONCEPT.
BY_FITQUN = {}
for _p in PARAMETERS:
    if _p.fitqun and _p.fitqun not in BY_FITQUN:
        BY_FITQUN[_p.fitqun] = _p
del _p


def render_fitqun_value(key: str, value) -> str:
    """`value` in fiTQun's own spelling for `key`.

    fiTQun's parameter parser takes numbers; `[shared] do_muon_fit = false`
    written through verbatim would land as `fiTQun.DoMuon1RFit = false` and be
    read as 0 only by luck of the parser, so booleans are rendered explicitly.
    """
    text = str(value).strip()
    parameter = BY_FITQUN.get(key)
    if parameter is not None and parameter.cast == "bool":
        return "1" if text.lower() in ("1", "true", "yes", "on") else "0"
    return text

#: Concepts that select a fit hypothesis. fiTQun has one boolean each; Coral
#: names them all in a single comma-separated `fits.particles`, so several
#: concepts share one key. Both places that read a concept off a Coral config
#: have to know that -- otherwise the last concept written wins and
#: `do_muon_fit = false` silently overwrites `fits.particles = e-` with the
#: string "false", which configparser accepts and Coral reads as no hypothesis
#: at all.
HYPOTHESIS_LABEL = {"do_electron_fit": "e-", "do_muon_fit": "mu-"}


@dataclass(frozen=True)
class CalibrationScheme:
    """A self-consistent set of optical-calibration constants."""

    name: str
    description: str
    coral: dict = field(default_factory=dict)
    fitqun: dict = field(default_factory=dict)
    #: fiTQun parameter-override file that realises this scheme, if one exists.
    fitqun_override_file: str = ""

    def values_for(self, algorithm: str) -> dict:
        return dict(self.coral if algorithm == "coral" else self.fitqun)


#: The schemes that actually exist. Adding one means measuring it, not guessing:
#: the constants are fitted together against observed light.
SCHEMES = {
    "coral-production": CalibrationScheme(
        name="coral-production",
        description=("Coral's validated production tuning: the charge model is "
                     "calibrated for DIRECT light only, with scattered light "
                     "routed to the time PDF."),
        coral={"physics.quantum_efficiency": "0.176",
               "physics.attenuation_length": "9500",
               "scattering.indirect_in_charge": "False",
               "in_out.use_scattering_table": "True",
               # Stated rather than left absent: Coral's production tuning
               # applies no per-hypothesis QE correction, and writing the 1.0
               # explicitly is what puts it in the frozen config and the
               # report, so a comparison against fiTQun's 0.990 is visible
               # rather than inferred from a missing section.
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-masked-2026-09": CalibrationScheme(
        name="coral-masked-2026-09",
        description=(
            "Coral's 2026-09 electron-campaign tuning: the charge model is "
            "still calibrated for DIRECT light only, but PMTs hit OUTSIDE the "
            "sub-event window are masked out of the likelihood instead of being "
            "declared unhit, and the photon yield, attenuation length and "
            "scattered-light factor were re-fitted together for that. The four "
            "move as one calibration point and must not be mixed with "
            "coral-production's (0.176 / 9500 / 0.25): masking alone takes the "
            "electron momentum iqr68 from 6.92% to 6.02%, re-normalising the "
            "yield for it to 5.85%, and the attenuation/indirect_factor pair "
            "to 5.70% (paired bootstrap -0.269%, P(better) = 0.996). Validated "
            "on 3000 v1.12.29 electrons; muons neutral and PID a wash."),
        coral={"physics.quantum_efficiency": "0.1940",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "False",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               # Stated rather than left absent, as in coral-production: this
               # tuning applies no per-hypothesis QE correction either, and
               # writing the 1.0 puts it in the frozen config and the report.
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-hitscat-2026-10": CalibrationScheme(
        name="coral-hitscat-2026-10",
        description=(
            "coral-masked-2026-09 plus scattered light in the HIT PMTs' charge "
            "terms only (indirect_in_charge = hit, same 0.15 factor the time PDF "
            "uses); unhit PMTs stay direct-only. The photon yield is re-fitted "
            "for the extra predicted light (0.1940 -> 0.1894, restoring the "
            "momentum scale for both hypotheses), and Coral's config carries an "
            "e- light curve and vertex curves derived for it. Held out vs "
            "coral-masked-2026-09 (1984 mu-, 5916 e-): mu-eff at 99% e-eff "
            "+1.4 pt, contained p > 800 mis-IDs halved, resolution unchanged."),
        coral={"physics.quantum_efficiency": "0.1894",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "hit",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-hitscat-att7000": CalibrationScheme(
        name="coral-hitscat-att7000",
        description=(
            "coral-hitscat-2026-10 with fiTQun's HK water attenuation length "
            "(7000 cm instead of 8500). The momentum scale is restored by the e- "
            "light curve re-derived for it, not by the photon yield. R60 test."),
        coral={"physics.quantum_efficiency": "0.1894",
               "physics.attenuation_length": "7000",
               "scattering.indirect_in_charge": "hit",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-fqcharge-measure": CalibrationScheme(
        name="coral-fqcharge-measure",
        description=(
            "MEASUREMENT (R60): fiTQun's whole charge model in Coral -- scattered "
            "light in every PMT's charge at fiTQun's dose 0.5 and fiTQun's water "
            "attenuation length 7000 cm (with its angular response and charge pdf "
            "files set in the arm), Coral's own time pdf."),
        coral={"physics.quantum_efficiency": "0.1894",
               "physics.attenuation_length": "7000",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-hitscat-allscat-measure": CalibrationScheme(
        name="coral-hitscat-allscat-measure",
        description=(
            "MEASUREMENT ONLY (R28 step 4, 2026-10-19): coral-hitscat-2026-10's "
            "constants with scattered light in EVERY PMT's charge "
            "(indirect_in_charge = True, factor 0.15), for likelihood-shape "
            "dumps at the true track. Uncalibrated: never judge resolution or "
            "bias with it."),
        coral={"physics.quantum_efficiency": "0.1894",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-fqarr-measure": CalibrationScheme(
        name="coral-fqarr-measure",
        description=(
            "MEASUREMENT / CALIBRATION START (R28 step 5, 2026-10-19): fiTQun's "
            "arrangement in Coral -- scattered light in every PMT's charge at "
            "fiTQun's dose 0.5, with the absolute gate, dark noise and time-pdf "
            "floor set per arm; yield left at 0.1894 until re-calibrated."),
        coral={"physics.quantum_efficiency": "0.1894",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-eta-qe-bisect": CalibrationScheme(
        name="coral-eta-qe-bisect",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Isolates why the 2026-09-09 "
            "high-statistics run did not reproduce the campaign's vertex and "
            "angle (14.86 cm / 1.60 deg on confirm3k against 21.29 / 2.96 on "
            "the same events). Everything the two share is pinned here -- "
            "direct-only charge, attenuation 8500, indirect_factor 0.15, "
            "masking on -- and the TWO settings that actually differ between "
            "confirm3k's tuned arm and this run's improved arm are left free, "
            "so a 2x2 over them separates them: quantum_efficiency (0.1869 -> "
            "0.1940) and likelihood.angular_response_clip (absent, i.e. the old "
            "index wrap, -> True). Arms under this scheme are NOT comparable to "
            "each other as calibrated configurations; only the vertex/angle "
            "failure tail is being read."),
        coral={"physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "False",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-pid": CalibrationScheme(
        name="coral-pid",
        description=("Coral's PID-optimised tuning: scattered light in the "
                     "charge model, with the matching attenuation length and "
                     "per-hypothesis QE corrections."),
        coral={"physics.quantum_efficiency": "0.176",
               "physics.attenuation_length": "6000",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.28",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "0.90"},
    ),
    "coral-production-qe-scan": CalibrationScheme(
        name="coral-production-qe-scan",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Coral's production tuning with the "
            "photon yield SCANNED: everything else -- direct-only charge, "
            "attenuation 9500, no per-hypothesis QE correction -- is held at "
            "coral-production, and quantum_efficiency is deliberately left "
            "unpinned because a scheme pins one calibration point and locating "
            "a new one needs several. It exists because a change to the FIT "
            "SEQUENCE can move the momentum scale without touching the light "
            "model: pinning p in the joint step improves the scale-corrected "
            "momentum spread from 6.92% to 6.27% but leaves a -9.4% bias, "
            "since the joint step was what had been absorbing it. The yield is "
            "the knob that puts the scale back. Arms under this scheme are not "
            "comparable to each other as calibrated configurations; declare a "
            "real scheme once the yield is measured."),
        coral={"scattering.indirect_in_charge": "False",
               "in_out.use_scattering_table": "True",
               # Only the scattered-light treatment is pinned. The yield, the
               # attenuation length and the per-hypothesis corrections are all
               # left free because they are what this scheme exists to scan:
               # they are fitted TOGETHER, a global yield tuned on electrons
               # lands on the muon hypothesis untouched, and the attenuation
               # length is not a normalisation but the radial SHAPE of the
               # predicted light -- Coral's 9500 cm is a stand-in for the
               # scattered light it does not model, so it moves when the light
               # budget does.
               },
    ),
    "coral-scattered-hit-only-decoupled": CalibrationScheme(
        name="coral-scattered-hit-only-decoupled",
        description=(
            "Scattered light in the HIT charge terms at the physical dose 0.5, "
            "with the TIME PDF left at its validated 0.15 coupling and the "
            "photon yield re-fitted for the charge side. The correction to "
            "coral-scattered-hit-only, which reached the charge dose by raising "
            "indirect_factor and thereby gave the time PDF 3.3x its tuned dose: "
            "jhit's charge term improved 27% (iqr68 6.73 -> 4.91, exactly as the "
            "truth-geometry oracle predicted) while the total did not move "
            "(9.01 -> 9.21) because the charge->total degradation grew from "
            "+2.28 to +4.30 points. Here indirect_factor stays 0.15 and the "
            "charge side reaches 0.5 through indirect_charge_scale = 3.3333, so "
            "only one coupling moves."),
        coral={"physics.quantum_efficiency": "0.1859",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.15",
               "scattering.indirect_charge_scale": "3.3333333",
               "scattering.indirect_charge_hit_only": "True",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-fitqun-gate-scat": CalibrationScheme(
        name="coral-fitqun-gate-scat",
        description=(
            "DIAGNOSTIC. Coral in fiTQun's actual arrangement: an ABSOLUTE gate "
            "instead of a +-10 ns residual window, dark noise modelled both as "
            "a floor under every PMT's predicted charge and as a flat component "
            "of the time mixture, and scattered light in the charge model on "
            "ALL tubes -- which is correct here precisely because the wide gate "
            "makes scattered-only tubes HIT rather than unhit. The three go "
            "together: the gate admits ~100% of the scattered light instead of "
            "36%, so the charge model has to predict it; and a wide gate "
            "without the dark terms hands the fit late hits the time PDF cannot "
            "explain at -log(1e-20) ~ 46 units each. The photon yield is left "
            "at production's 0.1940 ON PURPOSE, so the arm's BIAS reads the "
            "light budget honestly rather than hiding it in a refit -- expect "
            "it nonzero, and compare widths scale-removed."),
        coral={"physics.quantum_efficiency": "0.1940",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.15",
               "scattering.indirect_charge_scale": "3.3333333",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-fitqun-gate-qe-scan": CalibrationScheme(
        name="coral-fitqun-gate-qe-scan",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Coral in fiTQun's arrangement -- "
            "absolute gate, dark noise, 1e-6 tpdf floor, scattered light in the "
            "charge on all tubes at dose 0.5 -- with the photon yield SCANNED. "
            "jfq showed fiTQun's own yield does not transfer: under "
            "coral-fitqun-faithful (QE 0.221) the momentum bias is -19.2%, and "
            "totmu/totq still lands at 0.931 because the FIT drags p down 19% "
            "until the light matches. fiTQun's QEEff multiplies fiTQun's "
            "Phi0*nphot*J; Coral's quantum_efficiency multiplies a different "
            "nphot and a different angular response, so the constant is not "
            "portable even though the scattering table is shared. Everything "
            "else is pinned at fiTQun's point; quantum_efficiency is left "
            "unpinned because a scheme pins one calibration point and this "
            "exists to locate a new one. Arms here are NOT comparable to each "
            "other as calibrated configurations -- only their momentum BIAS is, "
            "as the pointer to the right yield. Declare a real scheme once it "
            "is measured."),
        coral={"physics.attenuation_length": "7000",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "0.990",
               "qe_corrections.mu-": "0.990"},
    ),
    "coral-wide-gate-qe-scan": CalibrationScheme(
        name="coral-wide-gate-qe-scan",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Locates the photon yield that nulls "
            "the momentum bias for a WIDE-GATE arrangement, so that arms can be "
            "compared at matched calibration instead of at opposite biases. jocc "
            "measured +13.1% for the wide gate without scattered light and "
            "-12.5% with it, and a contrast between two arms mis-calibrated 13% "
            "in OPPOSITE directions says nothing -- fiTQun's own 1.12-point gain "
            "from scattered light is measured between a pair it re-fitted (QEEff "
            "0.240 noscat -> 0.221 wscat). Since nphotons is very nearly linear "
            "in p, the fitted momentum scales as 1/QE, so the correction is "
            "quantum_efficiency * (1 + bias). Everything except the yield is "
            "pinned; quantum_efficiency is left UNPINNED because a scheme pins "
            "one calibration point and this exists to find one. Arms here are "
            "NOT comparable as calibrated configurations until their biases "
            "land near zero. scattering.indirect_in_charge is likewise left to "
            "the arm, since the scan covers both the scattered and unscattered "
            "arrangements. Declare a real scheme once it is measured."),
        coral={"physics.attenuation_length": "8500",
               "likelihood.mask_out_of_window": "False",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-wide-gate-scat": CalibrationScheme(
        name="coral-wide-gate-scat",
        description=(
            "fiTQun's ARRANGEMENT of the indirect light on CORAL'S OWN optical "
            "point: wide absolute gate, scattered light in the charge model on "
            "all tubes at fiTQun's dose 0.5, yield 0.1940 and attenuation 8500 "
            "unchanged from coral-masked-2026-09. The yield is NOT borrowed from "
            "fiTQun, and that is the whole point. Measured at truth geometry on "
            "23 v1.12.29 electrons, total predicted light over total observed "
            "charge is 1.152 for the narrow gate with direct light only "
            "(production), 0.761 for the wide gate with direct light only, and "
            "1.017 for the wide gate with direct plus scattered at 0.5 -- "
            "so the wide gate WITH scattered light is the only one of the three "
            "whose light budget balances, and it balances at Coral's existing "
            "0.1940 (the yield that would null it exactly is 0.1907, 2% away). "
            "The narrow gate over-predicts by 15% because it collects only 66% "
            "of the charge while the model predicts all the direct light; the "
            "wide gate without scattered light under-predicts by 24% because it "
            "collects the scattered charge and never predicts it. "
            "coral-fitqun-faithful's 0.221 was fiTQun's constant for fiTQun's "
            "Phi0*nphot*J and gave -19.3% momentum bias here; this scheme exists "
            "so the gate can be tested without that transfer. Intended to run "
            "with an absolute gate, dark noise and the 1e-6 time-PDF floor, "
            "which are not calibration constants and are set per arm."),
        coral={"physics.quantum_efficiency": "0.1940",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_charge_hit_only": "False",
               "scattering.indirect_factor": "0.5",
               "likelihood.mask_out_of_window": "False",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-wide-gate-noscat": CalibrationScheme(
        name="coral-wide-gate-noscat",
        description=(
            "Control for coral-wide-gate-scat: the same wide absolute gate with "
            "scattered light kept OUT of the charge model, Coral's own yield and "
            "attenuation. NOT a calibrated point -- its light budget is 0.761 "
            "(it collects the scattered charge and never predicts it), which is "
            "exactly what makes it the right control. fiTQun's own gain from "
            "scattered light was measured as wscat against noscat at a FIXED "
            "wide gate (4.21% against 5.33%, P = 1.000); this is that contrast's "
            "other arm, and no previous Coral experiment has run it."),
        coral={"physics.quantum_efficiency": "0.1940",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "False",
               "scattering.indirect_factor": "0.15",
               "likelihood.mask_out_of_window": "False",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-fitqun-faithful": CalibrationScheme(
        name="coral-fitqun-faithful",
        description=(
            "Coral treating indirect light the way fiTQun does: fiTQun's own "
            "optical calibration (yield 0.221, attenuation 7000, per-hypothesis "
            "QE corrections 0.990) with scattered light in the charge model on "
            "ALL tubes at fiTQun's dose 0.5. These five move as one point -- "
            "fiTQun's authors fitted the yield and the attenuation length "
            "TOGETHER with scattered-light-in-charge, which is why turning the "
            "scattering on in its shipped pair also moves QEEff 0.240 -> 0.221 "
            "and the attenuation 7500 -> 7000. Mixing them with Coral's own "
            "narrow-gate point (0.1940 / 8500) is what made jgate uninterpretable. "
            "Intended to be run WITH an absolute gate, dark noise and the 1e-6 "
            "time-PDF floor, which are not calibration constants and so are set "
            "per arm. Scattered light on all tubes is right here for the same "
            "reason hit-only was right at 10 ns: the wide gate makes "
            "scattered-only tubes HIT, so predicting light on them is matched by "
            "real charge instead of being a -log P_unhit penalty. Bonus: 7000 cm "
            "is near the ~6000 the scattering table's ratio denominator wants, so "
            "it partly corrects the indirect-J defect for free."),
        coral={"physics.quantum_efficiency": "0.221",
               "physics.attenuation_length": "7000",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "0.990",
               "qe_corrections.mu-": "0.990"},
    ),
    "coral-scattered-hit-only": CalibrationScheme(
        name="coral-scattered-hit-only",
        description=(
            "Scattered light in the HIT charge terms only, at fiTQun's dose "
            "0.5, with the photon yield re-fitted for it. The point of the "
            "scheme is that these three move together. At Coral's +-10 ns gate "
            "a tube lit only by scattered light arrives late, falls outside the "
            "window and is declared unhit, so predicting light there is a "
            "-log P_unhit penalty with no observed charge to pay for it -- and "
            "most scattered light is on such tubes, which is why all thirteen "
            "earlier routes cost momentum in proportion to the dose. Measured "
            "at truth geometry (scattered_oracle.py, 69 v1.12.22 electrons, "
            "charge-term momentum minimum): 6.41% iqr68 direct-only, 4.17% with "
            "the model on hit tubes, 8.50% with it on unhit tubes alone; a "
            "perfect prediction on hit tubes reaches 3.47%, so the model keeps "
            "76% of the available gain. The yield is 0.1940 x 0.958 = 0.1859, "
            "set by scanning the CHARGE TERM'S OWN momentum minimum to zero -- "
            "not the final fitted bias, which is a compromise between a charge "
            "term wanting +9% and a time term wanting -16% and whose nulling is "
            "the methodological error under the earlier attempts."),
        coral={"physics.quantum_efficiency": "0.1859",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "scattering.indirect_charge_hit_only": "True",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-indirect-charge-acceptance": CalibrationScheme(
        name="coral-indirect-charge-acceptance",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Reads which WINDOW ACCEPTANCE the "
            "charge term needs for scattered light, on the 2026-09 masked "
            "point, with the indirect light given its own attenuation length. "
            "Measured on jconv (994 v1.12.29 electrons, dump_momentum_scan at "
            "truth geometry): the direct-only charge model predicts 0.905 of "
            "the in-window charge -- 9.5% SHORT -- and its momentum minimum "
            "sits at +9.2%, the fit buying the missing light with p. Scattered "
            "light is the missing component, so the question is how much of it "
            "the window admits. Two different quantities are called 'the "
            "scattered acceptance' and they differ by nearly 2x at 10 ns: "
            "per-PE (18.0%, the fraction of scattered photoelectrons whose own "
            "arrival is in the window) and per-TUBE (34.5%, the scattered "
            "photoelectrons on tubes the window admits). The CHARGE term wants "
            "the per-tube number, because an admitted tube contributes its "
            "whole digitised charge including the scattered photoelectrons the "
            "digitiser integrated -- the per-PE number answers an occupancy "
            "question, not a charge one. wacc_recal corrected TOWARD per-PE and "
            "failed; this tests the other direction. Everything that would "
            "confound that reading is pinned, quantum_efficiency included, "
            "because the arms are compared on totmu/totq and on the charge "
            "term's own momentum minimum, not on a delivered width. Arms under "
            "this scheme are NOT calibrated configurations; once the acceptance "
            "is known, re-fit the yield for it and declare a real scheme."),
        coral={"physics.quantum_efficiency": "0.1940",
               "physics.attenuation_length": "8500",
               "scattering.indirect_in_charge": "True",
               "scattering.indirect_factor": "0.5",
               "likelihood.mask_out_of_window": "True",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "coral-scattered-charge-scan": CalibrationScheme(
        name="coral-scattered-charge-scan",
        description=(
            "DIAGNOSTIC, NOT A CALIBRATION. Coral with the scattered light "
            "moved INTO the predicted charge and the sub-event hit window "
            "opened to match -- the arrangement fiTQun uses -- while the "
            "photon yield is SCANNED to find where the normalisation lands. "
            "quantum_efficiency is deliberately left unpinned here: a scheme "
            "pins one calibration point, and locating a new one needs several. "
            "Arms under this scheme are therefore NOT comparable to each other "
            "as calibrated configurations; only their momentum BIAS is being "
            "read, as the pointer to the right yield. "
            "The two halves cannot be moved separately, which is why they are "
            "moved together: Coral's 10 ns window is compensation for a "
            "direct-only charge model, so widening it alone drives the "
            "momentum bias from +1.4% to +18.3%, and turning "
            "indirect_in_charge on alone over-predicts the light by ~18% "
            "(recorded CLOSED as 'harmful at every factor' -- that scan moved "
            "one member of this triple and left the rest). Once the yield is "
            "measured, declare a real scheme with it and re-validate."),
        coral={"physics.attenuation_length": "9500",
               "scattering.indirect_in_charge": "True",
               # Unchanged at its validated value on purpose. indirect_factor
               # scales n_indirect, which feeds the TIME PDF as well as the
               # charge, so moving it would move two couplings at once; the
               # time PDF divides by qe, so QE is the pure charge-side scale.
               "scattering.indirect_factor": "0.25",
               "in_out.use_scattering_table": "True",
               "qe_corrections.e-": "1.0",
               "qe_corrections.mu-": "1.0"},
    ),
    "fitqun-hk2024-wscat-on-coral": CalibrationScheme(
        name="fitqun-hk2024-wscat-on-coral",
        description=("fiTQun's HK-2024 wscat optical calibration applied to "
                     "Coral: the same photon yield, attenuation length and "
                     "scattered-light-in-charge treatment, so the two codes "
                     "predict light from the same physics and the comparison "
                     "isolates the algorithms. This is NOT Coral's own "
                     "validated tuning (see coral-production), and Coral's "
                     "momentum bias under it has not been separately "
                     "calibrated -- it is the price of a controlled "
                     "comparison."),
        coral={"physics.quantum_efficiency": "0.221",
               "physics.attenuation_length": "7000",
               "scattering.indirect_in_charge": "True",
               "in_out.use_scattering_table": "True",
               # fiTQun's QECorrElectron1RFit / QECorrMuon1RFit. Both the
               # sandbox and multiring-fit branches read [qe_corrections], so
               # this part of the calibration is matchable; upstream dev does
               # not, which is why the dev comparison had to leave it out.
               "qe_corrections.e-": "0.990",
               "qe_corrections.mu-": "0.990"},
    ),
    "fitqun-hk2024-wscat": CalibrationScheme(
        name="fitqun-hk2024-wscat",
        description="fiTQun official HyperK 2024 tune, scattered light on.",
        fitqun={"fiTQun.QEEffWCSim": "0.221",
                "fiTQun.WaterAttenuationLengthWCSim": "7000.",
                "fiTQun.UseScatteredLight": "1",
                # Read out of the beam production's own job log, so the table
                # compares against what actually ran rather than a default.
                "fiTQun.QECorrElectron1RFit": "0.990",
                "fiTQun.QECorrMuon1RFit": "0.990"},
        fitqun_override_file="Official_HyperK_2024_wscat.parameters.dat",
    ),
    "fitqun-hk2024-wscat-1remu": CalibrationScheme(
        name="fitqun-hk2024-wscat-1remu",
        description=("fiTQun official HyperK 2024 tune, scattered light ON, "
                     "single-ring e/mu only. Same calibration point as "
                     "fitqun-hk2024-wscat; the _1Remu override drops the "
                     "multi-ring and pi0 fits, which is what makes it ~101 s/"
                     "event instead of ~20 min."),
        fitqun={"fiTQun.QEEffWCSim": "0.221",
                "fiTQun.WaterAttenuationLengthWCSim": "7000.",
                "fiTQun.UseScatteredLight": "1",
                "fiTQun.QECorrElectron1RFit": "0.990",
                "fiTQun.QECorrMuon1RFit": "0.990"},
        fitqun_override_file="Official_HyperK_2024_wscat_1Remu.parameters.dat",
    ),
    "fitqun-hk2024-noscat-1remu": CalibrationScheme(
        name="fitqun-hk2024-noscat-1remu",
        description=("fiTQun official HyperK 2024 tune, scattered light OFF, "
                     "single-ring e/mu only. The matched partner of "
                     "fitqun-hk2024-wscat-1remu: the two shipped override files "
                     "differ in exactly three lines -- UseScatteredLight 1/0, "
                     "QEEffWCSim 0.221/0.240, WaterAttenuationLength 7000/7500 "
                     "-- because fiTQun's authors re-fitted the yield and the "
                     "attenuation length when they turned scattering on. Each "
                     "arm therefore sits at its own calibration point, which is "
                     "the fair comparison."),
        fitqun={"fiTQun.QEEffWCSim": "0.240",
                "fiTQun.WaterAttenuationLengthWCSim": "7500.",
                "fiTQun.UseScatteredLight": "0",
                "fiTQun.QECorrElectron1RFit": "0.990",
                "fiTQun.QECorrMuon1RFit": "0.990"},
        fitqun_override_file="Official_HyperK_2024_noscat_1Remu.parameters.dat",
    ),
    "fitqun-hk2024-wscat-1remu-scatoff": CalibrationScheme(
        name="fitqun-hk2024-wscat-1remu-scatoff",
        description=("DIAGNOSTIC, NOT A CALIBRATION. The scattering-ON tuning "
                     "with UseScatteredLight forced to 0 and NOTHING else "
                     "changed. Separates the two things the shipped wscat/"
                     "noscat pair moves at once: this arm removes the muscat "
                     "term while holding the yield and attenuation length at "
                     "the scattering-on point, so wscat -> this measures what "
                     "the scattered-light MODEL contributes, and this -> noscat "
                     "measures what the re-tuning contributes. Deliberately "
                     "mis-calibrated -- its total light is ~8% low by "
                     "construction -- so read its BIAS as a diagnostic, not as "
                     "a configuration anyone should run."),
        fitqun={"fiTQun.QEEffWCSim": "0.221",
                "fiTQun.WaterAttenuationLengthWCSim": "7000.",
                "fiTQun.UseScatteredLight": "0",
                "fiTQun.QECorrElectron1RFit": "0.990",
                "fiTQun.QECorrMuon1RFit": "0.990"},
        fitqun_override_file="Official_HyperK_2024_wscat_1Remu.parameters.dat",
    ),
    "fitqun-hk2024-noscat": CalibrationScheme(
        name="fitqun-hk2024-noscat",
        description="fiTQun official HyperK 2024 tune, scattered light off.",
        fitqun={"fiTQun.QEEffWCSim": "0.240",
                "fiTQun.WaterAttenuationLengthWCSim": "7500.",
                "fiTQun.UseScatteredLight": "0"},
        fitqun_override_file="Official_HyperK_2024_noscat.parameters.dat",
    ),
}

#: `scheme = config`: the arm takes the calibration constants of the config or
#: preset it runs, as committed on its branch; Sextant overrides none of them.
CONFIG_SCHEME = "config"

#: A Coral arm that names no scheme runs its config as committed. Until
#: 2026-11-14 the default was "coral-production" -- the JULY tuning (0.176 /
#: 9500 / direct only), silently written over whatever the branch's config or
#: preset carried.
DEFAULT_SCHEME = {"coral": CONFIG_SCHEME, "fitqun": "fitqun-hk2024-wscat"}


def scheme_conflicts(scheme_name: str, config_values: dict) -> list:
    """(key, config value, scheme value) where a Coral scheme would CHANGE the
    value the arm's own config or preset sets. Keys the config does not set
    are not conflicts (the scheme only states the code default there)."""
    scheme = SCHEMES.get(scheme_name)
    if scheme is None:
        return []
    return [(key, config_values[key], want) for key, want in scheme.coral.items()
            if key in config_values and not _equal(config_values[key], want)]


# ---------------------------------------------------------------------------

@dataclass
class MappingIssue:
    severity: str          # 'error' | 'warning' | 'info'
    concept: str
    message: str

    def __str__(self):
        return f"[{self.severity.upper():7s}] {self.concept}: {self.message}"


def verify_scheme(algorithm: str, scheme_name: str, values: dict) -> list:
    """Check an arm's calibration constants against the scheme it declares."""
    if scheme_name not in SCHEMES:
        return [MappingIssue("error", "scheme",
                             f"unknown calibration scheme '{scheme_name}' "
                             f"(have: {', '.join(sorted(SCHEMES))})")]
    scheme = SCHEMES[scheme_name]
    expected = scheme.values_for(algorithm)
    if not expected:
        return [MappingIssue("warning", "scheme",
                             f"scheme '{scheme_name}' defines no values for "
                             f"{algorithm}; nothing to verify")]
    issues = []
    for key, want in expected.items():
        got = values.get(key)
        if got is None:
            issues.append(MappingIssue(
                "warning", key,
                f"scheme '{scheme_name}' expects {key}={want} but the config "
                f"does not set it (the code's own default will apply)"))
        elif not _equal(got, want):
            issues.append(MappingIssue(
                "error", key,
                f"scheme '{scheme_name}' expects {key}={want}, config has "
                f"{got}. These constants are fitted together — changing one "
                f"alone reintroduces a momentum bias. Change the scheme, or "
                f"declare a new one."))
    return issues


def compare_arms(arm_settings: dict) -> "tuple[list, list]":
    """Compare every arm's mapped settings; returns (rows, issues).

    `arm_settings` maps an arm label to
    {'algorithm': 'coral'|'fitqun', 'scheme': name, 'values': {key: value}}.
    """
    rows, issues = [], []
    labels = list(arm_settings)

    for parameter in PARAMETERS:
        row = {"concept": parameter.concept, "kind": parameter.kind,
               "coral_key": parameter.coral or "—",
               "fitqun_key": parameter.fitqun or "—"}
        seen = {}
        for label in labels:
            arm = arm_settings[label]
            key = parameter.coral if arm["algorithm"] == "coral" else parameter.fitqun
            value = arm["values"].get(key) if key else None
            row[label] = "—" if value is None else str(value)
            if value is not None:
                seen[label] = _normalise(parameter, value,
                                         algorithm=arm["algorithm"])
        distinct = {v for v in seen.values() if v is not None}
        row["agrees"] = len(distinct) <= 1
        rows.append(row)

        if parameter.kind == SHARED and len(distinct) > 1:
            issues.append(MappingIssue(
                "warning", parameter.concept,
                f"arms disagree on a shared setting: "
                + ", ".join(f"{k}={v}" for k, v in seen.items())))

    schemes = {label: arm.get("scheme") for label, arm in arm_settings.items()}
    for label, arm in arm_settings.items():
        if arm.get("scheme") and arm["scheme"] != CONFIG_SCHEME:
            issues.extend(verify_scheme(arm["algorithm"], arm["scheme"],
                                        arm["values"]))
    if len(set(s for s in schemes.values() if s)) > 1:
        # Different scheme *names* do not necessarily mean different physics:
        # a scheme that ports one code's optical constants onto the other is
        # named differently but agrees on every mapped value, and saying "these
        # are two different calibrations" of that would be plainly wrong.
        calibration_rows = [r for r in rows if r["kind"] == CALIBRATION]
        matched = all(r["agrees"] for r in calibration_rows)
        named = ", ".join(f"{k}: {v}" for k, v in schemes.items() if v)
        if matched:
            issues.append(MappingIssue(
                "info", "scheme",
                f"arms declare different schemes ({named}) but agree on every "
                f"mapped calibration constant, so both codes predict light "
                f"from the same optical physics. The comparison isolates the "
                f"algorithms."))
        else:
            issues.append(MappingIssue(
                "info", "scheme",
                f"arms use different calibration schemes ({named}). That is "
                f"expected when comparing each code at its own validated "
                f"tuning; it means the comparison is of two calibrated "
                f"algorithms, not of one algorithm under two implementations."))
    return rows, issues


def _normalise(parameter: Parameter, value, algorithm: str = ""):
    text = str(value).strip()
    if algorithm == "coral" and parameter.concept in HYPOTHESIS_LABEL:
        # Coral's fits.particles is a LIST, so the question is membership, not
        # equality: "e-, mu-" means do_electron_fit AND do_muon_fit are on.
        # Reading it as a boolean makes every hypothesis look switched off and
        # raises a disagreement against fiTQun that does not exist.
        return HYPOTHESIS_LABEL[parameter.concept] in [
            part.strip() for part in text.split(",")]
    if parameter.cast == "bool":
        return text.lower() in ("1", "true", "yes", "on")
    if parameter.cast in ("float", "int"):
        try:
            return float(text.rstrip("."))
        except ValueError:
            return text
    return text.lower()


def _equal(a, b) -> bool:
    try:
        return abs(float(str(a).rstrip(".")) - float(str(b).rstrip("."))) < 1e-9
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def describe_table() -> list:
    """The mapping itself, for the report appendix."""
    return [p.describe() for p in PARAMETERS]
