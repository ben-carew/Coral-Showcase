"""R48 v3 helper: Coral's own predicted DIRECT light (p.e.) per PMT at a track, from a frozen Coral checkout
and its production config (tuning tables loaded once through FQManager)."""
import os, sys, tempfile
import numpy as np


def init(code_src, config, infile, particle="e-"):
    sys.path.insert(0, code_src)
    from fitqun.FQManager import FQManager
    FQManager(config, [infile], tempfile.mkdtemp())
    import fitqun.globals as gbl
    import fitqun_tuning.corrections as corr
    from fitqun_tuning.cherenkov_profile import cprofs
    from fitqun_fit.predicted_charges import compute_predicted_photons
    cprof, curves = cprofs[particle], corr.geom_light.get(particle)

    def mu(x7, pos, ori, loc):
        nd, _ = compute_predicted_photons(np.asarray(x7, float), pos, ori, loc, cprof, np.zeros(len(pos), dtype=np.bool_))
        return gbl.qe*corr.geometry_light_scale(np.asarray(x7, float), curves)*nd
    return mu
