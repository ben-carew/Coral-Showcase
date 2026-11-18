"""Excerpt from Coral's event loop (src/fitqun/fitqun.py): where the segmenter is loaded and
where each network ring is fitted as a single ring. Not runnable on its own."""

# --- once per job
# ML ring segmentation of the primary sub-event (optional, [ml])
segmenter = time_offset = None
if manager.ml_segmentation:
    segmenter, time_offset = load_segmenter(manager.ml_config)

# --- per sub-event, after the likelihood fits
# ML segmentation: fit every ring's hits as a single ring, with the
# other rings' PMTs masked (their light is not this ring's absence)
event_data.pop("rings", None)
if segmenter is not None and i_subevent == 0:
    t0_rings = time.time()
    event_data["rings"] = []
    for ring_id, ring_hits in segment_rings(segmenter, sub_event_hits, time_offset):
        if len(ring_hits) < 8:
            continue
        _, ring_seed, _, _, ring_rivals = likelihood_fit_seed(
            ring_hits, prefit_sigma, cn, max_travel_time, False,
            g_over_nhits_cut, sigma_grid_search, t_seed=seed["t"],
            direction_seed=manager.direction_seed,
            n_rival_axes=manager.n_rival_axes)
        ring_seed = dict(zip(full_fit_vars_names, ring_seed))
        ring_bounds = dict(bounds, t=(ring_seed["t"] - max_travel_time,
                                      ring_seed["t"] + max_travel_time))
        ring_mask = np.union1d(
            mask_tubeids if mask_tubeids is not None else np.array([], dtype=int),
            np.setdiff1d(sub_event_hits.tubeid.unique(), ring_hits.tubeid.unique()))
        ring_fits, _ = fit_single_ring(
            manager, ring_hits, ring_seed, ring_rivals, ring_mask, None,
            ring_bounds, event, i_subevent, None, None, dump=False)
        event_data["rings"].append((ring_id, len(ring_hits),
                                    float(ring_hits.q.sum()), ring_fits))
    logger.info("ML segmentation: %d rings fitted", len(event_data["rings"]))
    total_likelihood_time += time.time() - t0_rings

