"""CNN identity association and track history helpers."""


def cnn_build_association_entries(
    cnn_identity_cache, cnn_track_history, frame_idx, n_det, N
):
    """Return (det_classes, track_identities, frame_preds) for CNN identity assigner fields.

    Returns three values ready for use by the caller.  Either cache argument
    may be None; in that case (None, None, None) is returned and the caller
    should skip the update.  The returned *frame_preds* list can be passed
    directly to cnn_update_track_history to avoid a second cache.load() call.

    For flat (single-factor) histories the first factor value is extracted so
    that track_identities remains a plain list[str | None] compatible with the
    assigner's cost-matrix overlay.
    """
    if cnn_identity_cache is None or cnn_track_history is None:
        return None, None, None
    frame_preds = cnn_identity_cache.load(frame_idx)
    det_classes = [None] * n_det
    for pred in frame_preds:
        if pred.det_index < n_det:
            det_classes[pred.det_index] = pred.class_name
    identity_dict = cnn_track_history.build_track_identity_list()
    # Extract the first factor for each slot; default to None for unseen tracks.
    track_identities = [
        (identity_dict[i][0] if i in identity_dict else None) for i in range(N)
    ]
    return det_classes, track_identities, frame_preds


def cnn_update_track_history(cnn_track_history, frame_preds, frame_idx, N, rows, cols):
    """Record CNN predictions for the matched (track, detection) pairs.

    *frame_preds* should be the list already loaded by
    cnn_build_association_entries so that the cache is not read twice per
    frame.
    """
    if cnn_track_history is None or frame_preds is None:
        return
    pred_by_det = {pred.det_index: pred for pred in frame_preds}
    for r, c in zip(rows, cols):
        pred = pred_by_det.get(c)
        if pred is not None and pred.class_name is not None:
            conf = pred.confidences[0] if pred.confidences else 0.0
            cnn_track_history.record(
                track_id=r,
                class_names=(pred.class_name,),
                confidences=(conf,),
            )
