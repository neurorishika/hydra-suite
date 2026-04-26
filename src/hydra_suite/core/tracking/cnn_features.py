"""CNN identity association and track history helpers."""


def cnn_build_association_entries(
    cnn_identity_cache, cnn_track_history, frame_idx, n_det, N
):
    """Return (det_classes, track_identities, frame_preds) for CNN identity assigner fields.

    Returns three values ready for use by the caller.  Either cache argument
    may be None; in that case (None, None, None) is returned and the caller
    should skip the update.  The returned *frame_preds* list can be passed
    directly to cnn_update_track_history to avoid a second cache.load() call.

    Flat (single-factor) histories use the first factor value. Multi-factor
    histories propagate tuples so multihead predictions stay intact.
    """
    if cnn_identity_cache is None or cnn_track_history is None:
        return None, None, None
    frame_preds = cnn_identity_cache.load(frame_idx)
    history_factors = tuple(getattr(cnn_track_history, "factor_names", ()) or ())
    multi_factor_history = len(history_factors) > 1
    det_classes = [None] * n_det
    for pred in frame_preds:
        if pred.det_index < n_det:
            if multi_factor_history:
                det_classes[pred.det_index] = tuple(pred.class_names)
            else:
                det_classes[pred.det_index] = (
                    pred.class_names[0] if pred.class_names else None
                )
    identity_dict = cnn_track_history.build_track_identity_list()
    if multi_factor_history:
        track_identities = [
            (tuple(identity_dict[i]) if i in identity_dict else None)
            for i in range(N)
        ]
    else:
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
    history_factors = tuple(getattr(cnn_track_history, "factor_names", ()) or ())
    history_size = len(history_factors)
    pred_by_det = {pred.det_index: pred for pred in frame_preds}
    for r, c in zip(rows, cols):
        pred = pred_by_det.get(c)
        if pred is not None:
            if history_size > 1:
                class_names = tuple(
                    pred.class_names[idx] if idx < len(pred.class_names) else None
                    for idx in range(history_size)
                )
                confidences = tuple(
                    float(pred.confidences[idx])
                    if idx < len(pred.confidences)
                    else 0.0
                    for idx in range(history_size)
                )
            else:
                class_names = ((pred.class_names[0] if pred.class_names else None),)
                confidences = (
                    float(pred.confidences[0]) if pred.confidences else 0.0,
                )
            if not any(name is not None for name in class_names):
                continue
            cnn_track_history.record(
                track_id=r,
                class_names=class_names,
                confidences=confidences,
            )
