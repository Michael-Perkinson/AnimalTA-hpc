from AnimalTA.E_Post_tracking import Coos_loader_saver as CoosLS


def interpolate_coordinates(Coos, columns=None):
    """Fill gaps bracketed by known coordinates.

    Leading and trailing gaps are deliberately left untouched.  This matters
    for variable-target tracking, where those gaps can represent time before
    an animal entered or after it left the video.  The same rule is used by
    the correction window and by the all-video operation.

    Returns the number of missing coordinate rows that were filled.
    """
    if columns is None:
        columns = range(len(Coos))

    changed = 0
    for col in columns:
        valid = [row for row in range(len(Coos[col])) if Coos[col][row][0] != -1000]
        if len(valid) < 2:
            continue

        for first, last in zip(valid, valid[1:]):
            if last - first <= 1:
                continue

            # Fill only rows strictly between two known points.  Boundary
            # absence is never converted into presence implicitly.
            for raw in range(first + 1, last):
                ratio = (raw - first) / (last - first)
                Coos[col][raw][0] = Coos[col][first][0] + (
                    (Coos[col][last][0] - Coos[col][first][0]) * ratio
                )
                Coos[col][raw][1] = Coos[col][first][1] + (
                    (Coos[col][last][1] - Coos[col][first][1]) * ratio
                )
                changed += 1

    return changed


def interpolate_all(Vid):
    Coos, who_is_here = CoosLS.load_coos(Vid)
    changed = interpolate_coordinates(Coos)

    # Save only when there is an actual correction.  In particular, an
    # all-video request must not rewrite a video containing only boundary
    # absence.
    if changed:
        CoosLS.save(Vid, Coos)

    return changed
