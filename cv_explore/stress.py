"""stress.py — synthetic stress-test video + scoring for systems 40-44.

Takes the still board and generates a 120-frame sequence with KNOWN
perturbations (= ground truth): sensor noise, slow drift, a hard
table jerk, rotation, a brightness shift, and an occlusion event.
Every system runs the same sequence; score_run() grades the log.
Seeded RNG → fully reproducible.
"""

import cv2
import numpy as np

SEED = 42


def _warp(board, dx, dy, ang, bright, rng):
    """Apply rigid motion + brightness + sensor noise. bg = median."""
    h, w = board.shape[:2]
    bg = int(np.median(board))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(board, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(bg,) * 3)
    out = out.astype(np.int16) + bright
    out += rng.normal(0, 2, out.shape).astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def stress_sequence(board, n=120):
    """Yield (frame_bgr, truth) per frame. truth has the applied
    dx, dy, angle, bright, occlusion rect and event label."""
    rng = np.random.default_rng(SEED)
    h, w = board.shape[:2]
    dx = dy = 0
    ang = 0.0
    bright = 0
    for i in range(n):
        event = "steady"
        occl = None
        if 20 <= i < 40:                      # slow drift
            event = "drift"
            if i % 5 == 0:
                dx += int(rng.integers(-2, 3))
                dy += int(rng.integers(-2, 3))
        elif i == 40:                         # hard table jerk
            event = "jerk"
            dx += 17
            dy -= 12
        elif 41 <= i < 60:
            event = "post_jerk"
        elif i == 60:                         # rotation jerk
            event = "rotate"
            ang = 1.2
        elif 61 <= i < 80:
            event = "post_rotate"
        elif i == 80:                         # lighting change
            event = "bright"
            bright = 30
        elif 81 <= i < 100:
            event = "post_bright"
        elif 100 <= i < 108:                  # hand over the board
            event = "occlusion"
            occl = (int(w * 0.25), 0, int(w * 0.40), h)
        elif 108 <= i:
            event = "recovered"

        frame = _warp(board, dx, dy, ang, bright, rng)
        if occl is not None:
            x, y, ww, hh = occl
            frame[y:y + hh, x:x + ww] = 120
        yield frame, {"i": i, "event": event, "dx": dx, "dy": dy,
                      "ang": ang, "bright": bright, "occl": occl}


def score_run(log, baseline_occ):
    """Grade a system's per-frame log. Each entry must have:
    i, event, valid (bool), est_dx, est_dy, occ (tuple), ms (float).
    baseline_occ = occupancy read on frame 0 (ground truth — the
    plugs never change during the sequence)."""
    n = len(log)
    valid = [e for e in log if e["valid"]]
    uptime = len(valid) / max(n, 1) * 100

    # Shift error only where system claimed valid & no rotation
    errs = [abs(e["est_dx"] - e["truth"]["dx"])
            + abs(e["est_dy"] - e["truth"]["dy"])
            for e in valid if e["truth"]["ang"] == 0
            and e["truth"]["occl"] is None]
    shift_mae = float(np.mean(errs)) if errs else -1

    # Occupancy accuracy on claimed-valid, non-occluded frames
    accs = []
    for e in valid:
        if e["truth"]["occl"] is not None or e["occ"] is None:
            continue
        a = np.array(e["occ"], dtype=int)
        b = np.array(baseline_occ, dtype=int)
        if len(a) == len(b):
            accs.append(float(np.mean(a == b)))
    occ_acc = float(np.mean(accs)) * 100 if accs else 0

    # False-valid during occlusion (should be flagged or masked)
    occl_frames = [e for e in log if e["truth"]["occl"] is not None]
    occl_flagged = sum(1 for e in occl_frames if not e["valid"])
    occl_flag_pct = occl_flagged / max(len(occl_frames), 1) * 100

    # Recovery: frames from each event until valid AND occ matches
    recov = {}
    for ev, start in (("jerk", 40), ("rotate", 60),
                      ("bright", 80), ("occl_end", 108)):
        r = -1
        for e in log:
            if e["i"] < start:
                continue
            ok = e["valid"] and e["occ"] is not None
            if ok:
                a = np.array(e["occ"], dtype=int)
                b = np.array(baseline_occ, dtype=int)
                ok = len(a) == len(b) and np.mean(a == b) >= 0.9
            if ok:
                r = e["i"] - start
                break
        recov[ev] = r

    ms = float(np.mean([e["ms"] for e in log]))
    return {"uptime_pct": round(uptime, 1),
            "shift_mae_px": round(shift_mae, 2),
            "occ_acc_pct": round(occ_acc, 1),
            "occl_flagged_pct": round(occl_flag_pct, 1),
            "recovery_frames": recov,
            "ms_per_frame": round(ms, 2),
            "fps": round(1000.0 / max(ms, 1e-6), 1)}


def print_score(name, sc):
    print(f"\n  {'='*58}")
    print(f"  {name}")
    print(f"  {'-'*58}")
    print(f"  uptime {sc['uptime_pct']}%  occ_acc {sc['occ_acc_pct']}%"
          f"  shift_mae {sc['shift_mae_px']}px")
    print(f"  occlusion flagged {sc['occl_flagged_pct']}%"
          f"  {sc['ms_per_frame']}ms/frame ({sc['fps']} fps)")
    print(f"  recovery: {sc['recovery_frames']}")
