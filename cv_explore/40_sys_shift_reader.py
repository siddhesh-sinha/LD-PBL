"""40_sys_shift_reader.py — SYSTEM A: Shift-Tracked Lattice Reader.

Per frame: adapt threshold -> integer shift track vs locked reference
-> health gates (correlation peak ratios > 1.05, occupied count within
[0.7, 1.3]x baseline) -> read occupancy at the shifted lattice.
Re-lock: after 3 consecutive invalid frames, attempt a full
lock_lattice on the current frame; accept only a sane lattice
(9 strips, total rows within 15% of original). On accept, the new
lattice becomes the reference and occupancy is remapped to the
ORIGINAL per-strip row counts so the emitted vector never changes
length. Integer-pixel contract throughout.
"""

import json
import os
import time

import cv2
import numpy as np

from crux import (FRAME_PATH, acquire_board, pick_threshold, lock_lattice,
                  track_shift, read_cells, _corr_shift)
from stress import stress_sequence, score_run, print_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output", "40_sys_shift_reader")
DBG_FRAMES = (0, 45, 65, 85, 103)
RATIO_MIN = 1.05          # track_shift peak ratio: confident vs not
COUNT_BAND = (0.7, 1.3)   # sane occupied-count band vs baseline
RELOCK_AFTER = 3          # consecutive invalid frames before relock


def mapped_occ(cells, counts0):
    """Binary occupancy remapped to the ORIGINAL per-strip row counts
    (pad with 0 / truncate) so vector length is constant for life."""
    per = {}
    for si, ri, x, y, fr in cells:
        per.setdefault(si, []).append(1 if fr > 0.12 else 0)
    out = []
    for si, c in enumerate(counts0):
        bits = per.get(si, [])
        out.extend((bits + [0] * c)[:c])
    return tuple(out)


def try_relock(gray, lat0, counts0):
    """Full lock on the current frame; accept only if structure matches
    the original (9 strips, total rows within 15%). Returns
    (new_lat, base_dx, base_dy) with the integer offset of the new
    reference vs the frame-0 reference, or None on rejection."""
    nl = lock_lattice(gray)
    if nl is None or len(nl["strips"]) != len(lat0["strips"]):
        return None
    total = sum(len(s["rows"]) for s in nl["strips"])
    if abs(total - sum(counts0)) > 0.15 * sum(counts0):
        return None
    odx, _ = _corr_shift(lat0["ref_colproj"], nl["ref_colproj"], 60)
    ody, _ = _corr_shift(lat0["ref_rowproj"], nl["ref_rowproj"], 60)
    return nl, -odx, -ody


def draw_debug(frame, cells, valid, path):
    img = frame.copy()
    for si, ri, x, y, fr in cells:
        if fr < 0:
            continue
        col = (0, 200, 0) if fr > 0.12 else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 3, col, -1)
    cv2.putText(img, "VALID" if valid else "INVALID", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255) if valid else (0, 0, 255), 2)
    cv2.imwrite(path, img)


def main():
    os.makedirs(OUT, exist_ok=True)
    board, _ = acquire_board(cv2.imread(FRAME_PATH))

    log, relocks = [], []
    relock_attempts = 0
    lat0 = cur = None
    counts0, base_occ, base_n = None, None, 0
    bdx = bdy = 0            # offset of current reference vs frame 0
    invalid_run = 0

    for i, (fr, truth) in enumerate(stress_sequence(board)):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i == 0:           # lock on the FIRST stress frame
            lat0 = cur = lock_lattice(gray)
            assert lat0 is not None and len(lat0["strips"]) == 9
            counts0 = [len(s["rows"]) for s in lat0["strips"]]
            base_occ = mapped_occ(read_cells(gray, lat0), counts0)
            base_n = sum(base_occ)

        T = pick_threshold(gray)
        dx, dy, rx, ry = track_shift(cur, gray, T=T)
        cells = read_cells(gray, cur, dx, dy, T=T)
        occ = mapped_occ(cells, counts0)
        est_dx, est_dy = bdx + dx, bdy + dy
        health = rx > RATIO_MIN and ry > RATIO_MIN
        count_ok = COUNT_BAND[0] * base_n <= sum(occ) <= COUNT_BAND[1] * base_n
        valid = health and count_ok

        if valid:
            invalid_run = 0
        else:
            invalid_run += 1
            if invalid_run >= RELOCK_AFTER:
                relock_attempts += 1
                got = try_relock(gray, lat0, counts0)
                if got is not None:
                    cur, bdx, bdy = got
                    relocks.append(i)
                    cells = read_cells(gray, cur, 0, 0, T=T)
                    occ = mapped_occ(cells, counts0)
                    est_dx, est_dy = bdx, bdy
                    valid = (COUNT_BAND[0] * base_n <= sum(occ)
                             <= COUNT_BAND[1] * base_n)
                    if valid:
                        invalid_run = 0

        ms = (time.perf_counter() - t0) * 1000.0
        log.append({"i": i, "event": truth["event"], "valid": valid,
                    "est_dx": int(est_dx), "est_dy": int(est_dy),
                    "occ": occ, "ms": ms, "truth": truth})
        if i in DBG_FRAMES:
            draw_debug(fr, cells, valid,
                       os.path.join(OUT, f"dbg_{i:03d}.jpg"))

    sc = score_run(log, base_occ)
    print_score("SYSTEM A — shift-tracked lattice reader", sc)

    bvec = np.array(base_occ, int)
    table = {}
    for e in log:
        row = table.setdefault(e["event"],
                               {"n_valid": 0, "n_invalid": 0,
                                "acc_valid": [], "acc_all": []})
        acc = float(np.mean(np.array(e["occ"], int) == bvec))
        row["acc_all"].append(acc)
        if e["valid"]:
            row["n_valid"] += 1
            row["acc_valid"].append(acc)
        else:
            row["n_invalid"] += 1
    print(f"\n  {'event':<12} {'valid':>5} {'inval':>5} "
          f"{'acc_valid%':>10} {'acc_all%':>9}")
    for ev, row in table.items():
        accs_v = row.pop("acc_valid")
        av = round(float(np.mean(accs_v)) * 100, 1) if accs_v else None
        aa = round(float(np.mean(row.pop("acc_all"))) * 100, 1)
        row["occ_acc_valid_pct"], row["occ_acc_all_pct"] = av, aa
        print(f"  {ev:<12} {row['n_valid']:>5} {row['n_invalid']:>5} "
              f"{str(av if av is not None else '--'):>10} {aa:>9}")
    print(f"\n  relock attempts: {relock_attempts}, "
          f"accepted at frames: {relocks}")

    metrics = dict(sc)
    metrics["per_event"] = table
    metrics["relock_attempts"] = relock_attempts
    metrics["relocks_accepted_at"] = relocks
    metrics["n_cells"] = len(base_occ)
    metrics["baseline_occupied"] = base_n
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved: {OUT}/metrics.json + dbg overlays {DBG_FRAMES}")


if __name__ == "__main__":
    main()
