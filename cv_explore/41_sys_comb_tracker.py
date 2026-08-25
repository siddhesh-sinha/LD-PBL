"""41_sys_comb_tracker.py — SYSTEM B: comb-model tracker with per-strip
verification. Global integer shift (track_shift), then each strip votes a
residual row offset o_s via a comb over its own row projection. Reliable
strips (decisive comb peak, |o_s|<=2, healthy dark mass) refine dy by the
median vote; unreliable strips HOLD last-known occupancy (occlusion
localization). 3 consecutive invalid frames -> guarded re-lock: a fresh
lattice is adopted only if it maps onto the old one with high agreement.
Calibration note: the spec'd best>1.3*median-over-offsets is blind here —
plugs span ~dy/2 so the whole ±4 comb sits on plug mass; the dy/2 valley
(anti-phase comb) is the honest null reference, 1.3x kept as the ratio."""
import json
import os
import time

import cv2
import numpy as np

from crux import (FRAME_PATH, acquire_board, pick_threshold, dark_mask,
                  lock_lattice, track_shift, read_cells, occupancy_vector)
from stress import stress_sequence, score_run, print_score

OUT = "output/41_sys_comb_tracker"
O = 4              # comb residual search range: o in -4..+4
MAX_O = 2          # reliable strips must sit within +-2 rows of global dy
CONTRAST = 1.3     # comb peak vs anti-phase (dy/2) valley
MASS_FLOOR = 0.30  # comb mass vs lock-time mass (occlusion kill switch)
MIN_REL = 5        # valid frame needs >= 5 reliable strips
DBG = (0, 45, 65, 85, 103)


def slot_ranges(nrows):
    out, i = [], 0
    for n in nrows:
        out.append(i)
        i += n
    return out


def comb_verify(d, lat, dx, dy):
    """Per strip: (residual offset o_s, peak/valley contrast, comb mass)."""
    h, w = d.shape
    half, res = lat["dy"] // 2, []
    for s in lat["strips"]:
        x0, x1 = max(0, s["x0"] + dx), min(w, s["x1"] + dx)
        if x1 - x0 < 5:
            res.append((0, 0.0, 0))
            continue
        proj = d[:, x0:x1].sum(axis=1).astype(np.int64)
        ys = np.array(s["rows"], dtype=int) + dy

        def comb(o):
            yy = ys + o
            return int(proj[yy[(yy >= 0) & (yy < h)]].sum())
        C = [comb(o) for o in range(-O, O + 1)]
        o_s = int(np.argmax(C)) - O
        anti = (comb(o_s - half) + comb(o_s + half)) / 2.0
        res.append((o_s, C[o_s + O] / max(anti, 1e-9), C[o_s + O]))
    return res


def estimate(gray, d, lat, base_mass, T):
    """Track + per-strip verify. Returns dx, dy_refined, rel flags, o_s."""
    dx, dy, _, _ = track_shift(lat, gray, T=T)
    vres = comb_verify(d, lat, dx, dy)
    rel = [c > CONTRAST and abs(o) <= MAX_O and m > MASS_FLOOR * bm
           for (o, c, m), bm in zip(vres, base_mass)]
    votes = [o for (o, _, _), r in zip(vres, rel) if r]
    dyr = dy + (int(round(np.median(votes))) if votes else 0)
    return dx, dyr, rel, [v[0] for v in vres]


def mapped_agree(nocc, nrows2, ref, slots0, nrows0):
    s2, m, t = slot_ranges(nrows2), 0, 0
    for a0, b0, na, nb in zip(slots0, s2, nrows0, nrows2):
        for k in range(min(na, nb)):
            m, t = m + (ref[a0 + k] == nocc[b0 + k]), t + 1
    return m / max(t, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    board, _ = acquire_board(cv2.imread(FRAME_PATH))
    log, rel_tl, relocks, flips = [], [], [], []
    lat = None
    for i, (fbgr, truth) in enumerate(stress_sequence(board)):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(fbgr, cv2.COLOR_BGR2GRAY)
        T = pick_threshold(gray)
        d = dark_mask(gray, T)
        if lat is None:                                  # first-frame lock
            lat = lock_lattice(gray)
            nrows0 = [len(s["rows"]) for s in lat["strips"]]
            slots0 = slot_ranges(nrows0)
            base_mass = [m for _, _, m in comb_verify(d, lat, 0, 0)]
            occ_state = list(occupancy_vector(read_cells(gray, lat, T=T)))
            baseline_occ = tuple(occ_state)
            cur_nrows, bdx, bdy, streak = nrows0, 0, 0, 0
        dx, dyr, rel, o_all = estimate(gray, d, lat, base_mass, T)
        valid = sum(rel) >= MIN_REL
        streak = 0 if valid else streak + 1
        if streak >= 3:                                  # guarded re-lock
            nl = lock_lattice(gray)
            if nl is not None and len(nl["strips"]) == len(nrows0) \
                    and abs(nl["dy"] - lat["dy"]) <= 3:
                nr2 = [len(s["rows"]) for s in nl["strips"]]
                nocc = occupancy_vector(read_cells(gray, nl, T=T))
                ag = mapped_agree(nocc, nr2, occ_state, slots0, nrows0)
                if ag >= 0.65:
                    bdx, bdy, lat, cur_nrows = bdx + dx, bdy + dyr, nl, nr2
                    base_mass = [m for _, _, m in comb_verify(d, lat, 0, 0)]
                    relocks.append({"i": i, "agree": round(ag, 3)})
                    dx, dyr, rel, o_all = estimate(gray, d, lat, base_mass, T)
                    valid = sum(rel) >= MIN_REL
                    streak = 0 if valid else streak
        cells = read_cells(gray, lat, dx, dyr, T=T)
        occ_new = occupancy_vector(cells)
        cur_slots = slot_ranges(cur_nrows)
        for si, (r, a0, b0) in enumerate(zip(rel, slots0, cur_slots)):
            if not r:
                continue
            n = min(nrows0[si], cur_nrows[si])
            seg = occ_new[b0:b0 + n]
            mt = float(np.mean(np.array(seg) ==
                               np.array(baseline_occ[a0:a0 + n])))
            if mt < 0.5:
                flips.append({"i": i, "strip": si, "match": round(mt, 2)})
            occ_state[a0:a0 + n] = seg
        occ = tuple(occ_state)
        ms = (time.perf_counter() - t0) * 1000
        rel_tl.append("".join("1" if r else "." for r in rel))
        log.append({"i": i, "event": truth["event"], "valid": bool(valid),
                    "est_dx": int(bdx + dx), "est_dy": int(bdy + dyr),
                    "occ": occ, "ms": ms, "truth": truth})
        if i in DBG:
            img = fbgr.copy()
            for si, ri, x, y, _ in cells:
                if ri >= nrows0[si]:
                    continue
                v = occ_state[slots0[si] + ri]
                cv2.circle(img, (x, y), 4,
                           (0, 0, 230) if v else (0, 200, 0), -1)
                if not rel[si]:
                    cv2.circle(img, (x, y), 7, (0, 165, 255), 1)
            cv2.imwrite(f"{OUT}/dbg_{i:03d}.jpg", img)

    sc = score_run(log, baseline_occ)
    print_score("SYSTEM B — comb tracker + per-strip verification", sc)
    table, order = {}, []
    for e in log:
        ev = e["event"]
        if ev not in table:
            table[ev], _ = {"n": 0, "valid": 0, "m": []}, order.append(ev)
        table[ev]["n"] += 1
        table[ev]["valid"] += e["valid"]
        table[ev]["m"].append(float(np.mean(
            np.array(e["occ"]) == np.array(baseline_occ))))
    print(f"\n  {'event':<12}{'n':>4}{'valid':>7}{'invalid':>9}"
          f"{'occ_match%':>12}")
    for ev in order:
        t = table[ev]
        t["occ_match_pct"] = round(100 * np.mean(t.pop("m")), 1)
        t["invalid"] = t["n"] - t["valid"]
        print(f"  {ev:<12}{t['n']:>4}{t['valid']:>7}{t['invalid']:>9}"
              f"{t['occ_match_pct']:>12}")
    occl_unrel = {si: sum(rel_tl[k][si] == "." for k in range(100, 108))
                  for si in range(len(nrows0))}
    print(f"\n  relocks: {relocks}\n  flip suspects: {flips}")
    print("  strip reliability f100-107 (1=reliable):")
    for k in range(100, 108):
        print(f"    f{k}: {rel_tl[k]}")
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"score": sc, "per_event": table, "relocks": relocks,
                   "flip_suspects": flips,
                   "occl_unreliable_per_strip": occl_unrel,
                   "rel_timeline_100_107": rel_tl[100:108]}, f, indent=1)
    print(f"\n  saved {OUT}/metrics.json")


if __name__ == "__main__":
    main()
