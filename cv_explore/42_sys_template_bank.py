"""42_sys_template_bank.py — SYSTEM C: Template Bank Verifier.

Lock once: crux lattice geometry + k x k plug template (k=0.55*dy|1)
median-stacked from occupied cells. Per frame: cv2.matchTemplate
TM_CCOEFF_NORMED (illumination-invariant, no gray thresholds), dilate
-NMS peaks >= 0.35 sep 0.7*dy. Measured spec deviation: the raw 0.35
set is ~5x plugs and its median vote FAILS the jerk; plugs score
>= ~0.70 vs spurious p99 0.77, so vote/verify/occ use peaks with
NCC >= 0.70 (a match-quality floor). Strong detections vote integer
shift (median delta to nearest occupied cell, |d| <= 40); an internal
float angle fitted from pair residuals (emitted positions stay ints)
derotates the lattice, shift re-voted. valid = >=55% occupied cells
verified within 6px AND no strip < 30% (occlusion detector). occ =
strong detection within 0.6*dy of the transformed cell. 5 straight
invalid frames -> guarded template-only rebuild via lock_lattice
bootstrap (geometry stays frame-0). Full+half-res variants both run;
primary = whichever passes the steady >97% occupancy floor.
"""
import json, os, time
import cv2
import numpy as np
from crux import (FRAME_PATH, acquire_board, lock_lattice, read_cells,
                  occupancy_vector)
from stress import stress_sequence, score_run, print_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output", "42_sys_template_bank")
DBG_FRAMES = (0, 45, 65, 85, 103)
# NCC floors, verify radius/frac, min per-strip verified frac,
# angle-fit pair radius, invalid frames before rebuild, angle clamp
NCC_MIN, NCC_STRONG, VERIFY_PX, VERIFY_FRAC = 0.35, 0.70, 6, 0.55
STRIP_MIN, PAIR_R, RELOCK_AFTER, MAX_ANG = 0.30, 14, 5, 0.06

def build_template(gray, occ_pts, k):
    hk, (h, w) = k // 2, gray.shape
    ps = [gray[y - hk:y + hk + 1, x - hk:x + hk + 1] for x, y in occ_pts
          if hk <= x < w - hk and hk <= y < h - hk]
    return np.median(np.stack(ps), 0).astype(np.uint8)

def detect(gray, tmpl, sep, scale):
    # NCC map -> dilate-NMS peaks: centers (Nx2 int, full-res) + scores
    g, t = gray, tmpl
    if scale > 1:
        g = cv2.resize(g, (g.shape[1] // scale, g.shape[0] // scale), interpolation=cv2.INTER_AREA)
        t = cv2.resize(t, ((t.shape[1] // scale) | 1,) * 2, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
    dil = cv2.dilate(res, np.ones((max(3, sep // scale) | 1,) * 2, np.uint8))
    ys, xs = np.nonzero((res >= NCC_MIN) & (res >= dil))
    sc, hk = res[ys, xs], t.shape[0] // 2
    return (np.stack([xs + hk, ys + hk], 1) * scale).astype(int), sc

def vote_shift(det, ref):
    # componentwise median of (detection - nearest ref point), |d| <= 40
    d = det[:, None, :] - ref[None, :, :]
    dl = d[np.arange(len(det)), (d ** 2).sum(2).argmin(1)]
    dl = dl[(np.abs(dl) <= 40).all(1)]
    if len(dl) == 0: return 0, 0
    return int(round(np.median(dl[:, 0]))), int(round(np.median(dl[:, 1])))

def fit_angle(det, occ_pts, c, dx, dy):
    """Internal float angle from matched cell -> detection residuals."""
    if len(det) < 25: return 0.0
    sh = occ_pts + (dx, dy)
    d2 = ((sh[:, None, :] - det[None, :, :]) ** 2).sum(2)
    j = d2.argmin(1)
    ok = d2[np.arange(len(sh)), j] <= PAIR_R ** 2
    p, r = (occ_pts - c)[ok], (det[j] - sh)[ok]
    far = (p ** 2).sum(1) >= 80 ** 2
    if far.sum() < 25: return 0.0
    th = np.median((p[far, 0] * r[far, 1] - p[far, 1] * r[far, 0]) / (p[far] ** 2).sum(1))
    return float(np.clip(th, -MAX_ANG, MAX_ANG))

def warp_pts(pts, c, th, dx, dy):
    """Rotate about c by internal float th, integer-shift, emit ints."""
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.rint((pts - c) @ R.T + c + (dx, dy)).astype(int)

def ndist(ref, det):
    if len(det) == 0: return np.full(len(ref), 1e9)
    return np.sqrt(((ref[:, None, :].astype(float) - det[None, :, :]) ** 2).sum(2)).min(1)

def process(gray, S):
    det, sc = detect(gray, S["tmpl"], S["sep"], S["scale"])
    du = det[sc >= NCC_STRONG]
    med_all = float(np.median(sc)) if len(sc) else 0.0
    med_s = float(np.median(sc[sc >= NCC_STRONG])) if len(du) else 0.0
    dx, dy = vote_shift(du, S["occ_pts"])
    th = fit_angle(du, S["occ_pts"], S["c"], dx, dy)
    dx, dy = vote_shift(du, warp_pts(S["occ_pts"], S["c"], th, 0, 0))
    ver = ndist(warp_pts(S["occ_pts"], S["c"], th, dx, dy), du) <= VERIFY_PX
    strip_min = min(float(ver[m].mean()) for m in S["strip_masks"])
    valid = float(ver.mean()) >= VERIFY_FRAC and strip_min >= STRIP_MIN
    pos = warp_pts(S["cell_pts"], S["c"], th, dx, dy)
    return dx, dy, valid, tuple((ndist(pos, du) <= S["r_occ"]).astype(int)), pos, med_all, med_s

def rebuild(gray, S):
    # guarded template-only rebuild via lock_lattice bootstrap
    nl = lock_lattice(gray)
    if nl is None or len(nl["strips"]) != len(S["strip_masks"]) or abs(
            sum(len(s["rows"]) for s in nl["strips"]) - S["n_cells"]) > 0.15 * S["n_cells"]:
        return False
    op = np.array([(c[2], c[3]) for c in read_cells(gray, nl) if c[4] > 0.12])
    if len(op) < 0.5 * len(S["occ_pts"]):
        return False
    S["tmpl"] = build_template(gray, op, S["k"])
    return True

def run(board, scale):
    log, S, dbg = [], {"scale": scale}, {}
    base_occ, inval_run, reb_attempts, reb_at = None, 0, 0, []
    for i, (fr, truth) in enumerate(stress_sequence(board)):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i == 0:                      # lock on the FIRST stress frame
            lat = lock_lattice(gray)
            assert lat is not None and len(lat["strips"]) == 9
            cells = read_cells(gray, lat)
            base_occ = occupancy_vector(cells)
            cp, ov = np.array([(c[2], c[3]) for c in cells]), np.array(base_occ) == 1
            si = np.array([c[0] for c in cells])[ov]
            S.update(cell_pts=cp, occ_pts=cp[ov], n_cells=len(cells),
                     k=int(lat["dy"] * 0.55) | 1, sep=int(0.7 * lat["dy"]),
                     r_occ=int(0.6 * lat["dy"]),
                     c=np.array([gray.shape[1] / 2, gray.shape[0] / 2]),
                     strip_masks=[np.nonzero(si == s)[0]
                                  for s in range(9) if (si == s).any()])
            S["tmpl"] = build_template(gray, S["occ_pts"], S["k"])
        out = process(gray, S)
        if not out[2]:
            inval_run += 1
            if inval_run >= RELOCK_AFTER:
                reb_attempts += 1
                if rebuild(gray, S):
                    reb_at.append(i); out = process(gray, S)
        dx, dy, valid, occ, pos, med_all, med_s = out
        inval_run = 0 if valid else inval_run
        ms = (time.perf_counter() - t0) * 1000.0
        log.append({"i": i, "event": truth["event"], "valid": valid,
                    "est_dx": int(dx), "est_dy": int(dy), "occ": occ, "ms": ms,
                    "truth": truth, "med_all": med_all, "med_strong": med_s})
        if i in DBG_FRAMES: dbg[i] = (fr.copy(), pos, occ, valid)
    return log, base_occ, dbg, reb_attempts, reb_at

def event_table(log, bvec):
    tb = {}
    for e in log:
        r = tb.setdefault(e["event"], dict(n_valid=0, n_invalid=0, av=[], aa=[], s=[], t=[]))
        acc = float(np.mean(np.array(e["occ"], int) == bvec))
        r["aa"].append(acc); r["s"].append(e["med_all"]); r["t"].append(e["med_strong"])
        r["n_valid" if e["valid"] else "n_invalid"] += 1
        if e["valid"]:
            r["av"].append(acc)
    print(f"\n  {'event':<12} {'valid':>5} {'inval':>5} {'accV%':>6} "
          f"{'accAll%':>7} {'nccAll':>6} {'nccStrong':>9}")
    for ev, r in tb.items():
        av, aa = r.pop("av"), r.pop("aa")
        r["occ_acc_valid_pct"] = round(np.mean(av) * 100, 1) if av else None
        r["occ_acc_all_pct"] = round(float(np.mean(aa)) * 100, 1)
        r["med_ncc_all_peaks"] = round(float(np.median(r.pop("s"))), 3)
        r["med_ncc_strong"] = round(float(np.median(r.pop("t"))), 3)
        print(f"  {ev:<12} {r['n_valid']:>5} {r['n_invalid']:>5} "
              f"{str(r['occ_acc_valid_pct'] or '--'):>6} "
              f"{r['occ_acc_all_pct']:>7} {r['med_ncc_all_peaks']:>6} "
              f"{r['med_ncc_strong']:>9}")
    return tb

def main():
    os.makedirs(OUT, exist_ok=True); board, gi = acquire_board(cv2.imread(FRAME_PATH))
    cv2.matchTemplate(gi, gi[:15, :15].copy(), cv2.TM_CCOEFF_NORMED)  # warm
    res = {s: run(board, s) for s in (1, 2)}
    steady = {s: float(np.mean([np.mean(np.array(e["occ"], int) == np.array(r[1], int))
                                for e in r[0] if e["event"] == "steady"]))
              for s, r in res.items()}
    prim = 1 if steady[1] > 0.97 else 2   # full-res primary if it passes
    sec = 3 - prim
    log, base_occ, dbg, reb_attempts, reb_at = res[prim]
    sc = score_run(log, base_occ)
    print_score(f"SYSTEM C — template bank verifier (scale 1/{prim})", sc)
    table = event_table(log, np.array(base_occ, int))
    for i, (fr, pos, occ, valid) in dbg.items():
        for (x, y), o in zip(pos, occ):
            cv2.circle(fr, (int(x), int(y)), 3, (0, 200, 0) if o else (0, 0, 255), -1)
        cv2.putText(fr, "VALID" if valid else "INVALID", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255) if valid else (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUT, f"dbg_{i:03d}.jpg"), fr)
    sc2 = score_run(res[sec][0], res[sec][1])
    metrics = dict(sc, per_event=table, primary_scale=prim, secondary_scale=sec,
                   steady_acc_by_scale={s: round(v * 100, 2) for s, v in steady.items()},
                   secondary_score=sc2, rebuild_attempts=reb_attempts,
                   rebuilds_accepted_at=reb_at, n_cells=len(base_occ),
                   baseline_occupied=int(sum(base_occ)))
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"\n  steady acc by scale: {metrics['steady_acc_by_scale']}  "
          f"primary=1/{prim}  secondary 1/{sec}: {sc2['occ_acc_pct']}% occ, "
          f"{sc2['ms_per_frame']}ms/f\n  rebuild attempts {reb_attempts}, "
          f"accepted at {reb_at}\n  saved: {OUT}/metrics.json + dbg "
          f"overlays {DBG_FRAMES}")

if __name__ == "__main__":
    main()
