"""44_sys_event_diff.py — SYSTEM E: Event-Driven Differential Reader.
Do almost nothing until the frame diff (mean |gray[::8,::8]-ref|, int16,
noise floor nf from steady frames 1-9) says something changed.
QUIET (<3nf): emit previous occ+shift, near-free. LOCAL (<10nf): re-read
only cells whose window moved; escalate if >25% changed or count insane.
GLOBAL (>=10nf): per-frame T, track_shift health gate, full read + REBASE
of the reference; unhealthy -> invalid, guarded relock after 3 strikes
(System A). Reference rebases ONLY on healthy escalation or a healthy
quiet refresh every 20 frames — never while invalid. Integer contract.
"""

import json, os, time  # noqa: E401
import cv2
import numpy as np

from crux import (FRAME_PATH, acquire_board, pick_threshold, lock_lattice,
                  track_shift, read_cells, _corr_shift)
from stress import stress_sequence, score_run, print_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output", "44_sys_event_diff")
DBG_FRAMES = (0, 45, 65, 85, 103)
RATIO_MIN, RELOCK_AFTER, REFRESH_EVERY = 1.05, 3, 20

def mapped_occ(cells, counts0):
    """Occupancy remapped to ORIGINAL per-strip counts: constant length."""
    per = {}
    for si, ri, x, y, fr in cells:
        per.setdefault(si, []).append(1 if fr > 0.12 else 0)
    return tuple(b for si, c in enumerate(counts0)
                 for b in (per.get(si, []) + [0] * c)[:c])

def try_relock(gray, lat0, counts0):
    """Guarded full relock (System A): 9 strips, rows within 15%."""
    nl = lock_lattice(gray)
    if nl is None or len(nl["strips"]) != len(lat0["strips"]):
        return None
    if abs(sum(len(s["rows"]) for s in nl["strips"])
           - sum(counts0)) > 0.15 * sum(counts0):
        return None
    odx, _ = _corr_shift(lat0["ref_colproj"], nl["ref_colproj"], 60)
    ody, _ = _corr_shift(lat0["ref_rowproj"], nl["ref_rowproj"], 60)
    return nl, -odx, -ody

def draw_debug(frame, cells, valid, regime, path):
    img = frame.copy()
    for si, ri, x, y, fr in cells:
        if fr >= 0:
            cv2.circle(img, (int(x), int(y)), 3,
                       (0, 200, 0) if fr > 0.12 else (0, 0, 255), -1)
    cv2.putText(img, f"{'VALID' if valid else 'INVALID'} {regime}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255) if valid else (0, 0, 255), 2)
    cv2.imwrite(path, img)

def report(log, base_occ):
    """Print per-event accuracy/validity + per-regime timing; return both."""
    bvec, table, regs = np.array(base_occ, int), {}, {}
    for e in log:
        row = table.setdefault(e["event"], {"n_valid": 0, "n_invalid": 0,
                                            "acc_valid": [], "acc_all": []})
        acc = float(np.mean(np.array(e["occ"], int) == bvec))
        row["acc_all"].append(acc)
        row["n_valid" if e["valid"] else "n_invalid"] += 1
        e["valid"] and row["acc_valid"].append(acc)
        r = regs.setdefault(e["regime"], {"n": 0, "ms": []})
        r["n"], r["ms"] = r["n"] + 1, r["ms"] + [e["ms"]]
    print(f"\n  {'event':<12} {'valid':>5} {'inval':>5} "
          f"{'acc_valid%':>10} {'acc_all%':>9}")
    for ev, row in table.items():
        av = row.pop("acc_valid")
        av = round(float(np.mean(av)) * 100, 1) if av else None
        aa = round(float(np.mean(row.pop("acc_all"))) * 100, 1)
        row["occ_acc_valid_pct"], row["occ_acc_all_pct"] = av, aa
        print(f"  {ev:<12} {row['n_valid']:>5} {row['n_invalid']:>5} "
              f"{str(av if av is not None else '--'):>10} {aa:>9}")
    for r, d in regs.items():
        d["mean_ms"] = round(float(np.mean(d.pop("ms"))), 3)
        print(f"  regime {r:<7} n={d['n']:>3}  {d['mean_ms']}ms/frame")
    return table, regs

def main():
    os.makedirs(OUT, exist_ok=True)
    board, _ = acquire_board(cv2.imread(FRAME_PATH))
    log, cal, rebases, relocks = [], [], [], []
    nf, relock_attempts, escalations, invalid_run, last_rebase = None, 0, 0, 0, 0
    bdx = bdy = rdx = rdy = 0     # base offset (relock) + tracked shift

    def rebase(g, i):             # ONLY called on healthy frames
        nonlocal ref_gray, ref_small, last_rebase
        ref_gray, ref_small = g, g[::8, ::8].astype(np.int16)
        last_rebase = i; rebases.append(i)  # noqa: E702

    def full_read(gray):          # adapt T, track, gate, read (CAL+GLOBAL)
        nonlocal T, rdx, rdy, cell_list, occ
        T = pick_threshold(gray)
        dx, dyy, rx, ry = track_shift(cur, gray, T=T)
        if rx > RATIO_MIN and ry > RATIO_MIN:
            rdx, rdy = dx, dyy
            cell_list = read_cells(gray, cur, rdx, rdy, T=T)
            occ = mapped_occ(cell_list, counts0)
            return True
        return False

    for i, (fr, truth) in enumerate(stress_sequence(board)):
        t0 = time.perf_counter()
        if i == 0:                          # lock on the FIRST stress frame
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            lat0 = cur = lock_lattice(gray)
            assert lat0 is not None and len(lat0["strips"]) == 9
            counts0 = [len(s["rows"]) for s in lat0["strips"]]
            T, hk = lat0["T"], max(int(cur["dy"] * 0.6) | 1, 7) // 2
            cell_list = read_cells(gray, cur)
            occ = base_occ = mapped_occ(cell_list, counts0)
            base_n = sum(base_occ)
            ref_gray, ref_small = gray, gray[::8, ::8].astype(np.int16)
            regime, valid = "CAL", True
        else:
            small = cv2.cvtColor(np.ascontiguousarray(fr[::8, ::8]),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            score = float(np.mean(np.abs(small - ref_small)))
            gray = None
            if i < 10:                      # noise-floor calibration (steady)
                regime, cal = "CAL", cal + [score]
                if i == 9:
                    nf = max(float(np.mean(cal)), 0.2)
                valid = full_read(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            elif score < 3 * nf:            # (a) QUIET — dominant regime
                regime, valid = "QUIET", True
                if i - last_rebase >= REFRESH_EVERY:    # rolling refresh
                    rebase(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), i)
            else:
                do_global = score >= 10 * nf
                if not do_global:           # (b) LOCAL — re-read movers only
                    regime = "LOCAL"
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    new_list, changed, readable = list(cell_list), 0, 0
                    for idx, (si, ri, x, y, frc) in enumerate(cell_list):
                        if frc < 0:  # window off-image: skip
                            continue
                        readable += 1
                        w = gray[y - hk:y + hk + 1, x - hk:x + hk + 1]
                        r = ref_gray[y - hk:y + hk + 1, x - hk:x + hk + 1]
                        if float(np.mean(cv2.absdiff(w, r))) > 3 * nf:
                            changed += 1
                            new_list[idx] = (si, ri, x, y, float(np.mean(w < T)))
                    o2 = mapped_occ(new_list, counts0)
                    if (changed > 0.25 * max(readable, 1)
                            or not 0.7 * base_n <= sum(o2) <= 1.3 * base_n):
                        do_global, escalations = True, escalations + 1
                    else:                   # commit only the accepted update
                        cell_list, occ, valid = new_list, o2, True
                if do_global:               # (c) GLOBAL event — escalate
                    regime = "GLOBAL"
                    if gray is None:
                        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    valid = full_read(gray)
                    if valid:
                        rebase(gray, i)     # REBASE only when healthy
                    elif invalid_run + 1 >= RELOCK_AFTER:
                        relock_attempts += 1
                        got = try_relock(gray, lat0, counts0)
                        if got is not None:
                            cur, bdx, bdy = got
                            rdx = rdy = 0
                            nl2 = read_cells(gray, cur, 0, 0, T=T)
                            o2 = mapped_occ(nl2, counts0)
                            if 0.7 * base_n <= sum(o2) <= 1.3 * base_n:
                                cell_list, occ, valid = nl2, o2, True
                                relocks.append(i)
                                rebase(gray, i)
        invalid_run = 0 if valid else invalid_run + 1
        ms = (time.perf_counter() - t0) * 1000.0
        log.append({"i": i, "event": truth["event"], "valid": valid,
                    "est_dx": int(bdx + rdx), "est_dy": int(bdy + rdy),
                    "occ": occ, "ms": ms, "truth": truth, "regime": regime})
        if i in DBG_FRAMES:
            draw_debug(fr, cell_list, valid, regime,
                       os.path.join(OUT, f"dbg_{i:03d}.jpg"))

    sc = score_run(log, base_occ)
    print_score("SYSTEM E — event-driven differential reader", sc)
    table, regs = report(log, base_occ)
    qfps = round(1000.0 / max(regs.get("QUIET", {}).get("mean_ms", 1e9), 1e-6), 1)
    print(f"  steady-state (QUIET-only) fps: {qfps}   nf={round(nf, 3)}"
          f"   local->global escalations: {escalations}"
          f"\n  rebases at: {rebases}\n  relock attempts: {relock_attempts},"
          f" accepted at: {relocks}")
    metrics = dict(sc)
    metrics.update(per_event=table, regimes=regs, quiet_fps=qfps,
                   noise_floor=round(nf, 4), rebases_at=rebases,
                   escalations=escalations, relock_attempts=relock_attempts,
                   relocks_accepted_at=relocks, n_cells=len(base_occ),
                   baseline_occupied=base_n)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved: {OUT}/metrics.json + dbg overlays {DBG_FRAMES}")

if __name__ == "__main__":
    main()
