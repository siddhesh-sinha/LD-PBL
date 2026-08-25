"""43_sys_fourier_grid.py — SYSTEM D: 2D Fourier Grid Sensor.

Board rotation rotates the FFT of the dark mask by the same angle. One
integer peak bin at r0=H/dy quantizes to ~2.6 deg (useless for 1.2), so
we polar-sample annuli at grid harmonics 1-4 and cross-correlate each
harmonic's zero-mean angular profile against the lock frame's — the
whole speckle fingerprint is a matched filter (~0.01 deg repeatable).
Bin space is anisotropic (pad Wp!=Hp): rotation theta appears as a
bin-angle shift of (Wp/Hp)*theta near vertical, so the measurement is
scaled by Hp/Wp. If |angle|>=0.3 deg derotate the gray (INTER_NEAREST;
float angle internal only), then track/read as usual. FFT runs on a
2x-decimated mask; correlation score = occlusion detector."""

import json
import os
import time
import cv2
import numpy as np

from crux import (FRAME_PATH, acquire_board, pick_threshold, dark_mask,
                  lock_lattice, track_shift, read_cells, occupancy_vector)
from stress import stress_sequence, score_run, print_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output", "43_sys_fourier_grid")
DBG_FRAMES = (0, 45, 65, 85, 103)
STEP, SPAN, SEARCH = 0.05, 9.0, 80  # polar deg/sample; half-span; +-4deg
SNR_MIN, RATIO_MIN = 3.0, 1.04      # h1-peak SNR gate; track-ratio gate
SCORE_MIN, DEROT_MIN = 1.5, 0.3     # fingerprint gate; min deg to derotate


class FourierGrid:
    """Angle sensor: polar-resampled harmonic annuli of |FFT(dark mask)|,
    correlated in angle against the lock-frame reference."""

    def __init__(self, shape, dy, down=2):
        H, W = shape
        self.down = down
        h, w = (H + down - 1) // down, (W + down - 1) // down
        self.hp, self.wp = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
        self.aspect = self.hp / self.wp        # bin-angle -> physical angle
        self.hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        self.r0 = self.hp / (dy / down)        # fundamental row freq (bins)
        cy, cx = self.hp // 2, self.wp // 2
        ths = np.deg2rad(np.arange(-SPAN, SPAN + 1e-9, STEP))
        self.maps = []
        for n in (1, 2, 3, 4):                 # harmonic annuli
            band = max(1.5, 0.06 * n * self.r0)
            rr = np.arange(n * self.r0 - band, n * self.r0 + band, 0.5)
            self.maps.append((
                (cx + rr[:, None] * np.sin(ths)).astype(np.float32),
                (cy + rr[:, None] * np.cos(ths)).astype(np.float32)))
        fy, fx = np.meshgrid(np.arange(self.hp) - cy,
                             np.arange(self.wp) - cx, indexing="ij")
        self.rad = np.sqrt(fy * fy + fx * fx)
        self.ann = np.abs(self.rad - self.r0) < max(2, 0.1 * self.r0)
        self.wedge = self.ann & (fy > 0) & (np.abs(np.degrees(
            np.arctan2(fx, np.maximum(fy, 1)))) < 6)

    def profiles(self, gray, T):
        """-> (per-harmonic zero-mean unit profiles, spec SNR, r_peak)."""
        m = dark_mask(gray, T)[::self.down, ::self.down]
        m = m.astype(np.float32) * self.hann
        F = np.abs(np.fft.fftshift(
            np.fft.fft2(m, s=(self.hp, self.wp)))).astype(np.float32)
        out = []
        for mx, my in self.maps:
            q = cv2.remap(F, mx, my, cv2.INTER_LINEAR).sum(axis=0)
            q -= q.mean()
            out.append(q / (np.linalg.norm(q) + 1e-9))
        wv = np.where(self.wedge, F, 0)
        peak_i = np.unravel_index(int(np.argmax(wv)), F.shape)
        snr = float(F[peak_i] / np.median(F[self.ann]))
        return out, snr, float(self.rad[peak_i])

    def measure(self, ps, ps0):
        """Correlate profiles vs lock reference -> (deg, score).
        Positive = stress-convention angle (cv2 CCW)."""
        n = len(ps0[0])
        total = np.zeros(2 * SEARCH + 1)
        for a, b in zip(ps0, ps):
            for s in range(-SEARCH, SEARCH + 1):
                total[s + SEARCH] += np.dot(a[max(0, s):min(n, n + s)],
                                            b[max(0, -s):min(n, n - s)])
        best = int(np.argmax(total))
        d = 0.0
        if 0 < best < 2 * SEARCH:              # parabolic sub-sample refine
            y0, y1, y2 = total[best - 1:best + 2]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        return -(best - SEARCH + d) * STEP * self.aspect, float(total[best])


def main():
    os.makedirs(OUT, exist_ok=True)
    board, _ = acquire_board(cv2.imread(FRAME_PATH))
    log, cur_ang = [], 0.0

    for i, (fr, truth) in enumerate(stress_sequence(board)):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        T = pick_threshold(gray)
        if i == 0:                             # lock on FIRST stress frame
            lat, gray0 = lock_lattice(gray), gray
            assert lat is not None and len(lat["strips"]) == 9
            fg = FourierGrid(gray.shape, lat["dy"], down=2)
            ps0, _, rpk = fg.profiles(gray, T)
            dy_refined = round(fg.hp * fg.down / max(rpk, 1e-9), 2)
            base_occ = occupancy_vector(read_cells(gray, lat, T=T))

        ps, snr, _ = fg.profiles(gray, T)
        ang, score = fg.measure(ps, ps0)
        if score >= SCORE_MIN:                 # trust angle; else hold last
            cur_ang = ang
        g_used, (h, w) = gray, gray.shape
        if abs(cur_ang) >= DEROT_MIN:          # derotate about board center
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -cur_ang, 1.0)
            g_used = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_NEAREST,
                                    borderValue=int(np.median(gray)))
        dx, dyy, rx, ry = track_shift(lat, g_used, T=T)
        cells = read_cells(g_used, lat, dx, dyy, T=T)
        occ = occupancy_vector(cells)
        # score = fingerprint match: collapses (~0.3 vs ~3) when occluded
        valid = (snr > SNR_MIN and rx > RATIO_MIN and ry > RATIO_MIN
                 and score >= SCORE_MIN)
        ms = (time.perf_counter() - t0) * 1000.0

        log.append({"i": i, "event": truth["event"], "valid": valid,
                    "est_dx": int(dx), "est_dy": int(dyy), "occ": occ,
                    "ms": ms, "truth": truth, "ang": round(cur_ang, 3),
                    "score": round(score, 2), "snr": round(snr, 1)})
        if i in DBG_FRAMES:
            img = fr.copy() if g_used is gray else cv2.warpAffine(
                fr, M, (w, h), flags=cv2.INTER_NEAREST)
            for _, _, x, y, frr in cells:
                if frr >= 0:
                    cv2.circle(img, (x, y), 3,
                               (0, 200, 0) if frr > 0.12 else (0, 0, 255), -1)
            cv2.putText(img, f"{'VALID' if valid else 'INVALID'} "
                        f"ang={cur_ang:+.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255) if valid else (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUT, f"dbg_{i:03d}.jpg"), img)

    sc = score_run(log, base_occ)
    print_score("SYSTEM D — 2D Fourier grid sensor (derotating)", sc)

    bvec = np.array(base_occ, int)
    table = {}
    for e in log:
        row = table.setdefault(e["event"], {"n_valid": 0, "n_invalid": 0,
                                            "acc": [], "snr": [], "ang": []})
        row["acc"].append(float(np.mean(np.array(e["occ"], int) == bvec)))
        row["snr"].append(e["snr"])
        row["ang"].append(e["ang"])
        row["n_valid" if e["valid"] else "n_invalid"] += 1
    print(f"\n  {'event':<12} {'valid':>5} {'inval':>5} {'occ_acc%':>8} "
          f"{'snr':>6} {'ang':>6}")
    for ev, row in table.items():
        for k, src, f in (("occ_acc_pct", "acc", 100), ("mean_snr", "snr", 1),
                          ("mean_ang", "ang", 1)):
            row[k] = round(float(np.mean(row.pop(src))) * f, 2)
        print(f"  {ev:<12} {row['n_valid']:>5} {row['n_invalid']:>5} "
              f"{row['occ_acc_pct']:>8} {row['mean_snr']:>6} "
              f"{row['mean_ang']:>6}")

    # angle accuracy vs truth (scoring only) + full-res FFT benchmark
    pre = [abs(e["ang"]) for e in log if e["truth"]["ang"] == 0]
    post = [abs(e["ang"] - 1.2) for e in log
            if e["truth"]["ang"] != 0 and e["truth"]["occl"] is None]
    w55 = [e["ang"] for e in log if 55 <= e["i"] <= 75]

    def bench(f, ref):
        t0 = time.perf_counter()
        for _ in range(5):
            f.measure(f.profiles(gray0, pick_threshold(gray0))[0], ref)
        return (time.perf_counter() - t0) * 200
    fg1 = FourierGrid(gray0.shape, lat["dy"], down=1)
    ms_full = bench(fg1, fg1.profiles(gray0, pick_threshold(gray0))[0])
    ms_down = bench(fg, ps0)
    print(f"\n  angle MAE: pre-rot {np.mean(pre):.3f} deg, "
          f"post-rot {np.mean(post):.3f} deg (truth 1.2)\n  FFT cost: "
          f"full-res {ms_full:.1f} ms, 2x-down {ms_down:.1f} ms (prod)")

    metrics = dict(sc, per_event=table, n_cells=len(base_occ),
                   baseline_occupied=int(sum(base_occ)), dy_locked=lat["dy"],
                   dy_refined_from_fft=dy_refined,
                   angle_mae_prerot_deg=round(float(np.mean(pre)), 4),
                   angle_mae_postrot_deg=round(float(np.mean(post)), 4),
                   angle_frames_55_75=[round(a, 3) for a in w55],
                   fft_ms_fullres=round(ms_full, 1),
                   fft_ms_2xdown=round(ms_down, 1),
                   angle_held_frames=[e["i"] for e in log
                                      if e["score"] < SCORE_MIN])
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved: {OUT}/metrics.json + dbg overlays {DBG_FRAMES}")


if __name__ == "__main__":
    main()
