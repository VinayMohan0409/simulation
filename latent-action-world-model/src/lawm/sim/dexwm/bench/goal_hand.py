"""The goal hand from the goal frame alone (no target joint angles: DexWM's planner never sees them; its keypoint head predicts the goal hand from the goal image).
The hand pixels of the goal RGB-D frame (SAM 3, "robot hand") are back-projected and the hand model (HandKinematics: the same mesh points as the FK hand rows) is fitted to them
by trimmed ICP over wrist pose and the 16 joint angles, started from the hand's measured start pose moved onto the observed points. Returns full hand rows for the usual hand goal cost."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def fit_to_observed(kin, observed, R0, p0, q0, iters=12, keep=0.8, f_scale=0.01, max_nfev=40, yaw_starts=(0.0,)):
    """(rot, pos, q, info): hand pose whose model points best cover the observed hand points (one-sided observed -> model, the worst (1 - keep) dropped each round).
    Start: R0 (optionally turned about the vertical by each yaw in yaw_starts), translated so the model centroid sits on the observed centroid; q0; the best start is kept."""
    observed = np.asarray(observed, float)
    lo, hi = kin.limits[:, 0] + 1e-6, kin.limits[:, 1] - 1e-6
    best = None
    for yaw in yaw_starts:
        R = Rotation.from_rotvec([0, 0, yaw]).as_matrix() @ R0
        q = np.clip(np.asarray(q0, float), lo, hi)
        p = np.asarray(p0, float) + observed.mean(0) - kin.points(R, p0, q).mean(0)
        for _ in range(iters):
            model = kin.points(R, p, q)
            d, idx = cKDTree(model).query(observed)
            sel = d <= np.quantile(d, keep)
            obs_sel, idx_sel = observed[sel], idx[sel]

            def residual(x, R=R, obs_sel=obs_sel, idx_sel=idx_sel):
                Rx = Rotation.from_rotvec(x[:3]).as_matrix() @ R
                return (kin.points(Rx, x[3:6], x[6:])[idx_sel] - obs_sel).ravel()
            x0 = np.r_[np.zeros(3), p, q]
            sol = least_squares(residual, x0, bounds=(np.r_[[-np.pi] * 3, [-np.inf] * 3, lo], np.r_[[np.pi] * 3, [np.inf] * 3, hi]), loss="soft_l1", f_scale=f_scale,
                                max_nfev=max_nfev)
            R, p, q = Rotation.from_rotvec(sol.x[:3]).as_matrix() @ R, sol.x[3:6], sol.x[6:]
        d = cKDTree(kin.points(R, p, q)).query(observed)[0]
        score = float(np.mean(np.sort(d)[: max(1, int(keep * len(d)))]))
        if best is None or score < best[3]["trimmed_mean_mm"] / 1000:
            best = (R, p, q, dict(trimmed_mean_mm=score * 1000, median_mm=float(np.median(d) * 1000), n_observed=len(observed), yaw_start=float(yaw)))
    return best
