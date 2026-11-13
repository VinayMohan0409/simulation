"""Object segmentation for the production scene: SAM 3 (text prompt, in its own process: segment_server.py) gives instance masks per camera view; SAM 3 generic text prompts give candidate
masks; FK decides what is robot (the scene cloud's own robot removal): a mask whose points are mostly robot is rejected; the object is the remaining mask closest to the
FK hand points (fingers and pads included). No object names (DexWM's planner gets no language) and no simulator segmentation."""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
from pathlib import Path

import numpy as np

SAM3_PYTHON = os.environ.get("SAM3_PYTHON", "/vol/dissolve/vinay/envs/sam3/bin/python")
SAM3_REPO = os.environ.get("SAM3_REPO", "/vol/dissolve/vinay/work/third_party/sam3")
SAM3_HF_CACHE = os.environ.get("SAM3_HF_CACHE", "/vol/dissolve/vinay/work/hf_cache")  # the home quota is too small for the checkpoint
PROMPTS = ("small object",)  # generic only (DexWM gets no language); 2026-11-13 prompt test: covers the object at start 13/18, goal 18/18 ("object": 1/18, 10/18)
THRESHOLD = 0.3
MAX_ROBOT_FRAC = 0.5  # a SAM mask with more than this fraction of its 3D points removed by the FK robot removal IS the robot (rejected)


class Segmenter:
    """Handle to a SAM 3 worker process; segment(images) -> per image {prompt: dict(masks, scores, boxes)}."""

    def __init__(self, prompts=PROMPTS, threshold=THRESHOLD, log=None):
        self.prompts, self.threshold = tuple(prompts), threshold
        server = Path(__file__).with_name("segment_server.py")
        env = dict(os.environ, PYTHONPATH=SAM3_REPO, PYTHONNOUSERSITE="1", HF_HUB_CACHE=SAM3_HF_CACHE)
        self.proc = subprocess.Popen([SAM3_PYTHON, "-P", str(server)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env)
        hello = self._read()
        assert hello and hello.get("ready"), f"SAM 3 worker did not start: {hello}"
        if log:
            log(f"SAM 3 worker ready (torch {hello['torch']}), prompts {self.prompts}, threshold {threshold}")

    def _read(self):
        head = self.proc.stdout.read(8)
        if len(head) < 8:
            return None
        return pickle.loads(self.proc.stdout.read(struct.unpack("<Q", head)[0]))

    def segment(self, images, points=None):
        """Text prompts (default): per image {prompt: dict(masks, scores, boxes)}. With points (per image (coords (n, 2) x y, labels (n,)) or None): per image dict(mask, score) or None."""
        req = dict(images=[np.ascontiguousarray(i, np.uint8) for i in images], prompts=list(self.prompts), threshold=self.threshold)
        if points is not None:
            req["points"] = points
        data = pickle.dumps(req, protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self.proc.stdin.flush()
        reply = self._read()
        assert reply is not None, "SAM 3 worker died"
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(struct.pack("<Q", 0))
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()



def pixel_points(depth, K, pose, mask, near=0.1, far=3.0):
    """(v, u, world points) of the mask's pixels with valid depth (same convention as production.back_project)."""
    v, u = np.nonzero(mask & (depth > near) & (depth < far))
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return v, u, cam @ pose[:3, :3].T + pose[:3, 3]


def object_masks(segmenter, views, robot_points, hand_points, finger_mask=None):
    """Per view: (object mask (H, W) bool, info).
    1) SAM 3 candidate masks from every generic prompt; 2) back-project each; 3) FK robot removal, exactly as ProductionScene.scene_points (points within
    production.ROBOT_MARGIN of the FK robot surface prod.robot(data)); a mask with more than MAX_ROBOT_FRAC of its points removed is the robot -> rejected;
    4) of the rest, the object = the mask whose remaining points have the smallest median distance to the FK hand points. Nothing left: empty mask (info["reason"])."""
    from scipy.spatial import cKDTree

    from lawm.sim.dexwm.planspace.production import ROBOT_MARGIN
    robot, hand = cKDTree(np.asarray(robot_points, float)), cKDTree(np.asarray(hand_points, float))
    results = segmenter.segment([v[0] for v in views])
    out = []
    for (rgb, depth, K, pose), inst in zip(views, results):
        H, W = depth.shape
        best, best_d = np.zeros((H, W), bool), np.inf
        info = dict(n_candidates=0, n_robot=0, chosen=None)
        sam_hand = np.zeros((H, W), bool)
        for m in inst.get("robot hand", dict(masks=[]))["masks"]:
            sam_hand |= m
        for prompt, d in inst.items():
            if prompt == "robot hand":
                continue
            for i, (m, score) in enumerate(zip(d["masks"], d["scores"])):
                info["n_candidates"] += 1
                v, u, p = pixel_points(depth, K, pose, m)
                if not len(p):
                    continue
                is_robot = (robot.query(p, distance_upper_bound=ROBOT_MARGIN)[0] < np.inf) | sam_hand[v, u]  # robot = FK removal or SAM "robot hand"
                if is_robot.mean() > MAX_ROBOT_FRAC:
                    info["n_robot"] += 1
                    continue
                v, u, p = v[~is_robot], u[~is_robot], p[~is_robot]
                dist = float(np.median(hand.query(p)[0]))
                if dist < best_d:
                    best_d = dist
                    best = np.zeros((H, W), bool)
                    best[v, u] = True
                    info["chosen"] = dict(prompt=prompt, index=i, score=float(score), n_px=int(len(p)), robot_frac=round(float(is_robot.mean()), 2), hand_dist_mm=round(dist * 1000, 1))
        if info["chosen"] is None:
            info["reason"] = "NO OBJECT FOUND in this view"
        out.append((best, info))
    return out


# ---- Start / goal identification (place): the object is what the hand holds at the start; the goal object is the one that looks like it ----
N_POS, N_NEG = 3, 5  # point prompt from the seeds: positives = the seeds farthest from the robot, negatives = visible FK hand pixels
START_THRESHOLD = 0.3  # text-mask confidence at the start frame (0.1 tried 2026-11-13: weak large masks containing seeds got picked -> worse)
USE_SEED_POINTS = False  # SAM point prompt from the seeds when no text mask covers them (tried 2026-11-13: slivers / wrong masks -> worse than the seeds alone)
MIN_SEED_PX = 3  # a SAM mask is the held object if it contains at least this many grasp-seed pixels (the most seeds wins)
MIN_GOAL_PX = 20
MIN_SIMILARITY = 0.4  # goal candidate colour similarity to the start object (1 - Bhattacharyya distance of hue-saturation histograms); below -> flagged
SIZE_RANGE = (0.5, 2.0)  # goal candidate 3D size / start object size
AGREE_M = 0.05  # two views' goal objects agree if their 3D centroids are this close; else the more similar one is kept


def _robot_and_points(depth, K, pose, robot):
    from lawm.sim.dexwm.planspace.production import ROBOT_MARGIN
    H, W = depth.shape
    v, u, pts = pixel_points(depth, K, pose, np.ones((H, W), bool))
    is_robot = robot.query(pts, distance_upper_bound=ROBOT_MARGIN)[0] < np.inf if len(pts) else np.zeros(0, bool)
    return v, u, pts, is_robot


def colour_hist(rgb, mask):
    """Normalised hue-saturation histogram (16 x 8) of the mask's pixels."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)


def similarity(h1, h2):
    import cv2
    return 1.0 - float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def size_of(points):
    return float((np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)).max()) if len(points) >= 5 else 0.0


def identify_start(segmenter, views, robot_points, hand_points):
    """Per view (mask, info) of the HELD object at the start frame, and the per-view descriptors (colour histogram, 3D size) for the goal match.
    Seeds = non-robot camera points inside the convex hull of the FK hand points (between the fingers and the palm); the object = the SAM mask with the most seed
    pixels (robot pixels removed); no such mask -> the seed pixels themselves; no seeds in any view -> nothing (info reason: flag the episode)."""
    from scipy.spatial import Delaunay, cKDTree
    robot, hull, hand_tree = cKDTree(np.asarray(robot_points, float)), Delaunay(np.asarray(hand_points, float)), cKDTree(np.asarray(hand_points, float))
    old_t, segmenter.threshold = segmenter.threshold, START_THRESHOLD  # the held object is half hidden by the fingers: accept weaker text masks here
    try:
        results = segmenter.segment([v[0] for v in views])
    finally:
        segmenter.threshold = old_t
    out, desc = [], []
    for (rgb, depth, K, pose), inst in zip(views, results):
        H, W = depth.shape
        v, u, pts, is_robot = _robot_and_points(depth, K, pose, robot)
        seed = ~is_robot & (hull.find_simplex(pts) >= 0) if len(pts) else np.zeros(0, bool)
        seed_px, not_robot = np.zeros((H, W), bool), np.zeros((H, W), bool)
        seed_px[v[seed], u[seed]] = True
        not_robot[v[~is_robot], u[~is_robot]] = True
        info = dict(n_seed_px=int(seed.sum()), chosen=None)
        best, best_n = None, MIN_SEED_PX - 1
        robot_px = np.zeros((H, W), bool)
        robot_px[v[is_robot], u[is_robot]] = True
        for i, m in enumerate(inst[segmenter.prompts[0]]["masks"]):
            if m.sum() == 0 or (m & robot_px).sum() / m.sum() > MAX_ROBOT_FRAC:  # mostly the hand / arm itself (FK): never the held object
                continue
            n = int((m & seed_px).sum())
            if n > best_n:
                best, best_n = m & not_robot, n
                info["chosen"] = dict(source="sam", index=i, score=float(inst[segmenter.prompts[0]]["scores"][i]), seed_px_in_mask=n)
        if best is None and seed.any() and USE_SEED_POINTS:  # no text mask covers the held object (fingers occlude it): prompt SAM with points from the seeds instead
            d_rob = robot.query(pts[seed])[0]
            pos = np.argsort(-d_rob)[:N_POS]  # the seeds farthest from the robot surface: surely object, not hand
            vis_hand = is_robot & (hand_tree.query(pts, distance_upper_bound=0.01)[0] < np.inf)
            hv, hu = v[vis_hand], u[vis_hand]
            neg = np.linspace(0, len(hu) - 1, N_NEG).astype(int) if len(hu) >= N_NEG else np.zeros(0, int)
            coords = np.r_[np.stack([u[seed][pos], v[seed][pos]], 1), np.stack([hu[neg], hv[neg]], 1)].astype(np.float32)
            labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
            res = segmenter.segment([rgb], points=[(coords, labels)])[0]
            cands = [] if res is None else [(int((m & seed_px).sum()), int(m.sum()), j) for j, m in enumerate(res["masks"]) if (m & not_robot).any()]
            if cands:  # of SAM's 3 nested masks (small -> large): the one covering the most seeds, then the larger one (the highest score is often a sliver)
                _, _, j = max(cands)
                best, info["chosen"] = res["masks"][j] & not_robot, dict(source="sam_points_from_seeds", mask=j, score=float(res["scores"][j]), n_pos=len(pos), n_neg=len(neg))
            else:
                best, info["chosen"] = seed_px, dict(source="seeds_only")
        if best is None and seed.any():
            best, info["chosen"] = seed_px, dict(source="seeds_only")
        if best is None:
            best, info["reason"] = np.zeros((H, W), bool), "NO OBJECT between the fingers in this view"
        _, _, p = pixel_points(depth, K, pose, best)
        desc.append(dict(hist=colour_hist(rgb, best), size=size_of(p)) if best.any() else None)
        out.append((best, info))
    return out, desc


def identify_goal(segmenter, views, robot_points, start_desc):
    """Per view (mask, info) of the start object in the goal frame: of the SAM masks (robot pixels removed, not mostly robot), the one most similar in colour to the
    start object in the same view (the cameras are fixed for the episode), with a 3D size within SIZE_RANGE of it; similarity below MIN_SIMILARITY -> nothing.
    If both views find one and their 3D centroids disagree by more than AGREE_M, only the more similar view is kept."""
    from scipy.spatial import cKDTree
    robot = cKDTree(np.asarray(robot_points, float))
    ref = [d for d in start_desc if d is not None]
    results = segmenter.segment([v[0] for v in views])
    out, cents, sims = [], [], []
    for vi, ((rgb, depth, K, pose), inst) in enumerate(zip(views, results)):
        H, W = depth.shape
        d0 = start_desc[vi] if start_desc[vi] is not None else (ref[0] if ref else None)
        v, u, pts, is_robot = _robot_and_points(depth, K, pose, robot)
        not_robot = np.zeros((H, W), bool)
        not_robot[v[~is_robot], u[~is_robot]] = True
        info, best, best_s, best_c = dict(chosen=None, candidates=[]), np.zeros((H, W), bool), -1.0, None
        if d0 is None:
            info["reason"] = "no start object to match"
        else:
            for i, m in enumerate(inst[segmenter.prompts[0]]["masks"]):
                mm = m & not_robot
                if mm.sum() < MIN_GOAL_PX or mm.sum() < 0.5 * m.sum():
                    continue
                _, _, p = pixel_points(depth, K, pose, mm)
                sz = size_of(p)
                s = similarity(d0["hist"], colour_hist(rgb, mm))
                ok_size = d0["size"] <= 0 or SIZE_RANGE[0] <= sz / d0["size"] <= SIZE_RANGE[1]
                info["candidates"].append(dict(index=i, sim=round(s, 2), size_ratio=round(sz / d0["size"], 2) if d0["size"] > 0 else None))
                if ok_size and s >= MIN_SIMILARITY and s > best_s:
                    best, best_s, best_c = mm, s, p.mean(0)
                    info["chosen"] = dict(source="sam", index=i, sim=round(s, 2))
            if info["chosen"] is None:
                info["reason"] = "NO MATCH for the start object in this view"
        out.append((best, info)); cents.append(best_c); sims.append(best_s)
    found = [i for i, c in enumerate(cents) if c is not None]
    if len(found) == 2 and np.linalg.norm(cents[0] - cents[1]) > AGREE_M:
        drop = found[int(np.argmin([sims[i] for i in found]))]
        m, info = out[drop]
        out[drop] = (np.zeros_like(m), dict(info, chosen=None, reason=f"disagrees with the other view by {np.linalg.norm(cents[0] - cents[1]) * 1000:.0f} mm, less similar -> dropped"))
    return out


GRIP_REACH_M = 0.08  # start mask: some point within this of the FK hand (or inside its hull)
MIN_CHANGED_FRAC = 0.6  # a goal SAM mask is the placed object only if most of it changed
CHANGE_M = 0.02  # a pixel changed between start and goal if its depth moved by more than this


def identify_goal_by_change(segmenter, start_view, goal_view, start_robot, goal_robot):
    """(mask, info) of the placed object in the goal frame from the SAME fixed camera: pixels whose depth changed between the start and goal frames, robot pixels
    (FK, either frame) excluded, are where the object arrived; the object = the SAM goal mask holding the most changed pixels (robot pixels removed)."""
    from scipy.spatial import cKDTree
    (_, d0, K, pose), (rgb, d1, _, _) = start_view, goal_view
    H, W = d1.shape
    rob = np.zeros((H, W), bool)  # robot pixels in the GOAL frame only (at the start the hand often covered the place spot)
    v, u, pts, is_robot = _robot_and_points(d1, K, pose, cKDTree(np.asarray(goal_robot, float)))
    rob[v[is_robot], u[is_robot]] = True
    valid = (d0 > 0.1) & (d1 > 0.1) & (d0 < 3) & (d1 < 3)
    changed = valid & ~rob & (np.abs(d1 - d0) > CHANGE_M)  # the goal frame differs here from the start frame
    info = dict(n_changed_px=int(changed.sum()), chosen=None)
    inst = segmenter.segment([rgb])[0][segmenter.prompts[0]]
    best, best_n = np.zeros((H, W), bool), 0
    for i, m in enumerate(inst["masks"]):
        n = int((m & changed).sum())
        if n > best_n and n >= MIN_CHANGED_FRAC * m.sum():  # most of the mask is new in the goal frame
            best, best_n = m & ~rob, n
            info["chosen"] = dict(source="sam_changed", index=i, changed_px_in_mask=n, area=int(m.sum()))
    if info["chosen"] is None:
        info["reason"] = "NO CHANGED OBJECT in the goal frame"
    return best, info


def identify_by_change(segmenter, start_view, goal_view, start_robot, goal_robot, start_hand):
    """Both ends of a place from ONE start-goal difference of the same fixed camera (DexWM's robot0_robotview_2).
    Only pixels that are robot in NEITHER frame (FK) are compared, so nothing the hand covered in one frame counts. Of those, depth nearer at the goal by > CHANGE_M
    = where the object ARRIVED (goal object); nearer at the start = where it WAS (the held object). In each frame, the object = the SAM mask (not mostly robot)
    holding the most of its frame's changed pixels (>= MIN_CHANGED_FRAC of the mask); at the start the mask must also lie within reach of the fingers
    (seeds inside the FK hand hull, or within GRIP_REACH_M of the hand). No SAM mask -> the changed pixels themselves. Returns (start_mask, goal_mask, info)."""
    from scipy.spatial import Delaunay, cKDTree
    (rgb0, d0, K, pose), (rgb1, d1, _, _) = start_view, goal_view
    H, W = d1.shape
    rob = np.zeros((H, W), bool)
    for depth, rp in ((d0, start_robot), (d1, goal_robot)):
        v, u, _, is_robot = _robot_and_points(depth, K, pose, cKDTree(np.asarray(rp, float)))
        rob[v[is_robot], u[is_robot]] = True
    valid = (d0 > 0.1) & (d1 > 0.1) & (d0 < 3) & (d1 < 3) & ~rob
    arrived, left = valid & (d1 < d0 - CHANGE_M), valid & (d0 < d1 - CHANGE_M)
    hand = cKDTree(np.asarray(start_hand, float))
    hull = Delaunay(np.asarray(start_hand, float))
    info = dict(arrived_px=int(arrived.sum()), left_px=int(left.sum()))
    res = segmenter.segment([rgb0, rgb1])

    def pick(inst, changed, depth, near_hand):
        best, key, why = None, (0, 0), None
        for i, m in enumerate(inst[segmenter.prompts[0]]["masks"]):
            if m.sum() == 0 or (m & rob).sum() / m.sum() > MAX_ROBOT_FRAC:
                continue
            n = int((m & changed).sum())
            if n == 0 or n < MIN_CHANGED_FRAC * (m & ~rob).sum():
                continue
            if near_hand:
                _, _, p = pixel_points(depth, K, pose, m & ~rob)
                if not len(p) or not ((hull.find_simplex(p) >= 0).any() or hand.query(p)[0].min() < GRIP_REACH_M):
                    continue
            if (n, int(m.sum())) > key:
                best, key, why = m & ~rob, (n, int(m.sum())), dict(source="sam", index=i, changed_px=n, area=int(m.sum()))
        if best is None and changed.sum() >= MIN_GOAL_PX:
            best, why = changed.copy(), dict(source="changed_px_only", area=int(changed.sum()))
        return (best if best is not None else np.zeros((H, W), bool)), why

    ms, info["start"] = pick(res[0], left, d0, near_hand=True)
    mg, info["goal"] = pick(res[1], arrived, d1, near_hand=False)
    return ms, mg, info


def identify_start_from_goal(segmenter, start_view, start_robot, start_hand, goal_rgb, goal_mask, goal_depth=None):
    """(mask, info) of the held object at the start, using the goal object (unoccluded, from identify_by_change) as the reference: of the SAM masks of the start
    frame that are not mostly robot and lie within reach of the fingers (inside the FK hand hull or within GRIP_REACH_M of it), the one most similar to the goal
    object in colour (seeds inside the mask break ties); similarity below MIN_SIMILARITY -> nothing (flag)."""
    from scipy.spatial import Delaunay, cKDTree
    rgb, depth, K, pose = start_view
    H, W = depth.shape
    v, u, pts, is_robot = _robot_and_points(depth, K, pose, cKDTree(np.asarray(start_robot, float)))
    rob = np.zeros((H, W), bool)
    rob[v[is_robot], u[is_robot]] = True
    hull, hand = Delaunay(np.asarray(start_hand, float)), cKDTree(np.asarray(start_hand, float))
    seed = ~is_robot & (hull.find_simplex(pts) >= 0)
    seed_px = np.zeros((H, W), bool)
    seed_px[v[seed], u[seed]] = True
    ref = colour_hist(goal_rgb, goal_mask)
    inst = segmenter.segment([rgb])[0][segmenter.prompts[0]]
    best, key, info = np.zeros((H, W), bool), None, dict(chosen=None, candidates=[])
    for i, m in enumerate(inst["masks"]):
        if m.sum() == 0 or (m & rob).sum() / m.sum() > MAX_ROBOT_FRAC:
            continue
        mm = m & ~rob
        _, _, p = pixel_points(depth, K, pose, mm)
        if not len(p) or not ((hull.find_simplex(p) >= 0).any() or hand.query(p)[0].min() < GRIP_REACH_M):
            continue
        s, n_seed = similarity(ref, colour_hist(rgb, mm)), int((mm & seed_px).sum())
        info["candidates"].append(dict(index=i, sim=round(s, 2), seeds=n_seed))
        if s >= MIN_SIMILARITY and (key is None or (round(s, 1), n_seed) > key):
            best, key = mm, (round(s, 1), n_seed)
            info["chosen"] = dict(source="sam_like_goal", index=i, sim=round(s, 2), seeds=n_seed)
    if info["chosen"] is None:
        info["reason"] = "NO START MASK like the goal object within reach of the fingers"
    return best, info
