"""The held object of a place episode WITHOUT any segmentation model and without object names: geometry of the RGB-D views + the FK hand/robot only.

  start_object      the place start: tabletop segmentation (Rusu et al.) -- drop the robot (FK surface), drop large horizontal support surfaces, Euclidean-cluster the
                    rest -- and take the compact cluster held by the fingers (inside the FK hand hull or within REACH_M of the hand).
  goal_object       the place target: the points that ARRIVED between the start and the goal frame of the same fixed cameras (depth nearer by > CHANGE_M), robot excluded,
                    clustered; the cluster whose size matches the start object.
  ObjectTracker     every later observation: the object points of the last observation are moved under two hypotheses -- HELD (rigidly with the FK hand, Kabsch on the
                    hand rows) and RELEASED (static) -- the hypothesis the new observation supports more wins; the object = the observed clusters it lands on.
                    Falls back to start_object when the support collapses; never runs a segmentation model.
All points are world frame (z up). Views: [(rgb, depth, K, pose)] as ProductionScene.views."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

VOXEL = 0.0075  # working resolution of the identification (the scene cloud itself is 1.5 cm)
ROBOT_M = 0.015  # a point this close to the FK hand points (dense) is robot
ARM_M = 0.035  # ... or this close to the FK robot surface away from the hand (the 20k-point arm/base sampling is sparse: ~2-3 cm spacing)
ARM_HAND_M = 0.05  # robot surface points within this of the hand get the small margin ROBOT_M (the held object touches the hand)
EPS = 0.015  # Euclidean clustering radius
PLANE_TOL = 0.01  # a point within this of a support height is on the support
PLANE_MIN_PTS = 400  # a horizontal flat component with at least this many (VOXEL) points is a support surface (~ 15 x 15 cm)
REACH_M = 0.03  # start: a cluster is held if it has points this close to the hand (or inside its hull)
MAX_EXTENT_M = 0.30  # an object is smaller than this along every axis
MIN_PTS = 8  # fewer points than this is not an object
CHANGE_M = 0.02  # goal: a pixel changed if its depth is nearer by more than this
TRACK_M = 0.015  # transport: an observed point within this of a moved object point supports the hypothesis
LOST_FRAC = 0.2  # transport: support below this fraction of the last object size -> re-identify from the hand


def view_points(view, lo=None, hi=None, mask=None):
    """(v, u, points) of the valid pixels of one view (optionally only inside the box lo/hi and the pixel mask)."""
    _, depth, K, pose = view
    ok = (depth > 0.1) & (depth < 3.0)
    if mask is not None:
        ok &= mask
    v, u = np.nonzero(ok)
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1) @ pose[:3, :3].T + pose[:3, 3]
    if lo is not None:
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        v, u, pts = v[inside], u[inside], pts[inside]
    return v, u, pts


def voxel(points, size=VOXEL):
    if not len(points):
        return points
    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def clusters(points, eps=EPS):
    """Connected components of the eps-neighbour graph: (n,) labels."""
    if not len(points):
        return np.zeros(0, int)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    return connected_components(g, directed=False)[1]


def support_mask(points):
    """Points on large horizontal support surfaces (counter, shelves, sink floor): height-histogram peaks, kept only where the flat points form a big connected sheet,
    so a small object crossing that height is not cut."""
    out = np.zeros(len(points), bool)
    if not len(points):
        return out
    z = points[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + PLANE_TOL, PLANE_TOL / 2))
    for i in np.argsort(hist)[::-1][:6]:
        if hist[i] < PLANE_MIN_PTS / 4:
            break
        c = 0.5 * (edges[i] + edges[i + 1])
        flat = np.abs(z - c) < PLANE_TOL
        idx = np.nonzero(flat & ~out)[0]
        if len(idx) < PLANE_MIN_PTS:
            continue
        lab = clusters(points[idx])
        sizes = np.bincount(lab)
        out[idx[sizes[lab] >= PLANE_MIN_PTS]] = True
    return out


def robot_free(pts, hand, robot):
    """The points that are not robot: farther than ROBOT_M from the FK hand points and farther than ARM_M from the FK robot surface away from the hand."""
    if not len(pts):
        return pts
    keep = np.ones(len(pts), bool)
    if hand is not None:
        hand = np.asarray(hand, float)
        keep &= cKDTree(hand).query(pts, distance_upper_bound=ROBOT_M)[0] == np.inf
    if robot is not None:
        robot = np.asarray(robot, float)
        near = cKDTree(hand).query(robot, distance_upper_bound=ARM_HAND_M)[0] < np.inf if hand is not None else np.zeros(len(robot), bool)
        for rb, margin in ((robot[near], ROBOT_M), (robot[~near], ARM_M)):
            if len(rb):
                keep &= cKDTree(rb).query(pts, distance_upper_bound=margin)[0] == np.inf
    return pts[keep]


def scene_points(views, hand, robot, lo=None, hi=None):
    """Non-robot camera points of all views at VOXEL resolution, and the same without the support surfaces: (points, free points)."""
    pts = robot_free(voxel(np.concatenate([view_points(v, lo, hi)[2] for v in views])), hand, robot)
    return pts, pts[~support_mask(pts)]


def extent(p):
    return np.ptp(p, axis=0) if len(p) else np.zeros(3)


def near_hand(points, hand, hull=None, reach=REACH_M):
    hull = Delaunay(hand) if hull is None else hull
    return (hull.find_simplex(points) >= 0) | (cKDTree(hand).query(points)[0] < reach)


def _hit(free, pts, r=1.5 * VOXEL):
    """(n_free,) bool: free points within r of pts."""
    return cKDTree(pts).query(free)[0] < r if len(pts) and len(free) else np.zeros(len(free), bool)


def start_object(views, hand, robot, lo=None, hi=None, left=None):
    """(object points, info) at the place start: of the non-support clusters within reach of the hand (inside the FK hand hull or within REACH_M) that have
    LEFT points (it is gone from there by the goal frame: the object that is carried; a thing merely next to the hand stays), most concentrated at the fingers. Without left points (no goal frame given, or
    the object did not move): the one with the most points within reach, among those mostly within reach. The fingers often split the object: the other pieces that are
    next to it and also left (or held) are added while it stays object-sized."""
    hand = np.asarray(hand, float)
    _, free = scene_points(views, hand, robot, lo, hi)
    info = dict(n_free=len(free), source=None)
    if not len(free):
        info["flag"] = "not_visible"
        return free, info
    lab = clusters(free)
    held = near_hand(free, hand)
    gone = _hit(free, left) if left is not None else np.zeros(len(free), bool)
    ok = lambda m: m.sum() >= MIN_PTS and extent(free[m]).max() <= MAX_EXTENT_M
    cand = [c for c in np.unique(lab[held]) if ok(lab == c)]
    score = {c: (int((gone & (lab == c)).sum()), int((held & (lab == c)).sum())) for c in cand}
    moving = [c for c in cand if score[c][0] >= MIN_PTS]
    if moving:
        # among the moving clusters near the hand, the one most concentrated at the fingers (held points x held fraction): a carried object is mostly within reach,
        # a bigger thing the hand bumps into (and moves) only partly
        best_c, info["source"] = max(moving, key=lambda c: (score[c][1] ** 2 / (lab == c).sum(), score[c][0])), "held_and_left"
        extra = lambda m: (gone & m).sum() >= 0.5 * m.sum()
    else:
        mostly = [c for c in cand if score[c][1] >= 0.3 * (lab == c).sum()]
        if not mostly:
            info["flag"] = "not_visible"
            return np.zeros((0, 3)), info
        best_c, info["source"] = max(mostly, key=lambda c: score[c][1]), "held"
        extra = lambda m: (held & m).sum() >= 3
    best = lab == best_c
    for c in np.unique(lab):
        m = lab == c
        if c == best_c or not extra(m):
            continue
        if cKDTree(free[best]).query(free[m])[0].min() < 2 * REACH_M and ok(best | m):
            best = best | m
    info.update(n_points=int(best.sum()), n_held=int((best & held).sum()), n_left=int((best & gone).sum()), n_pieces=len(np.unique(lab[best])),
                extent_cm=np.round(extent(free[best]) * 100, 1).tolist())
    return free[best], info


def changed_points(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, lo=None, hi=None):
    """(arrived, left) world points, VOXEL resolution: valid in both frames of the same camera, not robot in either frame (FK surface), depth nearer at the goal (arrived)
    or at the start (left) by more than CHANGE_M."""
    arrived, left = [], []
    for sv, gv in zip(start_views, goal_views):
        d0, d1 = sv[1], gv[1]
        valid = (d0 > 0.1) & (d1 > 0.1) & (d0 < 3) & (d1 < 3)
        arrived.append(view_points(gv, lo, hi, valid & (d1 < d0 - CHANGE_M))[2])
        left.append(view_points(sv, lo, hi, valid & (d0 < d1 - CHANGE_M))[2])
    out = []
    for pts in (np.concatenate(arrived), np.concatenate(left)):
        pts = voxel(pts)
        for h, rb in ((start_hand, start_robot), (goal_hand, goal_robot)):  # a point the robot covers in either frame says nothing about the object
            pts = robot_free(pts, h, rb)
        out.append(pts)
    return out


def kabsch(a, b):
    """R, t with R a + t ~ b (rows)."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1, 1, d]) @ u.T
    return R, cb - R @ ca


def transport(obj, hand0, hand1, free, ref_extent=None):
    """(object points in the new frame or None, info): the old object points moved HELD (rigidly with the hand, Kabsch hand0 -> hand1) or RELEASED (static); the hypothesis with
    the most free points within TRACK_M wins (ties, e.g. a still hand: held); the object = the free clusters it lands on (only the near points if they merged with
    something bigger, e.g. set down on a plate). None when the support is below LOST_FRAC of the object."""
    R, t = kabsch(np.asarray(hand0, float), np.asarray(hand1, float))
    hyps = dict(held=obj @ R.T + t, released=obj)
    tree = cKDTree(free) if len(free) else None
    support = {k: (int((tree.query(p)[0] < TRACK_M).sum()) if tree is not None else 0) for k, p in hyps.items()}
    name = max(support, key=support.get)
    info = dict(support=support, hypothesis=name, support_frac=round(support[name] / max(len(obj), 1), 2))
    if support[name] < max(MIN_PTS, LOST_FRAC * len(obj)):
        return None, info
    lab = clusters(free)
    d, nn = tree.query(hyps[name])
    on = np.unique(lab[nn[d < TRACK_M]])
    m = np.isin(lab, on)
    ref = extent(obj) if ref_extent is None else ref_extent
    if extent(free[m]).max() > max(1.5 * ref.max(), 0.05):
        info["merged"] = True
        return free[np.unique(nn[d < TRACK_M])], info
    return free[m], info


def goal_object(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, start_obj, lo=None, hi=None, arrived=None):
    """(object points in the goal frame, info). Two candidates: the ARRIVED cluster (non-support goal cluster made mostly of points that are new since the start; the one
    with the most new points) and the start object TRANSPORTED (held / released). The transport wins when it is well supported (>= 0.5 of the start object: e.g. the
    object barely moved, so nothing 'arrived'), or when there is no arrived cluster."""
    if arrived is None:
        arrived, _ = changed_points(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, lo, hi)
    _, free = scene_points(goal_views, goal_hand, goal_robot, lo, hi)
    info = dict(n_arrived=len(arrived), source=None)
    if not len(free):
        info["flag"] = "not_visible"
        return np.zeros((0, 3)), info
    lab = clusters(free)
    hit = _hit(free, arrived)
    best, best_n = None, 0
    for c in np.unique(lab[hit]):
        m = lab == c
        n_new = int((m & hit).sum())
        if n_new >= MIN_PTS and n_new >= 0.5 * m.sum() and extent(free[m]).max() <= MAX_EXTENT_M and n_new > best_n:  # mostly new where it is now, and object-sized
            best, best_n = m, n_new
    tr, tinfo = transport(start_obj, start_hand, goal_hand, free) if len(start_obj) else (None, {})
    info["transport"] = tinfo
    if tr is not None and (best is None or tinfo["support_frac"] >= 0.5):
        info.update(source=f"transport_{tinfo['hypothesis']}", n_points=len(tr), extent_cm=np.round(extent(tr) * 100, 1).tolist())
        return tr, info
    if best is None:
        info["flag"] = "not_visible"
        return np.zeros((0, 3)), info
    info.update(source="arrived_cluster", n_points=int(best.sum()), n_new=best_n, extent_cm=np.round(extent(free[best]) * 100, 1).tolist())
    return free[best], info


def identify(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, lo=None, hi=None):
    """(start object points, goal object points, info) of a place episode from its start and goal frames (same fixed cameras)."""
    arrived, left = changed_points(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, lo, hi)
    so, si = start_object(start_views, start_hand, start_robot, lo, hi, left=left)
    go, gi = goal_object(start_views, goal_views, start_hand, start_robot, goal_hand, goal_robot, so, lo, hi, arrived=arrived)
    return so, go, dict(start=si, goal=gi)


class ObjectTracker:
    """Object points through the observations of one episode (no segmentation model): from the start object, transport (held / released) to every new observation.
    When the transport loses support (e.g. the object was set down or slipped between two observations): the cluster of points that ARRIVED since the last observation
    (same fixed cameras, robot excluded) nearest the transported object, within NEAR_M; then start_object (re-identify from the hand); then the moved points themselves (flag 'lost')."""

    NEAR_M = 0.12

    def __init__(self, start_obj, start_hand, lo=None, hi=None, start_views=None, start_robot=None):
        self.obj, self.hand = np.asarray(start_obj, float), np.asarray(start_hand, float)
        self.lo, self.hi, self.ref_extent = lo, hi, extent(self.obj)
        self.views, self.robot = start_views, start_robot

    def _arrived_near(self, views, hand, robot, free, around):
        if self.views is None or not len(around) or not len(free):
            return None
        arrived, _ = changed_points(self.views, views, self.hand, self.robot, hand, robot, self.lo, self.hi)
        hit = _hit(free, arrived)
        if hit.sum() < MIN_PTS:
            return None
        lab = clusters(free)
        tree, best, best_d = cKDTree(around), None, self.NEAR_M
        for c in np.unique(lab[hit]):
            m = lab == c
            if (m & hit).sum() < MIN_PTS or (m & hit).sum() < 0.5 * m.sum() or extent(free[m]).max() > MAX_EXTENT_M:
                continue
            d = float(np.median(tree.query(free[m])[0]))
            if d < best_d:
                best, best_d = m, d
        return None if best is None else free[best]

    def update(self, views, hand, robot):
        """(object points, info) for this observation."""
        hand = np.asarray(hand, float)
        _, free = scene_points(views, hand, robot, self.lo, self.hi)
        obj, info = (None, {}) if not len(self.obj) else transport(self.obj, self.hand, hand, free, self.ref_extent)
        info["source"] = "transport"
        if obj is None and len(self.obj):
            R, t = kabsch(self.hand, hand)
            moved = self.obj @ R.T + t if info.get("hypothesis") == "held" else self.obj
            obj = self._arrived_near(views, hand, robot, free, np.concatenate([moved, self.obj]))
            if obj is not None:
                info["source"] = "arrived_near"
        if obj is None:
            obj, info2 = start_object(views, hand, robot, self.lo, self.hi)
            info.update(source="reidentify", **{k: v for k, v in info2.items() if k != "source"})
            if not len(obj) and len(self.obj):  # keep the moved points themselves (positions known from the last observation), flagged
                obj, info["flag"] = moved, "lost"
        self.obj, self.hand, self.views, self.robot = obj, hand, views, robot
        info["n_points"] = len(obj)
        return obj, info


def label_rows(points, obj, r=0.012):
    """(n,) bool: scene rows within r of the object points."""
    if not len(obj) or not len(points):
        return np.zeros(len(points), bool)
    return cKDTree(np.asarray(obj, float)).query(np.asarray(points, float))[0] < r


def pixel_mask(view, obj, lo=None, hi=None, r=0.012):
    """(H, W) bool: the pixels of a view whose point lies within r of the object points (for scoring / overlays)."""
    v, u, pts = view_points(view, lo, hi)
    m = np.zeros(view[1].shape, bool)
    sel = label_rows(pts, obj, r)
    m[v[sel], u[sel]] = True
    return m
