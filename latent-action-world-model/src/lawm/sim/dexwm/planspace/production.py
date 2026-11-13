"""The production-level scene cloud of a DexWM kitchen, as the real cell sees it (lawm-deployment perception): the hand from its mesh at the measured joint state (FK, proprioception) and the
scene from 2 simulated RGB-D cameras (back-projected, cropped to a box around the hand's start, robot removed by distance to its own FK surface, one point per 1.5 cm voxel, subsampled to a fixed count).
No privileged scene state: no object poses, no meshes of the scene, no segmentation masks."""
from __future__ import annotations

import re

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from lawm.sim import clouds
from lawm.sim.costs import SEG_HAND, SEG_OBJECT, SEG_TABLE
from lawm.sim.dexwm.scene.scene_cloud import VISUAL_GROUP, top_level_body

SEG_SCENE = SEG_OBJECT  # every camera point, as lawm-deployment labels them
SEG_REST = SEG_TABLE  # with object masks (segment.py): camera points outside the object mask
HEIGHT, WIDTH = 240, 424
N_SCENE = 5120
VOXEL = 0.015
ROBOT_MARGIN = 0.015
NEAR, FAR = 0.1, 3.0
BOX_HALF = np.array([0.6, 0.6])  # metres around the hand's start position, horizontally
BOX_Z = (-0.45, 0.45)  # metres around the hand's start height
FREE_EYE = np.array([-0.7, 0.45, 0.6])  # the free camera's eye from the hand's start, in the robot base frame: behind, left of and above the hand
NAMED_CAMERA = "robot0_robotview"  # the kitchen's own robot-mounted side view
ROBOT_TOPS = re.compile(r"^(robot0_|mobilebase|gripper0_)")
# cameras="dexwm": DexWM's only camera and its model-input pixel grid (work/dexwm/sim_eval/distributed_controller.py:233-250, 576-583): robot0_robotview_2 (pose and fovy from the
# stored kitchen XML, which already holds DexWM's override) rendered at 600 x 960, rows 26..573 kept (548), resized to 224 x 392 (cv2 INTER_LINEAR; depth: INTER_NEAREST).
DEXWM_CAMERA = "robot0_robotview_2"
DEXWM_RENDER = (600, 960)
DEXWM_CROP = 26
DEXWM_OUT = (224, 392)
ALIGN_MAX_MM = 12.0  # set_start(): camera points near the robot must lie this close (median) to the FK robot surface


def back_project(depth, K, pose, near=NEAR, far=FAR, mask=None):
    """World points (n, 3) of the valid pixels of a depth image (m), intrinsics K, 4x4 pose of the optical frame (x right, y down, z forward); as lawm_deploy.perception.
    With a pixel mask (H, W) bool: (points, mask value of each point)."""
    v, u = np.nonzero((depth > near) & (depth < far))
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts = cam @ pose[:3, :3].T + pose[:3, 3]
    return pts if mask is None else (pts, mask[v, u])


def voxel_down(points, size=VOXEL, return_index=False):
    """One measured point (the first) per occupied voxel; as lawm_deploy.perception. return_index: the kept rows instead."""
    if size <= 0 or not len(points):
        return np.arange(len(points)) if return_index else points
    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    keep = np.sort(first)
    return keep if return_index else points[keep]


def robot_geoms(model):
    """Visual geoms of the robot (arm, mobile base, hand): every geom under a top-level body named like the robot."""
    out = []
    for g in range(model.ngeom):
        b = top_level_body(model, int(model.geom_bodyid[g]))
        if model.geom_group[g] == VISUAL_GROUP and ROBOT_TOPS.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""):
            out.append(g)
    return out


def dexwm_grid(rgb, depth, K):
    """DexWM's crop (rows DEXWM_CROP..-DEXWM_CROP) and resize to DEXWM_OUT of a 600 x 960 render, with the intrinsics moved onto the new pixel grid."""
    import cv2
    c = DEXWM_CROP
    rgb, depth = rgb[c:-c], depth[c:-c]
    (h0, w0), (h1, w1) = depth.shape, DEXWM_OUT
    sx, sy = w1 / w0, h1 / h0
    K = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5], [0, K[1, 1] * sy, (K[1, 2] - c + 0.5) * sy - 0.5], [0, 0, 1.0]])
    return cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_LINEAR), cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_NEAREST), K


class ProductionScene:
    """Cameras and perception of one kitchen; set_start(root_pose) places the free camera and the crop box from the hand's start (proprioception), then __call__(data, root_pose, q) gives the cloud."""

    def __init__(self, sim, seed=0, n_scene=N_SCENE, cameras="two", centre_pose=True):
        """cameras: "two" (free camera + robot0_robotview, 240 x 424; the planspace runs) or "dexwm" (robot0_robotview_2 only, on DexWM's 224 x 392 input grid).
        centre_pose: camera poses at the true (mono) centre (default). False = MuJoCo's left eye, the legacy pose of everything made before 2026-11-13 (see views)."""
        assert cameras in ("two", "dexwm"), cameras
        self.sim, self.model, self.n_scene, self.camera_set, self.centre_pose = sim, sim.model, n_scene, cameras, centre_pose
        self.rng = np.random.default_rng(seed)
        rh, rw = DEXWM_RENDER if cameras == "dexwm" else (HEIGHT, WIDTH)
        g = self.model.vis.global_
        g.offwidth, g.offheight = max(int(g.offwidth), rw), max(int(g.offheight), rh)
        self.renderer = mujoco.Renderer(self.model, rh, rw)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[VISUAL_GROUP] = 1
        self.opt.sitegroup[:] = 0
        self.robot = clouds.SurfaceSampler(self.model, robot_geoms(self.model), 20000, np.random.default_rng(seed + 1))
        self.cams = None

    def set_start(self, root_pose):
        """Fix the cameras and the crop box for an episode from the hand's start pose and the robot base (both known to the robot)."""
        m, d = self.model, self.sim.data
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        target = root_pose[:3, 3].copy()
        eye = target + d.xmat[base].reshape(3, 3) @ FREE_EYE
        free = mujoco.MjvCamera()
        free.type = mujoco.mjtCamera.mjCAMERA_FREE
        v = target - eye
        free.lookat[:], free.distance = target, float(np.linalg.norm(v))
        free.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        free.elevation = float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))
        if self.camera_set == "dexwm":
            named = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, DEXWM_CAMERA)
            assert named >= 0, f"no camera {DEXWM_CAMERA}"
            self.cams = [(named, float(m.cam_fovy[named]))]
        else:
            named = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, NAMED_CAMERA)
            self.cams = [(free, float(m.vis.global_.fovy)), (named, float(m.cam_fovy[named]))]
        self.lo = np.r_[target[:2] - BOX_HALF, target[2] + BOX_Z[0]]
        self.hi = np.r_[target[:2] + BOX_HALF, target[2] + BOX_Z[1]]
        self.alignment_mm = self.check_alignment(d)

    def check_alignment(self, data, max_mm=ALIGN_MAX_MM):
        """Median distance (mm, per view) from the camera points near the robot to its FK surface; raises if a view exceeds max_mm (a wrong camera pose: the robot ghosts
        through the robot removal and the scene is offset against the FK hand). Correct poses give ~7-9 mm (the FK surface sampling spacing); the legacy left-eye pose ~16-25 mm.
        Skipped (only reported) for the explicitly requested legacy pose."""
        tree = cKDTree(self.robot(data))
        out = []
        for rgb, depth, K, pose in self.views(data):
            d = tree.query(back_project(depth, K, pose))[0]
            near = d < 0.08
            out.append(float(np.median(d[near]) * 1000) if near.sum() > 200 else None)  # robot hardly in view: nothing to check
        bad = [m for m in out if m is not None and m > max_mm]
        if bad and self.centre_pose:
            raise RuntimeError(f"camera point cloud is misaligned with the FK robot: median {out} mm per view (> {max_mm} mm). Check the camera pose in production.views().")
        return out

    def views(self, data):
        """[(rgb uint8 (H, W, 3), depth m float32 (H, W), K (3, 3), optical pose (4, 4))] of both cameras for the current state."""
        out = []
        for cam, fovy in self.cams:
            r = self.renderer
            r.update_scene(data, camera=cam, scene_option=self.opt)
            r.disable_depth_rendering()
            rgb = r.render().copy()
            r.enable_depth_rendering()
            depth = r.render().astype(np.float32)
            r.disable_depth_rendering()
            gl = r.scene.camera[0]
            fwd, up = np.asarray(gl.forward, float), np.asarray(gl.up, float)
            pose = np.eye(4)
            pose[:3, :3] = np.stack([np.cross(fwd, up), -up, fwd], axis=1)
            # scene.camera[0] is MuJoCo's LEFT stereo eye, ipd/2 (34 mm by default) left of the camera; the mono render is from the centre between the eyes.
            # Correct (default): the centre. centre_pose=False reproduces the legacy left-eye pose that the play data / fine-tunes / runs before 2026-11-13 used
            # (every cloud ~34 mm off the FK hand / robot); set_start() refuses a misaligned camera unless legacy was asked for explicitly.
            pose[:3, 3] = 0.5 * (np.asarray(gl.pos, float) + np.asarray(r.scene.camera[1].pos, float)) if self.centre_pose else np.asarray(gl.pos, float)
            rh, rw = depth.shape
            f = 0.5 * rh / np.tan(np.radians(fovy) / 2)
            K = np.array([[f, 0, (rw - 1) / 2], [0, f, (rh - 1) / 2], [0, 0, 1.0]])
            if self.camera_set == "dexwm":
                rgb, depth, K = dexwm_grid(rgb, depth, K)
            out.append((rgb, depth, K, pose))
        return out

    def centred(self, views):
        """The views with the geometrically correct (mono centre) camera pose; unchanged if they already have it (the default)."""
        if self.centre_pose:
            return views
        out = []
        for rgb, depth, K, pose in views:
            pose = pose.copy()
            pose[:3, 3] += pose[:3, 0] * float(self.model.vis.global_.ipd) / 2
            out.append((rgb, depth, K, pose))
        return out

    def scene_points(self, views, data, masks=None):
        """Camera points inside the box, one per voxel, without the robot; then exactly n_scene of them (random subsample, or repeats if fewer).
        masks (one (H, W) bool per view, e.g. segment.object_masks): also the mask value of every returned point (a voxel keeps its first point's value)."""
        if masks is None:
            pts = np.concatenate([back_project(dep, K, pose) for _, dep, K, pose in views], axis=0)
            lab = np.zeros(len(pts), bool)
        else:
            both = [back_project(dep, K, pose, mask=m) for (_, dep, K, pose), m in zip(views, masks)]
            pts, lab = np.concatenate([b[0] for b in both], axis=0), np.concatenate([b[1] for b in both])
        inside = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
        pts, lab = pts[inside], lab[inside]
        keep = voxel_down(pts, return_index=True)
        pts, lab = pts[keep], lab[keep]
        if len(pts):
            near_robot = cKDTree(self.robot(data)).query(pts, distance_upper_bound=ROBOT_MARGIN)[0] < np.inf
            pts, lab = pts[~near_robot], lab[~near_robot]
        seen = len(pts)
        assert seen > 0, "no scene points"
        idx = self.rng.choice(seen, self.n_scene, replace=seen < self.n_scene)
        if masks is None:
            return pts[idx].astype(np.float32), seen
        return pts[idx].astype(np.float32), seen, lab[idx]

    def __call__(self, data, root_pose, q, masks=None):
        """(coords (n_hand + n_scene, 3), segment, views, n_scene_seen): FK hand points at the measured state, then the camera scene points.
        masks: a callable views -> one (H, W) bool object mask per view; the scene rows are then SEG_OBJECT inside the mask and SEG_REST outside."""
        k = self.sim.kinematics
        hand = k.points(root_pose[:3, :3], root_pose[:3, 3], q).astype(np.float32)
        views = self.views(data)
        if masks is None:
            scene, seen = self.scene_points(views, data)
            seg_scene = np.full(len(scene), SEG_SCENE)
        else:
            scene, seen, lab = self.scene_points(views, data, masks(views))
            seg_scene = np.where(lab, SEG_OBJECT, SEG_REST)
        segment = np.r_[np.full(len(hand), SEG_HAND), seg_scene].astype(np.int8)
        return np.concatenate([hand, scene]), segment, views, seen
