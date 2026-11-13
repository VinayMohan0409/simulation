"""The simulator of one DexWM demo for the episode loop: one kitchen switched between the start and target states of the tasks, the scene cloud and goal, the success flag and the task distance."""
from __future__ import annotations

import json
from dataclasses import dataclass

import mujoco
import numpy as np

from lawm.sim.dexwm.control import controller
from lawm.sim.dexwm.control.hand_fit import HandKinematics
from lawm.sim.dexwm.env import benchmark
from lawm.sim.dexwm.env.observation import Observation
from lawm.sim.dexwm.scene.hand_surface import hand_finger_mask
from lawm.sim.dexwm.scene.scene_cloud import DexWMScene

TIP_BODIES = ("robot0_right_hand", "gripper0_right_link_3.0", "gripper0_right_link_7.0", "gripper0_right_link_11.0", "gripper0_right_link_15.0")  # what DexWM's reach check measures


@dataclass(frozen=True)
class EnvSettings:
    """The kitchen of a demo: arm stiffness and damping ratio, finger PD gains, the control frequency (Hz), whether the offscreen renderer exists (for DexWMEnv.render) and the points per entity of the scene cloud."""

    arm_kp: float
    arm_damping: float
    finger_kp: float
    finger_kd: float
    control_freq: int
    cameras: bool  # on: the renderer is built and DexWMEnv.render draws images when asked; the steps never draw
    n_hand: int
    n_object: int
    n_table: int
    n_distractor: int


def build_env(env_args, control_freq, cameras=False):
    """The RoboCasa environment of a stored episode with DexWM's evaluation overrides (absolute arm targets in the world frame); cameras = build the offscreen renderer, but no camera is drawn at any step."""
    import robosuite

    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(env_name=env_args["env_name"], has_renderer=False, has_offscreen_renderer=cameras, use_camera_obs=False, ignore_done=True, control_freq=control_freq)
    for side in ("right", "left"):
        kwargs["controller_configs"]["body_parts"][side]["input_type"] = "absolute"
    kwargs["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = "world"
    return robosuite.make(**kwargs)


def open_demo(episode, settings, fast=True):
    """Build the kitchen of a stored episode and wrap it. fast: one build straight from the demo (its ep_meta for the Python-side fixtures and objects, its saved
    scene xml for the sim); the old path (fast=False) builds a random kitchen, rebuilds it from ep_meta, then compiles the saved xml (DexWM's reset_to)."""
    import robocasa.scripts.playback_utils as playback

    if fast:
        env = build_env_from_demo(episode, settings.control_freq, settings.cameras)
        env.sim.set_state_from_flattened(episode["states"][0])
        env.sim.forward()
    else:
        env = build_env(json.loads(episode["env_args"]), settings.control_freq, settings.cameras)
        playback.reset_to(env, dict(states=episode["states"][0], model=episode["model_file"], ep_meta=episode["ep_meta"]))
    return DexWMEnv(env, episode, settings)


def build_env_from_demo(episode, control_freq, cameras=False):
    """build_env, but the constructor's single build is already the demo's kitchen: _load_model gets the demo's ep_meta (layout, style, fixtures, object configs;
    no random sampling), the sim is compiled once from the demo's saved xml, and object placement sampling is skipped (the stored state sets every pose)."""
    import robosuite
    import robocasa.scripts.playback_utils as playback
    from robosuite.environments.base import REGISTERED_ENVS

    env_args = json.loads(episode["env_args"])
    ep_meta = json.loads(episode["ep_meta"]) if episode.get("ep_meta") is not None else {}
    cls = REGISTERED_ENVS[env_args["env_name"]]
    load0, init0 = cls._load_model, cls._initialize_sim

    def load(self):
        self._ep_meta = dict(ep_meta)
        self.deterministic_reset = True  # the constructor's reset must not sample placements
        try:
            load0(self)
        except (FileNotFoundError, PermissionError):  # as reset_to: object assets moved since recording
            if "object_cfgs" in self._ep_meta:
                self._ep_meta["object_cfgs"] = playback.update_mjcf_paths(self._ep_meta["object_cfgs"])
            load0(self)

    def init_sim(self, xml_string=None):
        if xml_string is not None:
            return init0(self, xml_string)
        xml = playback._prepare_xml(self, episode["model_file"])
        try:
            init0(self, xml)
        except (FileNotFoundError, PermissionError, ValueError):  # as reset_to: asset paths of another machine
            init0(self, playback.path_change(xml))

    own = {k: cls.__dict__[k] for k in ("_load_model", "_initialize_sim") if k in cls.__dict__}
    cls._load_model, cls._initialize_sim = load, init_sim
    try:
        env = build_env(env_args, control_freq, cameras)
    finally:
        for k in ("_load_model", "_initialize_sim"):
            if k in own:
                setattr(cls, k, own[k])
            else:
                delattr(cls, k)
    env.deterministic_reset = False
    return env


class DexWMEnv:
    """One demo's kitchen: reset(task) switches to the task's start state and sets the goal from its target state; then observe(), step(action) and distance() serve the episode loop."""

    def __init__(self, env, episode, settings, make_wrapper=None):
        """Keep the environment, set the arm gains and the finger torque law once, and look up the bodies the success checks use."""
        if make_wrapper is None:
            from lawm.sim.dexwm.success.sim_wrappers import EvaluationWrapper as make_wrapper
        self.env, self.episode, self.make_wrapper = env, episode, make_wrapper
        self.model, self.data = env.sim.model._model, env.sim.data._data
        self.ctrl = env.robots[0].composite_controller.part_controllers["right"]
        controller.set_arm_gains(self.ctrl, settings.arm_kp, settings.arm_damping)
        controller.set_finger_pd(env, settings.finger_kp, settings.finger_kd)
        self.tip_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TIP_BODIES]
        self.object_id = int(env.obj_body_id[env.objects["obj"].name])
        self.wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, env.robots[0].gripper["right"].root_body)
        self.task, self.cameras, self.settings = None, settings.cameras, settings

    def switch(self, frame):
        """Set the stored state of a frame and clear what the last episode left in the solver and the controllers."""
        self.env.sim.set_state_from_flattened(self.episode["states"][frame])
        self.env.sim.forward()
        for name in ("update_sites", "update_state"):
            if hasattr(self.env, name):
                getattr(self.env, name)()
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.ctrl.update(force=True)
        for part in self.env.robots[0].composite_controller.part_controllers.values():
            if hasattr(part, "reset_goal"):
                part.reset_goal()

    def reset(self, task, scene_seed=0):
        """Start the task: build the scene cloud, take the goal cloud and the success targets from the target state, and return the first observation."""
        start, target = benchmark.get_indices(self.episode["waypoints"], task)
        self.switch(target)
        target_hand = np.array([self.data.xpos[i] for i in self.tip_ids])
        target_object = self.data.xpos[self.object_id].copy()
        self.switch(start)
        centers = [self.data.xpos[self.object_id].copy(), target_object]
        self.scene = DexWMScene(self.model, self.data, centers, seed=scene_seed, n_hand=self.settings.n_hand, n_object=self.settings.n_object, n_table=self.settings.n_table,
                                n_distractor=self.settings.n_distractor, object_body=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_id))
        self.kinematics = HandKinematics(self.model, self.scene.hand)
        self.finger_mask = hand_finger_mask(self.model, self.scene.hand.owner)
        self.switch(target)
        self.goal, self.segment = self.scene(self.data)
        self.switch(start)
        targets = dict(reach=dict(target_hand_pose=target_hand), place=dict(target_obj_pos=target_object), grasp={})
        self.wrapped = self.make_wrapper(self.env, task_name=task, **targets[task])
        self.task, self.target_hand, self.target_object = task, target_hand, target_object
        return self.observe()

    def observe(self):
        """The scene cloud, the hand root pose, the finger angles and the controller's reference pose of the current state."""
        self.ctrl.update(force=True)
        root = self.kinematics.bodies[self.kinematics.root]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = self.data.xmat[root].reshape(3, 3), self.data.xpos[root]
        return Observation(self.scene(self.data)[0], pose, self.data.qpos[self.kinematics.qpos_adr].copy(), np.array(self.ctrl.ref_pos), np.array(self.ctrl.ref_ori_mat))

    def step(self, action):
        """One control step; True when DexWM's success check of the task holds afterwards."""
        return bool(self.wrapped.step(action)[1])

    def distance(self):
        """The task's distance (m): reach = mean distance of the five hand bodies to their target, grasp = wrist to object, place = object to its target."""
        if self.task == "reach":
            return float(np.linalg.norm(np.array([self.data.xpos[i] for i in self.tip_ids]) - self.target_hand, axis=1).mean())
        if self.task == "grasp":
            return float(np.linalg.norm(self.data.xpos[self.wrist_id] - self.data.xpos[self.object_id]))
        return float(np.linalg.norm(self.data.xpos[self.object_id] - self.target_object))

    def render(self, camera, height, width, depth=False):
        """An upright image (height, width, 3) from a named camera, with its depth in metres when depth=True; only when EnvSettings.cameras is on."""
        assert self.cameras, "EnvSettings.cameras is off: no renderer was built"
        out = self.env.sim.render(width=width, height=height, camera_name=camera, depth=depth)
        if not depth:
            return out[::-1]
        from robosuite.utils.camera_utils import get_real_depth_map

        return out[0][::-1], get_real_depth_map(self.env.sim, out[1])[::-1]
