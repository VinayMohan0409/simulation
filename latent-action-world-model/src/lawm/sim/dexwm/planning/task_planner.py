"""The planner of one DexWM episode: the cost weights, phase gate and search budget of the task, one replan per call, and the hand points to execute."""
from __future__ import annotations

import numpy as np

from lawm.sim.costs import SEG_HAND, SEG_OBJECT
from lawm.sim.dexwm.planning.cem_replan import plan_replan, replan_rng
from lawm.sim.dexwm.planning.plan_costs import PlanCost
from lawm.sim.dexwm.planning.phase_gates import HoverGate, SecureGate

EXECUTED_STEP = 0  # of the 5 predicted steps only the first is executed, then the planner looks again


class TaskPlanner:
    """Plans the replans of one episode: grasp hovers before it descends, place secures the grip before it carries, reach has no phases."""

    def __init__(self, task, goal, segment, finger_mask, model, seed, settings, weights, gates, goal_object_centroid=None, goal_object_points=None):
        """Keep the goal cloud, the model adapter (model.prior = (mean, std) of the latents; model.decoder(coords, segment) -> decode(z) for an observed cloud) and the episode seed; the gate follows the task; weights = CostWeights (place: one per secure stage)."""
        self.task, self.goal, self.segment, self.finger_mask = task, goal, segment, finger_mask
        self.model, self.seed = model, seed
        self.settings, self.weights = settings, weights
        self.goal_object_centroid, self.goal_object_points = goal_object_centroid, goal_object_points
        staged_place = task == "place" and isinstance(weights, (list, tuple))  # single-stage place weights (e.g. planspace place_simple): no secure gate
        self.gate = HoverGate(goal, segment, gates.hover) if task == "grasp" else SecureGate(gates.secure, weights) if staged_place else None

    def plan(self, coords, replan, segment=None):
        """(hand points to execute (n_hand, 3) float32, the Plan) for the observed cloud at this replan; segment: this observation's labels when they change between observations."""
        segment = self.segment if segment is None else segment
        goal = self.gate.goal if isinstance(self.gate, HoverGate) else self.goal
        weights = self.gate.weights if isinstance(self.gate, SecureGate) else self.weights
        extra = {} if self.goal_object_centroid is None else dict(goal_object_centroid=self.goal_object_centroid)
        if self.goal_object_points is not None:
            extra["goal_object_points"] = self.goal_object_points
        cost = PlanCost(coords, goal, segment, self.finger_mask, weights, **extra)
        z_mean, z_std = self.model.prior
        plan = plan_replan(self.model.decoder(coords, self.model_segment(segment)), cost, z_mean, z_std, self.settings, replan_rng(self.seed, replan))
        hand = segment == SEG_HAND
        return (coords[hand] + plan.delta[EXECUTED_STEP][hand]).astype(np.float32), plan

    def model_segment(self, segment):
        """The labels the model adapter sees. With per-observation object masks (place, production scene + SAM 3) the scene rows are object / rest for the COST only; the
        action-to-latent map was trained with every scene row labelled object (production.SEG_SCENE), so it gets that back (its object-centroid feature = the scene centroid).
        Otherwise (reach, grasp, privileged scene) the segment is passed through unchanged."""
        if segment is self.segment:
            return segment
        return np.where(segment == SEG_HAND, SEG_HAND, SEG_OBJECT).astype(segment.dtype)

    def update(self, coords, replan):
        """After a replan was executed: let the gate look at the observed cloud; True at the moment the phase changes."""
        return self.gate is not None and self.gate.update(coords, self.segment, replan)
