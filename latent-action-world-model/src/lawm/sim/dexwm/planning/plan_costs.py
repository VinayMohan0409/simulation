"""The DexWM planner's cost terms on predicted point displacements (millimetres) and the weighted sum the planner minimises."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from lawm.sim.costs import SEG_HAND, SEG_OBJECT, horizon_weights

CONTACT_THRESHOLD_M = 0.04  # a hand point closer than this to the predicted object counts as touching it


def hand_goal_cost_mm(pred_deltas, coords0, goal, hand_mask, weights, point_weights=None):
    """Horizon-weighted mean per-point L2 distance (mm) of the predicted hand points to the goal hand points, optionally weighted per point."""
    start, target = coords0[None, hand_mask, :], goal[None, hand_mask, :]
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0:
            continue
        dist = np.linalg.norm(start + pred_deltas[:, k][:, hand_mask, :] - target, axis=-1) * 1000.0
        if point_weights is not None:
            step_cost = (dist * point_weights[None, :]).sum(axis=-1) / point_weights.sum()
        else:
            step_cost = dist.mean(axis=-1)
        total += wk * step_cost
    return total


def object_still_cost_mm(pred_deltas, mask, weights):
    """Horizon-weighted mean predicted displacement (mm) of the masked points: penalises moving the object."""
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        total += wk * np.linalg.norm(pred_deltas[:, k][:, mask, :], axis=-1).mean(axis=-1) * 1000.0
    return total


def object_goal_cost_mm(pred_deltas, mask, weights, goal_centroid, start_centroid):
    """Horizon-weighted distance (mm) of the predicted object centroid to the goal object centroid."""
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        pred_c = start_centroid[None] + pred_deltas[:, k][:, mask, :].mean(axis=1)
        total += wk * np.linalg.norm(pred_c - goal_centroid[None], axis=-1) * 1000.0
    return total


def object_chamfer_cost_mm(pred_deltas, mask, weights, start_object, goal_object_points):
    """Horizon-weighted one-sided Chamfer distance (mm): the mean distance of each predicted object point to the nearest goal-object point. One-sided (predicted -> goal) on
    purpose: under partial views the visible start part only has to land ON the goal object, not cover it; far from the goal it behaves like the centroid distance."""
    tree = cKDTree(np.asarray(goal_object_points, np.float64))
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        pred = start_object[None] + pred_deltas[:, k][:, mask, :]
        total += wk * tree.query(pred.reshape(-1, 3))[0].reshape(pred.shape[:2]).mean(axis=1) * 1000.0
    return total


def attachment_cost_mm(pred_deltas, hand_mask, obj_mask, weights, start_hand_centroid, start_object_centroid):
    """Horizon-weighted growth (mm) of the predicted hand-centroid to object-centroid distance over its start value: penalises the object moving away from the hand (0 while it stays or comes closer)."""
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    offset0 = float(np.linalg.norm(start_object_centroid - start_hand_centroid))
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        obj_c = start_object_centroid[None] + pred_deltas[:, k][:, obj_mask, :].mean(axis=1)
        hand_c = start_hand_centroid[None] + pred_deltas[:, k][:, hand_mask, :].mean(axis=1)
        total += wk * np.maximum(np.linalg.norm(obj_c - hand_c, axis=-1) - offset0, 0.0) * 1000.0
    return total


def kabsch_batched(reference, points):
    """Rotations (..., 3, 3) and translations (..., 3) with R @ reference[i] + t ~= points[..., i, :]; reference (n, 3), points (..., n, 3)."""
    ca = reference.mean(axis=0)
    cb = points.mean(axis=-2)
    h = np.einsum("ni,...nj->...ij", reference - ca, points - cb[..., None, :])
    u, _, vt = np.linalg.svd(h)
    ut = np.swapaxes(u, -1, -2)
    v = np.swapaxes(vt, -1, -2)
    det = np.linalg.det(np.einsum("...ij,...jk->...ik", v, ut))
    d = np.zeros(det.shape + (3, 3))
    d[..., 0, 0] = 1.0
    d[..., 1, 1] = 1.0
    d[..., 2, 2] = np.sign(det)
    r = np.einsum("...ij,...jk,...kl->...il", v, d, ut)
    return r, cb - np.einsum("...ij,j->...i", r, ca)


def rigid_attachment_cost_mm(coords0_hand, coords0_obj, pred_deltas, hand_mask, obj_mask, weights):
    """Horizon-weighted mean drift (mm) of the hand points from moving rigidly with the predicted object motion."""
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        pred_obj = coords0_obj[None] + pred_deltas[:, k][:, obj_mask, :]
        pred_hand = coords0_hand[None] + pred_deltas[:, k][:, hand_mask, :]
        r, t = kabsch_batched(coords0_obj, pred_obj)
        hand_in_start = np.einsum("pij,phi->phj", r, pred_hand - t[:, None, :])
        total += wk * np.linalg.norm(hand_in_start - coords0_hand[None, :, :], axis=-1).mean(axis=-1) * 1000.0
    return total


def grasp_tight_cost_mm(pred_deltas, hand_mask, obj_mask, weights, start_object_centroid, start_fingers, finger_select):
    """Horizon-weighted mean distance (mm) of the finger points to the predicted object centroid; finger_select picks the finger rows of the hand (None: all)."""
    total = np.zeros(pred_deltas.shape[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        if wk == 0.0:
            continue
        pred_c = start_object_centroid[None] + pred_deltas[:, k][:, obj_mask, :].mean(axis=1)
        hand_deltas = pred_deltas[:, k][:, hand_mask, :]
        finger_deltas = hand_deltas[:, finger_select] if finger_select is not None else hand_deltas
        d = np.linalg.norm(start_fingers[None] + finger_deltas - pred_c[:, None, :], axis=-1).mean(axis=-1)
        total += wk * d * 1000.0
    return total


def grasp_quality_penalty(coords0_hand, coords0_obj, pred_deltas, hand_mask, obj_mask, finger_select=None, contact_threshold_m=CONTACT_THRESHOLD_M, step=-1):
    """Antipodal grasp penalty per candidate at one step: 0 when two touching finger points face each other across the object, 1 with fewer than two touching points."""
    k = step if step >= 0 else pred_deltas.shape[1] + step
    pred_obj = coords0_obj[None] + pred_deltas[:, k][:, obj_mask, :]
    pred_hand = coords0_hand[None] + pred_deltas[:, k][:, hand_mask, :]
    if finger_select is not None:
        pred_hand = pred_hand[:, finger_select]
    penalty = np.ones(pred_obj.shape[0], dtype=np.float64)
    for p in range(len(penalty)):
        distance, nearest = cKDTree(pred_obj[p]).query(pred_hand[p])
        touching = distance < contact_threshold_m
        if touching.sum() < 2:
            continue
        to_obj = pred_obj[p][nearest[touching]] - pred_hand[p][touching]
        unit = to_obj / (np.linalg.norm(to_obj, axis=1, keepdims=True) + 1e-9)
        opposite = -(unit @ unit.T)
        np.fill_diagonal(opposite, -1.0)
        penalty[p] = 1.0 - max(float(opposite.max()), 0.0)
    return penalty


@dataclass
class CostWeights:
    """The weights of one phase of one task; the phase gates change them between replans."""

    hand_scale: float = 1.0
    fingertip: float = 1.0
    object_still: float = 0.0
    object_goal: float = 0.0
    grasp_quality: float = 0.0
    rigid_attachment: float = 0.0
    grasp_tight: float = 0.0
    attachment: float = 0.0


class PlanCost:
    """The planner's cost of a population of predicted displacements (pop, n_steps, N, 3) for one replan: the scaled hand goal cost plus the weighted extra terms."""

    def __init__(self, coords0, goal, segment, finger_mask, weights, n_steps=5, last_weight=0.5, goal_object_centroid=None, goal_object_points=None):
        """Fix everything that does not depend on the candidates: masks, centroids and the start points of the fingers.
        goal_object_centroid: given when the goal's object rows are not the observed object rows (production scene with object masks: the rows change every observation).
        goal_object_points: the observed goal-object points (partial view); the object-goal term is then the one-sided Chamfer distance instead of the centroid distance."""
        self.coords0, self.goal, self.weights = coords0, goal, weights
        self.goal_object_points = None if goal_object_points is None or not len(goal_object_points) else goal_object_points
        self.horizon = horizon_weights(n_steps, last_weight=last_weight)
        self.hand_mask, self.object_mask = segment == SEG_HAND, segment == SEG_OBJECT
        self.has_object = bool(self.object_mask.any())
        self.finger_mask = finger_mask
        self.start_hand, self.start_object = coords0[self.hand_mask], coords0[self.object_mask]
        self.start_object_centroid = self.start_object.mean(axis=0) if self.has_object else None
        self.goal_object_centroid = goal_object_centroid if goal_object_centroid is not None else goal[self.object_mask].mean(axis=0) if self.has_object else None
        self.start_fingers = self.start_hand[finger_mask] if finger_mask is not None else self.start_hand

    def __call__(self, pred_deltas):
        """(pop,) cost of each candidate under the current weights."""
        w = self.weights
        point_weights = None
        if self.finger_mask is not None and w.fingertip != 1.0:
            point_weights = np.where(self.finger_mask, w.fingertip, 1.0).astype(np.float32)
        total = hand_goal_cost_mm(pred_deltas, self.coords0, self.goal, self.hand_mask, self.horizon, point_weights) * w.hand_scale
        extra = []
        if w.object_still > 0.0:
            extra.append(w.object_still * object_still_cost_mm(pred_deltas, self.object_mask, self.horizon))
        if w.object_goal > 0.0 and self.has_object and self.goal_object_points is not None:
            extra.append(w.object_goal * object_chamfer_cost_mm(pred_deltas, self.object_mask, self.horizon, self.start_object, self.goal_object_points))
        elif w.object_goal > 0.0 and self.has_object and self.goal_object_centroid is not None:
            extra.append(w.object_goal * object_goal_cost_mm(pred_deltas, self.object_mask, self.horizon, self.goal_object_centroid, self.start_object_centroid))
        if w.grasp_quality > 0.0:
            extra.append(w.grasp_quality * grasp_quality_penalty(self.start_hand, self.start_object, pred_deltas, self.hand_mask, self.object_mask, self.finger_mask))
        if w.rigid_attachment > 0.0:
            extra.append(w.rigid_attachment * rigid_attachment_cost_mm(self.start_hand, self.start_object, pred_deltas, self.hand_mask, self.object_mask, self.horizon))
        if w.grasp_tight > 0.0:
            extra.append(w.grasp_tight * grasp_tight_cost_mm(pred_deltas, self.hand_mask, self.object_mask, self.horizon, self.start_object_centroid, self.start_fingers, self.finger_mask))
        if w.attachment > 0.0 and self.has_object:
            extra.append(w.attachment * attachment_cost_mm(pred_deltas, self.hand_mask, self.object_mask, self.horizon, self.start_hand.mean(axis=0), self.start_object_centroid))
        if extra:
            summed = extra[0]
            for term in extra[1:]:
                summed = summed + term
            total = total + summed
        return total
