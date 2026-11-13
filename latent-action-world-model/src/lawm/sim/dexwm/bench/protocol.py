"""DexWM's episode budget and success counting (work/dexwm/sim_eval/test.py), reproduced exactly.

test_trajectory runs PRED_STEPS outer steps of SUB_WAYPOINTS control steps each (env.step at control_freq 10). Success is only counted in the last outer step
(waypoint_control, `idx == pred_steps - 1`): a streak of SUSTAIN consecutive successful steps ends the episode as a success; a failed step resets the streak.
If no streak completes, the episode's result is the success check of the very last control step (test_trajectory returns waypoint_control's last `success`).
Before planning, reach and grasp (not place) take one zero-action step that is not counted (sim_eval/test.py:127-134)."""
from __future__ import annotations

PRED_STEPS = 3
SUB_WAYPOINTS = 100
SUSTAIN = 10
BUDGET = PRED_STEPS * SUB_WAYPOINTS  # 300 control steps
COUNT_FROM = (PRED_STEPS - 1) * SUB_WAYPOINTS  # step index (0-based) where counting starts
ZERO_ACTION_TASKS = ("reach", "grasp")


class DexWMCounter:
    """Feed the success check of every control step in order; .done says the episode ends now; .success is the episode's result once done."""

    def __init__(self):
        self.t, self.streak, self.last, self.done, self.success = 0, 0, False, False, False

    def update(self, ok):
        assert not self.done, "the episode is over"
        ok = bool(ok)
        if self.t >= COUNT_FROM:
            self.streak = self.streak + 1 if ok else 0
            if self.streak >= SUSTAIN:
                self.done, self.success = True, True
        self.last = ok
        self.t += 1
        if not self.done and self.t >= BUDGET:
            self.done, self.success = True, self.last
        return self.done


def episode_result(checks):
    """Success of an episode from its per-step success checks (len <= BUDGET, stops where DexWM's loop would stop)."""
    c = DexWMCounter()
    for ok in checks:
        if c.update(ok):
            break
    assert c.done, f"only {c.t} of {BUDGET} steps"
    return c.success, c.t
