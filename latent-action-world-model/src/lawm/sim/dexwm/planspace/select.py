"""Pick the episodes of a task (reach, grasp or place): from the 50 DexWM evaluation demos (minus the tuning demos), keep those whose success check fails at the start state and during a hold-still rollout; the first n in manifest order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planspace.common import TUNING_DEMOS, eval_demos, eval_dir, hold_action, write_json
from lawm.sim.dexwm.runner import load_config


def check_demo(demo, env_settings, hold_steps, task="reach"):
    """(start distance m, success at start, success during hold, min distance during hold) of a demo's task; start success is DexWM's check of the task."""
    sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), demo), env_settings)
    try:
        sim.reset(task)
        d0 = sim.distance()
        start_ok = bool(getattr(sim.wrapped, f"check_{task}_success")())
        obs = sim.observe()
        action = hold_action(obs)
        hold_ok, dmin = False, d0
        for _ in range(hold_steps):
            hold_ok |= sim.step(action)
            dmin = min(dmin, sim.distance())
        return d0, start_ok, hold_ok, dmin
    finally:
        sim.env.close()


def _check(args):
    """Pool worker."""
    demo, hold_steps, task = args
    env_settings = load_config(Path(__file__).resolve().parents[5] / "configs/stage3/dexwm.json").env
    return demo, check_demo(demo, env_settings, hold_steps, task)


def main():
    """python -m lawm.sim.dexwm.planspace.select --out episodes.json [--n 10] [--hold-steps 100] [--workers 4]: checks demos in manifest order, a batch of `workers` at a time, until n are eligible; resumable."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--hold-steps", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--task", default="reach", choices=["reach", "grasp", "place"])
    a = p.parse_args()
    import multiprocessing as mp
    out = Path(a.out)
    state = json.loads(out.read_text()) if out.exists() else dict(task=a.task, checked={})
    assert state.get("task", "reach") == a.task, f"{out} holds {state.get('task', 'reach')} episodes, not {a.task}"
    order = [d for d in eval_demos() if d not in TUNING_DEMOS]

    def selected():
        """First n eligible demos in manifest order, counting only an unbroken checked prefix."""
        sel = []
        for d in order:
            if d not in state["checked"]:
                break
            if state["checked"][d]["eligible"]:
                sel.append(d)
            if len(sel) == a.n:
                break
        return sel

    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        while len(selected()) < a.n:
            todo = [d for d in order if d not in state["checked"]]
            if not todo:
                break
            need = a.n - len(selected())
            batch = todo[:max(a.workers, min(len(todo), need + 2))][:max(a.workers, need + 2)]
            for demo, (d0, s_ok, h_ok, dmin) in pool.imap(_check, [(d, a.hold_steps, a.task) for d in batch]):
                eligible = (not s_ok) and (not h_ok)
                state["checked"][demo] = dict(start_distance_m=d0, start_success=bool(s_ok), hold_success=bool(h_ok), hold_min_distance_m=dmin, eligible=eligible)
                write_json(out, state)
                print(f"{demo}: start {d0:.3f} m, start ok {s_ok}, hold ok {h_ok} -> {'eligible' if eligible else 'skip'}", flush=True)
    state["selected"] = selected()
    state["done"] = len(state["selected"]) == a.n
    write_json(out, state)
    print("selected", state["selected"], flush=True)


if __name__ == "__main__":
    main()
