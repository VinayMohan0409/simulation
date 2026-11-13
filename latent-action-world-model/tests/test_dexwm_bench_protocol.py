from lawm.sim.dexwm.bench.protocol import BUDGET, COUNT_FROM, SUSTAIN, episode_result


def test_success_before_last_outer_step_does_not_count():
    checks = [True] * COUNT_FROM + [False] * (BUDGET - COUNT_FROM)
    assert episode_result(checks) == (False, BUDGET)


def test_streak_in_last_outer_step_ends_early():
    checks = [False] * (COUNT_FROM + 5) + [True] * SUSTAIN + [False] * 100
    assert episode_result(checks) == (True, COUNT_FROM + 5 + SUSTAIN)


def test_broken_streak_resets():
    checks = [False] * COUNT_FROM + ([True] * (SUSTAIN - 1) + [False]) * 10
    assert episode_result(checks) == (False, BUDGET)


def test_last_step_success_counts_without_streak():
    checks = [False] * (BUDGET - 1) + [True]
    assert episode_result(checks) == (True, BUDGET)


def test_streak_spanning_into_counting_window_starts_at_window():
    checks = [True] * (COUNT_FROM + SUSTAIN - 1) + [False] * (BUDGET - COUNT_FROM - SUSTAIN + 1)
    assert episode_result(checks) == (False, BUDGET)
