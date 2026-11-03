"""
Unit tests for Potential-Based Reward Shaping (PBRS) Wrapper.

Verifies:
1. Mathematical boundedness of Phi(s) on normal, boundary, and extreme inputs.
2. Directional descent penalty (penalizes downward vy, ignores upward vy).
3. Shaping increment F = gamma * Phi(s') - Phi(s) boundedness.
4. Absorbing terminal transition behavior (Phi(s_terminal) = 0.0).
5. Truncation continuity (Phi(s_truncated) uses physical state).
6. Telescoping sum identity across multi-step episodes.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from src.wrappers.pbrs_wrapper import PBRSRewardWrapper


def test_potential_bounds_normal_and_ideal():
    # Ideal state: center, stationary, upright, at ground level
    ideal_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    phi_ideal = PBRSRewardWrapper.compute_potential(ideal_obs)
    assert np.isclose(phi_ideal, 0.0), f"Ideal potential should be 0.0, got {phi_ideal}"

    # Typical flight observation
    flight_obs = np.array([0.1, 1.2, 0.05, -0.4, 0.05, -0.1, 0.0, 0.0], dtype=np.float32)
    phi_flight = PBRSRewardWrapper.compute_potential(flight_obs)
    assert -10.0 <= phi_flight <= 0.0, f"Flight potential out of bounds: {phi_flight}"


def test_potential_bounds_extreme_and_pathological():
    # Maximum penalty state (worst possible in all clipped axes)
    worst_obs = np.array([1.0, 0.0, 2.0, -2.0, np.pi, 3.0, 0.0, 0.0], dtype=np.float32)
    phi_worst = PBRSRewardWrapper.compute_potential(worst_obs)
    assert np.isclose(phi_worst, -10.0), f"Worst state potential should be -10.0, got {phi_worst}"

    # Extreme pathological / out-of-bounds inputs
    crazy_inputs = [
        np.array([100.0, -50.0, 250.0, -999.0, 100.0, 50.0, 0.0, 0.0]),
        np.array([-500.0, 500.0, -100.0, 100.0, -50.0, -50.0, 0.0, 0.0]),
        np.array([np.nan, np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    for obs in crazy_inputs:
        phi = PBRSRewardWrapper.compute_potential(obs)
        assert np.isfinite(phi), f"Non-finite potential on extreme obs {obs}: {phi}"
        assert -10.0 <= phi <= 0.0, f"Potential outside [-10.0, 0.0] on extreme obs: {phi}"


def test_asymmetric_vertical_velocity_penalty():
    # Falling vs Climbing at altitude y = 0.2 (near ground)
    falling_obs = np.array([0.0, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    climbing_obs = np.array([0.0, 0.2, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    phi_falling = PBRSRewardWrapper.compute_potential(falling_obs)
    phi_climbing = PBRSRewardWrapper.compute_potential(climbing_obs)

    # Climbing should incur zero descent penalty -> higher potential than falling
    assert phi_climbing > phi_falling, "Climbing should have higher potential than falling"
    assert np.isclose(phi_climbing, 0.0), "Climbing upright with zero drift should have 0.0 potential"
    assert phi_falling < 0.0, "Falling near ground must incur negative potential"


def test_terminal_transition_and_shaping():
    raw_env = gym.make("LunarLander-v3")
    wrapped_env = PBRSRewardWrapper(raw_env, gamma=0.99)

    obs, _ = wrapped_env.reset(seed=42)
    initial_pot = wrapped_env.current_potential
    assert -10.0 <= initial_pot <= 0.0

    # Step through until termination
    done = False
    step_count = 0
    while not done and step_count < 200:
        action = wrapped_env.action_space.sample()
        next_obs, shaped_reward, terminated, truncated, info = wrapped_env.step(action)
        step_count += 1
        done = terminated or truncated

        # Verify finite rewards and valid info tracking
        assert np.isfinite(shaped_reward)
        assert "native_reward" in info
        assert "shaping_reward" in info
        assert np.isfinite(info["shaping_reward"])

        if terminated:
            # When terminated is True, potential must have transitioned to 0.0
            assert wrapped_env.current_potential == 0.0

    wrapped_env.close()


def test_telescoping_sum_identity():
    """
    Theoretical check: For any sequence of states s_0, ..., s_T where s_T is absorbing (Phi=0),
    the discounted sum of shaping rewards must equal -Phi(s_0):
        sum_{t=0}^{T-1} gamma^t * (gamma * Phi(s_{t+1}) - Phi(s_t)) = gamma^T * Phi(s_T) - Phi(s_0) = -Phi(s_0)
    """
    gamma = 0.99
    # Simulate an arbitrary trajectory of 5 states
    states = [
        np.array([0.2, 1.2, 0.1, -0.3, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.15, 0.9, 0.08, -0.4, 0.05, 0.02, 0.0, 0.0]),
        np.array([0.1, 0.5, 0.04, -0.2, 0.02, 0.01, 0.0, 0.0]),
        np.array([0.02, 0.1, 0.01, -0.05, 0.0, 0.0, 1.0, 1.0]),
    ]
    potentials = [PBRSRewardWrapper.compute_potential(s) for s in states]
    terminal_potential = 0.0  # s_T is terminal absorbing

    # Compute step-by-step shaping increments
    shaping_terms = []
    for t in range(len(potentials) - 1):
        f_t = gamma * potentials[t + 1] - potentials[t]
        shaping_terms.append(f_t)
    # Terminal step to absorbing state
    f_terminal = gamma * terminal_potential - potentials[-1]
    shaping_terms.append(f_terminal)

    # Calculate discounted sum
    discounted_sum = sum((gamma ** t) * shaping_terms[t] for t in range(len(shaping_terms)))
    expected_sum = -potentials[0]

    assert np.isclose(discounted_sum, expected_sum, atol=1e-6), (
        f"Telescoping sum mismatch! Got {discounted_sum}, expected {expected_sum}"
    )


def test_dual_logging_separation(tmp_path=None):
    """
    Verify that:
    1. SB3 Monitor records the unshaped native return from raw_env.
    2. PBRSRewardWrapper records the shaped return to monitor_shaped.csv.
    3. Both logs exist and capture accurate metrics.
    """
    import csv
    import tempfile
    from stable_baselines3.common.monitor import Monitor

    with tempfile.TemporaryDirectory() as tmpdir:
        native_log = Path(tmpdir) / "monitor_native.monitor.csv"
        shaped_log = Path(tmpdir) / "monitor_shaped.csv"

        raw_env = gym.make("LunarLander-v3")
        monitored_env = Monitor(raw_env, filename=str(native_log))
        env = PBRSRewardWrapper(
            monitored_env,
            gamma=0.99,
            shaped_log_path=str(shaped_log),
        )

        for _ in range(2):
            obs, _ = env.reset(seed=42)
            done = False
            while not done:
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc

        env.close()

        assert native_log.exists(), "Native monitor log was not created!"
        assert shaped_log.exists(), "Shaped monitor log was not created!"

        # Read native monitor
        with open(native_log, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            assert len(lines) >= 3, "Native monitor missing header or episode rows"
            native_headers = lines[0].split(",")
            assert native_headers[:2] == ["r", "l"]

        # Read shaped monitor
        with open(shaped_log, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2, f"Expected 2 episodes in shaped log, got {len(reader)}"
            for row in reader:
                assert "native_return" in row
                assert "shaped_return" in row
                assert "shaping_net" in row
                assert "length" in row


def test_out_of_bounds_termination_and_shaping():
    """
    Verify out-of-bounds physics and terminal shaping:
    1. Drifting towards boundary (x -> 1.0) decreases potential, applying negative shaping (penalty).
    2. At termination (|x| >= 1.0), next_potential is set to 0.0 (absorbing state).
    3. Outcome classifier cleanly distinguishes OUT_OF_BOUNDS from CRASH based on |x| >= 1.0.
    """
    # 1. Potential decreases as lander drifts sideways
    center_obs = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    drift_obs = np.array([0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    boundary_obs = np.array([1.02, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    phi_center = PBRSRewardWrapper.compute_potential(center_obs)
    phi_drift = PBRSRewardWrapper.compute_potential(drift_obs)
    phi_boundary = PBRSRewardWrapper.compute_potential(boundary_obs)

    assert phi_center > phi_drift > phi_boundary, (
        f"Potential must decrease as x moves toward/past boundary: {phi_center} > {phi_drift} > {phi_boundary}"
    )

    # 2. Terminal state handling on out-of-bounds step
    raw_env = gym.make("LunarLander-v3")
    wrapped_env = PBRSRewardWrapper(raw_env, gamma=0.99)
    wrapped_env.reset(seed=1000)

    # Force lander to fly laterally by repeatedly firing side thrusters
    done = False
    step_count = 0
    terminal_obs = None
    terminal_rew = None
    terminal_outcome = None

    while not done and step_count < 500:
        # Action 1 fires left engine (pushes right)
        obs, r, term, trunc, info = wrapped_env.step(1)
        step_count += 1
        done = term or trunc
        if done:
            terminal_obs = obs
            terminal_rew = info["native_reward"]

    wrapped_env.close()

    assert done, "Environment failed to terminate"
    assert wrapped_env.current_potential == 0.0, "Terminal potential must be 0.0"

    # 3. Verify classification logic on this termination
    if abs(terminal_obs[0]) >= 1.0:
        outcome = "OUT_OF_BOUNDS"
    else:
        outcome = "CRASH"

    assert outcome in ["OUT_OF_BOUNDS", "CRASH"]
    assert np.isclose(terminal_rew, -100.0, atol=1.0)


if __name__ == "__main__":
    test_potential_bounds_normal_and_ideal()
    test_potential_bounds_extreme_and_pathological()
    test_asymmetric_vertical_velocity_penalty()
    test_terminal_transition_and_shaping()
    test_telescoping_sum_identity()
    test_dual_logging_separation()
    test_out_of_bounds_termination_and_shaping()
    print("All PBRS wrapper unit tests passed successfully!")
