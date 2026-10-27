"""
Phase 2: Environment Exploration and Diagnostic Script for LunarLander-v3.

Purpose:
- Inspect observation and action spaces in detail.
- Execute an episode using an untrained, uniform-random policy.
- Log intermediate physical state transitions, engine actions, and step rewards.
- Provide baseline metrics for what an unguided agent achieves.
"""

from collections import Counter
import gymnasium as gym
import numpy as np


# Human-readable labels for state vector indices
STATE_LABELS = [
    "x (pos_x)",
    "y (pos_y)",
    "vx (vel_x)",
    "vy (vel_y)",
    "theta (angle)",
    "omega (ang_vel)",
    "leg1_contact",
    "leg2_contact",
]

ACTION_LABELS = {
    0: "NO-OP (engines off)",
    1: "FIRE_LEFT (push right / clockwise)",
    2: "FIRE_MAIN (upward thrust)",
    3: "FIRE_RIGHT (push left / counter-clockwise)",
}


def format_observation(obs: np.ndarray) -> str:
    """Format the 8-dim state vector into a clean, annotated string."""
    parts = [f"{label}: {val:+.3f}" for label, val in zip(STATE_LABELS, obs)]
    return " | ".join(parts)


def inspect_environment():
    print("=" * 80)
    print("1. ENVIRONMENT SPECIFICATION: LunarLander-v3")
    print("=" * 80)

    env = gym.make("LunarLander-v3")
    obs_space = env.observation_space
    act_space = env.action_space

    print(f"Observation Space: {obs_space}")
    print(f"  - Shape:        {obs_space.shape}")
    print(f"  - Dtype:        {obs_space.dtype}")
    print(f"  - Lower bounds: {np.round(obs_space.low, 2)}")
    print(f"  - Upper bounds: {np.round(obs_space.high, 2)}")

    print(f"\nAction Space:      {act_space}")
    print(f"  - Discrete actions: {act_space.n}")
    for act_idx, act_name in ACTION_LABELS.items():
        print(f"      [{act_idx}] -> {act_name}")

    env.close()


def run_random_episode(seed: int = 42):
    print("\n" + "=" * 80)
    print(f"2. RANDOM POLICY EPISODE EXECUTION (Fixed Seed = {seed})")
    print("=" * 80)

    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=seed)

    print("\nInitial State (Step 0):")
    print(f"  {format_observation(obs)}")
    print(f"  Info: {info}")

    step_idx = 0
    total_reward = 0.0
    action_counts = Counter()
    logged_steps = {1, 20, 50, 80, 100}
    done = False
    first_contact_logged = False

    print("\n--- Stepping Episode ---")
    while not done:
        action = int(env.action_space.sample())
        action_counts[action] += 1

        next_obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1
        total_reward += reward

        # Check for first leg ground contact
        has_leg_contact = bool(next_obs[6] > 0.5 or next_obs[7] > 0.5)
        is_first_contact = has_leg_contact and not first_contact_logged
        if is_first_contact:
            first_contact_logged = True

        # Log selected key steps or terminal step
        if step_idx in logged_steps or is_first_contact or terminated or truncated:
            tag = "FINAL" if (terminated or truncated) else ("CONTACT" if is_first_contact else "FLIGHT")
            print(f"\n[Step {step_idx:03d} | {tag}] Action taken: [{action}] {ACTION_LABELS[action]}")
            print(f"  State:   {format_observation(next_obs)}")
            print(f"  Step reward: {reward:+.3f} | Cumulative return: {total_reward:+.2f}")

        obs = next_obs
        done = terminated or truncated

    env.close()

    print("\n" + "=" * 80)
    print("3. EPISODE SUMMARY & METRICS")
    print("=" * 80)
    print(f"Total steps executed: {step_idx}")
    print(f"Cumulative Return:    {total_reward:.2f}")
    print(f"Terminated (crashed or landed): {terminated}")
    print(f"Truncated (step limit reached): {truncated}")
    print("\nAction Distribution:")
    for act_idx in sorted(ACTION_LABELS.keys()):
        count = action_counts[act_idx]
        pct = (count / step_idx) * 100 if step_idx > 0 else 0
        print(f"  [{act_idx}] {ACTION_LABELS[act_idx]:<42}: {count:3d} ({pct:5.1f}%)")

    # Diagnose outcome
    if terminated:
        if total_reward >= 200.0:
            outcome = "SUCCESSFUL LANDING (Target zone reached with gentle velocity)"
        else:
            outcome = "CRASH / STRUCTURAL FAILURE (Lander collided violently or tilted beyond limits)"
    else:
        outcome = "TIME LIMIT EXCEEDED (Max episode steps reached without resolution)"

    print(f"\nFinal Outcome Assessment: {outcome}")
    print("=" * 80)


if __name__ == "__main__":
    inspect_environment()
    run_random_episode(seed=42)
