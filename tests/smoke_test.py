"""
Smoke test to verify:
1. All critical RL dependencies import cleanly.
2. LunarLander-v3 environment initializes and steps through a full episode.
3. Clean environment closure.
"""

import sys


def test_imports():
    print("[1/3] Verifying core imports...")
    import gymnasium as gym
    import Box2D
    import torch
    import stable_baselines3 as sb3
    import numpy as np
    import matplotlib

    print(f"  - Python:            {sys.version.split()[0]}")
    print(f"  - Gymnasium:         {gym.__version__}")
    print(f"  - Box2D:             {Box2D.__version__}")
    print(f"  - PyTorch:           {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"  - Stable-Baselines3: {sb3.__version__}")
    print(f"  - NumPy:             {np.__version__}")
    print(f"  - Matplotlib:        {matplotlib.__version__}")
    print("  -> All imports verified successfully!\n")


def test_lunar_lander_episode():
    print("[2/3] Initializing Gymnasium LunarLander-v3 environment...")
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=42)

    print(f"  - Observation shape: {obs.shape} (dtype: {obs.dtype})")
    print(f"  - Action space:      {env.action_space}")
    print(f"  - Initial obs snippet: {obs[:4]}...")

    print("\n[3/3] Stepping through a random-action episode...")
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()

    print(f"  - Episode finished after {steps} steps.")
    print(f"  - Total cumulative reward: {total_reward:.2f}")
    print(f"  - Terminated: {terminated}, Truncated: {truncated}")
    print("  - Environment closed cleanly.")
    print("\n>>> SMOKE TEST PASSED: LunarLander-v3 physics simulation is fully operational! <<<\n")


if __name__ == "__main__":
    try:
        test_imports()
        test_lunar_lander_episode()
    except Exception as e:
        print(f"\n[ERROR] Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)
