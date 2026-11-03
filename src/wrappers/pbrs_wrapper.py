import csv
from pathlib import Path
import gymnasium as gym
import numpy as np


class PBRSRewardWrapper(gym.Wrapper):
    """
    Potential-Based Reward Shaping (PBRS) Wrapper for LunarLander-v3.

    Augments native step reward r with:
        F(s, s') = gamma * Phi(s') - Phi(s)

    Properties:
    1. Strictly bounded potential: Phi(s) in [-10.0, 0.0] for all s in R^8.
    2. Guaranteed finite shaping increments F.
    3. Proper terminal state handling: Phi(s_terminal) = 0.0 when terminated is True.
    4. Truncation continuity: evaluates physical Phi(s) when truncated is True (no artificial boundary bonus).
    5. Tracks uncorrupted native episode return alongside shaped return.
    6. Optional dedicated shaped return logger to separate disk file.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        w_descent: float = 4.0,
        w_tilt: float = 3.0,
        w_spin: float = 1.0,
        w_center: float = 2.0,
        shaped_log_path: str = None,
    ):
        super().__init__(env)
        self.gamma = float(gamma)
        self.w_descent = float(w_descent)
        self.w_tilt = float(w_tilt)
        self.w_spin = float(w_spin)
        self.w_center = float(w_center)
        self.shaped_log_path = shaped_log_path

        self.current_potential = 0.0
        self.current_obs = None
        self.cumulative_native_return = 0.0
        self.cumulative_shaped_return = 0.0
        self.episode_steps = 0
        self.episode_count = 0

        # Initialize shaped return logger if path provided
        self._csv_file = None
        self._csv_writer = None
        if self.shaped_log_path:
            log_p = Path(self.shaped_log_path)
            log_p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(log_p, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=["episode", "native_return", "shaped_return", "shaping_net", "length"],
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()

    @staticmethod
    def compute_potential(
        obs: np.ndarray,
        w_descent: float = 4.0,
        w_tilt: float = 3.0,
        w_spin: float = 1.0,
        w_center: float = 2.0,
    ) -> float:
        """
        Compute bounded potential Phi(s) for a state vector s.

        State s = [x, y, vx, vy, theta, omega, leg1, leg2]
        Guaranteed: Phi(s) in [- (w_descent + w_tilt + w_spin + w_center), 0.0]
        """
        if obs is None:
            return 0.0

        # Replace non-finite values if encountered defensively
        clean_obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        x = float(clean_obs[0])
        y = float(clean_obs[1])
        vx = float(clean_obs[2])
        vy = float(clean_obs[3])
        theta = float(clean_obs[4])
        omega = float(clean_obs[5])

        # 1. Altitude & descent: k_prox in [0.0, 1.0], v_down_norm in [0.0, 1.0]
        y_norm = float(np.clip(y, 0.0, 1.5)) / 1.5
        k_prox = 1.0 - y_norm
        v_down = max(0.0, -vy)
        v_down_norm = float(np.clip(v_down, 0.0, 2.0)) / 2.0
        penalty_descent = k_prox * v_down_norm

        # 2. Tilt angle: penalty in [0.0, 1.0]
        penalty_tilt = float(np.clip(abs(theta), 0.0, np.pi)) / np.pi

        # 3. Spin rate: penalty in [0.0, 1.0]
        penalty_spin = float(np.clip(abs(omega), 0.0, 3.0)) / 3.0

        # 4. Horizontal alignment and drift: penalty in [0.0, 1.0]
        x_norm = float(np.clip(abs(x), 0.0, 1.0))
        vx_norm = float(np.clip(abs(vx), 0.0, 2.0)) / 2.0
        penalty_center = 0.5 * x_norm + 0.5 * vx_norm

        phi = -(
            w_descent * penalty_descent
            + w_tilt * penalty_tilt
            + w_spin * penalty_spin
            + w_center * penalty_center
        )

        return float(phi)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        self.current_potential = self.compute_potential(
            obs, self.w_descent, self.w_tilt, self.w_spin, self.w_center
        )
        self.cumulative_native_return = 0.0
        self.cumulative_shaped_return = 0.0
        self.episode_steps = 0
        return obs, info

    def step(self, action):
        next_obs, native_reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.cumulative_native_return += native_reward

        # Compute next state potential
        if terminated:
            # Physical absorbing terminal state (safe rest or crash): Phi(s_terminal) = 0.0
            next_potential = 0.0
        else:
            # Continuing state or clock truncation: compute physical potential
            next_potential = self.compute_potential(
                next_obs, self.w_descent, self.w_tilt, self.w_spin, self.w_center
            )

        # Potential-Based Reward Shaping formula: F = gamma * Phi(s') - Phi(s)
        shaping = (self.gamma * next_potential) - self.current_potential
        shaped_reward = native_reward + shaping

        self.cumulative_shaped_return += shaped_reward

        # Update tracking
        self.current_potential = next_potential
        self.current_obs = next_obs

        # Record metrics in info dict for monitoring
        info["native_reward"] = native_reward
        info["shaping_reward"] = shaping
        info["native_cumulative_return"] = self.cumulative_native_return
        info["shaped_cumulative_return"] = self.cumulative_shaped_return

        # If episode finished and logger configured, write row
        if (terminated or truncated) and self._csv_writer is not None:
            self.episode_count += 1
            self._csv_writer.writerow({
                "episode": self.episode_count,
                "native_return": round(self.cumulative_native_return, 4),
                "shaped_return": round(self.cumulative_shaped_return, 4),
                "shaping_net": round(self.cumulative_shaped_return - self.cumulative_native_return, 4),
                "length": self.episode_steps,
            })
            self._csv_file.flush()

        return next_obs, shaped_reward, terminated, truncated, info

    def close(self):
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        return super().close()

