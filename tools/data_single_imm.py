# This code generates single-run and Monte Carlo IMM trajectories.
# The tracked state is: x, y, yaw, vx, vy, yaw_rate, w, l.

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())


class DataGenerator:

    def __init__(self, config_file, rng=None):
        self.config_file = config_file
        self.configs = self.load_configs(config_file)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset_from_configs(self.configs)

    def reset_from_configs(self, configs):
        self.configs = copy.deepcopy(configs)
        self.save_name = self.configs["save_name"]
        self.duration = self.configs["data"]["time"]
        self.sample_time = self.configs["data"]["dT"]
        self.init = np.array(self.configs["data"]["init_pos"], dtype=float)

        self.cv = self.configs["cv"]
        self.ct = self.configs["ct"]
        self.ca = self.configs["ca"]

        self.v = float(self.cv["v"])
        self.a = float(self.ca["a"])
        self.w = float(self.ct["w"])
        self.R = np.array(self.configs["data"]["R"], dtype=float)

    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def generate(self):
        data = [self.init.copy()]
        current_speed = None

        num_samples = int(self.duration / self.sample_time)
        for i in range(num_samples):
            curr_time = i * self.sample_time
            x, y, yaw, vx, vy, yaw_rate, width, length = data[i]
            next_state = data[i].copy()

            if self.cv["active"] and self.cv["time_start"] <= curr_time < self.cv["time_end"]:
                if current_speed is None:
                    current_speed = self.v
                else:
                    current_speed = np.hypot(vx, vy)

                next_state[2] = yaw
                next_state[3] = current_speed * np.cos(yaw)
                next_state[4] = current_speed * np.sin(yaw)
                next_state[5] = 0.0
                next_state[0] = x + next_state[3] * self.sample_time
                next_state[1] = y + next_state[4] * self.sample_time

            if self.ca["active"] and self.ca["time_start"] <= curr_time < self.ca["time_end"]:
                next_state[2] = yaw
                next_state[3] = vx + self.a * np.cos(yaw) * self.sample_time
                next_state[4] = vy + self.a * np.sin(yaw) * self.sample_time
                next_state[5] = 0.0
                next_state[0] = x + next_state[3] * self.sample_time
                next_state[1] = y + next_state[4] * self.sample_time

            if self.ct["active"] and self.ct["time_start"] <= curr_time < self.ct["time_end"]:
                turn_speed = self.w * self.ct["radius"]
                next_state[2] = yaw + self.w * self.sample_time
                next_state[3] = turn_speed * np.cos(yaw)
                next_state[4] = turn_speed * np.sin(yaw)
                next_state[5] = self.w
                next_state[0] = x + next_state[3] * self.sample_time
                next_state[1] = y + next_state[4] * self.sample_time

            data.append(next_state)

        return np.array(data, dtype=float)

    def add_noise(self, data):
        noise = self.rng.normal(loc=0.0, scale=np.sqrt(self.R), size=data.shape)
        return data + noise

    def sample_monte_carlo_configs(self, run_idx):
        configs = copy.deepcopy(self.configs)
        init = np.array(configs["data"]["init_pos"], dtype=float)

        init[0] += self.rng.uniform(-5.0, 5.0)
        init[1] += self.rng.uniform(-5.0, 5.0)
        init[2] += self.rng.uniform(-0.35, 0.35)
        init[6] = max(1.5, init[6] + self.rng.uniform(-0.2, 0.2))
        init[7] = max(3.0, init[7] + self.rng.uniform(-0.4, 0.4))
        configs["data"]["init_pos"] = init.tolist()

        configs["cv"]["v"] = max(2.0, self.cv["v"] + self.rng.uniform(-2.0, 2.0))
        configs["ca"]["a"] = self.ca["a"] + self.rng.uniform(-0.5, 0.5)
        configs["ct"]["w"] = self.ct["w"] + self.rng.uniform(-0.35, 0.35)
        configs["ct"]["radius"] = max(2.0, self.ct["radius"] + self.rng.uniform(-1.0, 1.0))

        ct_window_shift = self.rng.integers(-30, 31)
        ca_window_shift = self.rng.integers(-20, 21)
        configs["ct"]["time_start"] = int(np.clip(self.ct["time_start"] + ct_window_shift, 10, self.duration - 30))
        configs["ct"]["time_end"] = int(np.clip(configs["ct"]["time_start"] + (self.ct["time_end"] - self.ct["time_start"]), configs["ct"]["time_start"] + 10, self.duration - 10))
        configs["ca"]["time_start"] = int(np.clip(self.ca["time_start"] + ca_window_shift, configs["ct"]["time_end"], self.duration - 10))
        configs["ca"]["time_end"] = int(np.clip(configs["ca"]["time_start"] + (self.ca["time_end"] - self.ca["time_start"]), configs["ca"]["time_start"] + 5, self.duration))

        noise_scale = self.rng.uniform(0.5, 1.5, size=self.R.shape)
        configs["data"]["R"] = (self.R * noise_scale).tolist()

        configs["save_name"] = os.path.join(
            "data",
            "monte_carlo_simulation_data",
            f"imm_mc_{run_idx:04d}.txt",
        )
        return configs

    def save_single_run(self, save_name=None):
        data = self.generate()
        noise_data = self.add_noise(data)
        output_path = save_name or self.save_name
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, noise_data, delimiter=',')
        np.savetxt(output_path.replace('.txt', '_gt.txt'), data, delimiter=',')
        return output_path

    def save_monte_carlo_runs(self, output_dir, num_runs=100, seed=42):
        self.rng = np.random.default_rng(seed)
        os.makedirs(output_dir, exist_ok=True)
        metadata = []

        for run_idx in range(num_runs):
            sampled_configs = self.sample_monte_carlo_configs(run_idx)
            self.reset_from_configs(sampled_configs)
            output_path = os.path.join(output_dir, f"imm_mc_{run_idx:04d}.txt")
            self.save_single_run(save_name=output_path)
            metadata.append(
                {
                    "file": os.path.basename(output_path),
                    "ground_truth_file": os.path.basename(output_path.replace('.txt', '_gt.txt')),
                    "config": sampled_configs,
                }
            )

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({"seed": seed, "num_runs": num_runs, "runs": metadata}, f, indent=2)

        self.reset_from_configs(self.load_configs(self.config_file))
        return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate IMM training data.")
    parser.add_argument('--config', default='configs/imm.json', help='Path to IMM config file.')
    parser.add_argument(
        '--output-dir',
        default='data/monte_carlo_simulation_data',
        help='Directory used for Monte Carlo trajectory files.',
    )
    parser.add_argument('--num-runs', type=int, default=100, help='Number of Monte Carlo trajectories to create.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible data generation.')
    parser.add_argument(
        '--mode',
        choices=['single', 'monte-carlo'],
        default='monte-carlo',
        help='Generate one file or a Monte Carlo dataset.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataGenerator(config_file=args.config)

    if args.mode == 'single':
        data_generator.save_single_run()
    else:
        data_generator.save_monte_carlo_runs(
            output_dir=args.output_dir,
            num_runs=args.num_runs,
            seed=args.seed,
        )