import os
import sys
sys.path.append(os.getcwd())

import json
import numpy as np
import matplotlib.pyplot as plt
from core.imm_model_based import ModelBasedIMM


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


def load_observations(data_file):
    data = np.loadtxt(data_file, delimiter=',')
    return data[:, [0, 1, 6, 7]]


def load_ground_truth(gt_file):
    return np.loadtxt(gt_file, delimiter=',')


def build_motion_profile(config, length):
    dT = config['data']['dT']
    times = np.arange(length) * dT
    phases = np.full(length, 'CV', dtype='<U10')

    ca_start = config['ca']['time_start']
    ca_end = config['ca']['time_end']
    ct_start = config['ct']['time_start']
    ct_end = config['ct']['time_end']

    for i, t in enumerate(times):
        if ct_start <= t < ct_end:
            phases[i] = 'CT'
        elif ca_start <= t < ca_end:
            phases[i] = 'CA'
        else:
            phases[i] = 'CV'

    return times, phases


def run_experiment(config_file, obs_file, gt_file, model_path=None):
    config = load_config(config_file)
    observations = load_observations(obs_file)
    ground_truth = load_ground_truth(gt_file)
    num_steps = observations.shape[0]

    times, motion_phases = build_motion_profile(config, num_steps)

    imm = ModelBasedIMM(config_file, model_path=model_path, device='cpu')

    estimates = np.zeros((num_steps, imm.state_dim))
    probs = np.zeros((num_steps, imm.model_cnt))

    for i in range(num_steps):
        z = observations[i]
        imm.update(z)
        estimates[i] = imm.get_estimate()
        probs[i] = imm.get_model_prob().copy()

    errors = estimates[:, :2] - ground_truth[:, :2]
    rmse_x = np.sqrt(np.mean(errors[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(errors[:, 1] ** 2))
    rmse_total = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))

    print('Experiment results:')
    print(f'  Samples: {num_steps}')
    print(f'  RMSE x: {rmse_x:.6f}')
    print(f'  RMSE y: {rmse_y:.6f}')
    print(f'  RMSE position: {rmse_total:.6f}')

    return times, motion_phases, probs, errors, rmse_x, rmse_y, rmse_total


def plot_model_probabilities(times, motion_phases, probs, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, probs[:, 0], label='CA', color='tab:red')
    ax.plot(times, probs[:, 1], label='CV', color='tab:blue')
    ax.plot(times, probs[:, 2], label='CT', color='tab:green')

    unique_phases = ['CV', 'CT', 'CA']
    colors = {'CV': '#e0f3ff', 'CT': '#e0ffe0', 'CA': '#ffe0e0'}

    start = 0
    current = motion_phases[0]
    for i, phase in enumerate(motion_phases):
        if phase != current or i == len(motion_phases) - 1:
            end = i if phase != current else i + 1
            ax.axvspan(times[start], times[end - 1], color=colors[current], alpha=0.15)
            start = i
            current = phase
    # handle final segment
    ax.axvspan(times[start], times[-1], color=colors[current], alpha=0.15)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Model probability')
    ax.set_title('Figure 1: Model probability vs motion profile')
    ax.legend(loc='upper right')
    ax.grid(True)

    phase_legend = [plt.Line2D([0], [0], color='k', lw=0, marker='s', markersize=8,
                               markerfacecolor=colors[p], alpha=0.3, label=p)
                    for p in unique_phases]
    ax.legend(handles=ax.lines + phase_legend, loc='upper right')

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        print(f'Saved model probability figure to {save_path}')
    plt.show()


def plot_position_errors(times, errors, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, errors[:, 0], label='Error x', color='tab:blue')
    ax.plot(times, errors[:, 1], label='Error y', color='tab:red')
    ax.axhline(0, color='k', lw=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position error [m]')
    ax.set_title('Position error in x and y over time')
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        print(f'Saved position error figure to {save_path}')
    plt.show()


def main():
    config_file = 'configs/imm.json'
    obs_file = 'data/imm_single.txt'
    gt_file = 'data/imm_single_gt.txt'
    model_path = 'models/imm_gru_model.pth'

    times, motion_phases, probs, errors, rmse_x, rmse_y, rmse_total = run_experiment(
        config_file, obs_file, gt_file, model_path=model_path)

    figure_dir = 'figures'
    os.makedirs(figure_dir, exist_ok=True)
    plot_model_probabilities(times, motion_phases, probs,
                             save_path=os.path.join(figure_dir, 'figure1_model_probabilities.png'))
    plot_position_errors(times, errors,
                         save_path=os.path.join(figure_dir, 'figure2_position_errors.png'))

    print('\nDone. Figures saved in the figures/ directory.')


if __name__ == '__main__':
    main()
