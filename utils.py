from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from episodes_length import EPISODES_LENGTH_DATA


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')


def smooth_data(data, window_size=5):
    return np.array([np.mean(data[window_size * i:window_size * (i + 1)]) for i in range(len(data) // window_size)])


def make_single_plots(plot_data, name, metaname, smooth_window=1):
    for k, values in plot_data.items():
        if k == 'epoch':
            continue
        make_combined_plot({k: values, 'epoch': plot_data['epoch']},
                           f'{name}: {k}', metaname, smooth_window)


def make_combined_plot(plot_data, name, metaname, smooth_window=1):
    plt.figure()
    for k, values in plot_data.items():
        if k == 'epoch':
            continue
        plt.plot(smooth_data(plot_data['epoch'], smooth_window),
                 smooth_data(values, smooth_window), label=k)
    plt.xlabel('Epoch')
    plt.title(name)
    # plt.yscale('log')
    plt.legend()
    plt.text(0.01, -0.08, metaname, fontsize=8, ha='left', va='center', transform=plt.gca().transAxes)
    plt.savefig(f'figures/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{name.lower().replace(" ", "_")}.png')
    plt.close()


def get_mmrv(val, sc):
    def rank_violation(i, j):
        if (val[i] < val[j]) != (sc[i] < sc[j]):
            return np.abs(sc[i] - sc[j])
        else:
            return 0

    return np.mean([np.max([rank_violation(i, j) for j in range(len(sc)) if j != i]) for i in range(len(sc))])


def get_adv_over_final(val, sc):
    def get_smooth_score(idx):
        return np.mean([np.mean(sc[max(0, idx - r):idx + r + 1]) for r in [0, 1, 2, 3]])
    return get_smooth_score(np.argmax(val)) - get_smooth_score(len(sc) - 1)


def get_matching_values(values, indices):
    # Average the values between consecutive indices
    matching_values = []

    full_indices = np.concatenate([[0], indices + 1])

    for i in range(len(full_indices) - 1):
        matching_values.append(np.mean(values[full_indices[i]:full_indices[i + 1]]))

    assert len(matching_values) == len(indices)

    return np.array(matching_values)


def get_correlation_metrics(plots, scores, score_epochs):
    correlations = {}
    plot_epochs = plots['epoch']
    scores = np.array(scores)

    epoch_to_index = {epoch: i for i, epoch in enumerate(plot_epochs)}
    plot_indices = np.array([epoch_to_index[epoch] for epoch in score_epochs])

    for metric, values in plots.items():
        if metric == 'epoch':  # Skip the 'epochs' key
            continue

        correlations[metric] = dict()

        # Extract values for matching timesteps
        matching_values = get_matching_values(np.array(values), plot_indices)

        for smooth_window in [1, 5]:
            prefix = 'smooth ' if smooth_window > 1 else ''
            smooth_matching_values = smooth_data(matching_values, smooth_window)
            smooth_scores = smooth_data(scores, smooth_window)

            # Compute correlation
            correlation = np.corrcoef(smooth_matching_values, smooth_scores)[0, 1]
            correlations[metric][prefix + 'correlation'] = -correlation

            correlations[metric][prefix + 'negative MMRV'] = get_mmrv(-smooth_matching_values, smooth_scores)

            correlations[metric][prefix + 'adv over final'] = get_adv_over_final(-smooth_matching_values, smooth_scores)

    return correlations


def find_closest_states(val_data, n_closest=10):
    states = []

    for index in range(int(1e5)):
        batch_number = index // 256
        index_in_batch = index % 256

        if batch_number >= len(val_data) or index_in_batch >= len(val_data[batch_number]['action']):
            break

        state = val_data[batch_number]['obs'][index_in_batch]
        states.append(state.flatten())

    states = np.array(states)

    # compute pairwise distances between states
    distances = np.linalg.norm(states[:, None] - states[None], axis=-1, ord=2)

    # for each state, find n_closest states
    closest_states = []
    for i in range(len(states)):
        closest_states.append(np.argsort(distances[i])[:n_closest])

    return np.array(closest_states)


def process_episode_lengths():
    for taskname, data in EPISODES_LENGTH_DATA.items():
        is_end = []
        for i in range(len(data['idxs'])):
            if data['val_mask'][data['idxs'][i]]:
                if i == len(data['idxs']) - 1 or data['idxs'][i] != data['idxs'][i + 1]:
                    is_end = is_end[:-data['offset']]
                    is_end.append(1)
                else:
                    is_end.append(0)
        data['is_end'] = np.array(is_end)


def simulate_rollout(closest_states, prediction_results, taskname, max_length=1000):
    state = np.random.randint(0, len(closest_states))
    visited_states = [state]
    success = False

    for _ in range(max_length):
        losses = []

        for supply_index in closest_states[state]:
            gt_action = prediction_results[supply_index // 256]['gt_action'][supply_index % 256]
            pred_action = prediction_results[supply_index // 256]['pred_action'][supply_index % 256]
            losses.append(np.linalg.norm(gt_action - pred_action, ord=2))

        trans = closest_states[state][np.argmin(losses)]

        if EPISODES_LENGTH_DATA[taskname]['is_end'][trans]:
            success = True
            break
        else:
            state = trans + 1
            visited_states.append(state)

    return success


def prepare_table(data, n_datasets=4):
    for metric, values in data.items():
        print(f"{metric}\t\t{values['correlation']}\t{values['negative MMRV']}")
        for _ in range(n_datasets - 1):
            print()


if __name__ == '__main__':
    process_episode_lengths()
    prepare_table({'l1': {'correlation': 0.7548842625246855, 'negative MMRV': 0.02725119001593361}, 'l2': {'correlation': 0.7495254783613362, 'negative MMRV': 0.02725119001593361}, 'l2_sq': {'correlation': 0.7386109458897673, 'negative MMRV': 0.02725119001593361}, 'max': {'correlation': 0.7474944077200927, 'negative MMRV': 0.0273425331779796}, 'geom': {'correlation': 0.8209172811028643, 'negative MMRV': 0.026296711873917147}, 'huber': {'correlation': 0.7495254897259612, 'negative MMRV': 0.02725119001593361}, 'smooth_prob_pi': {'correlation': 0.8623980228656042, 'negative MMRV': 0.04348259453884881}, 'smooth_prob_5': {'correlation': 0.8623977494296868, 'negative MMRV': 0.04348259453884881}})
