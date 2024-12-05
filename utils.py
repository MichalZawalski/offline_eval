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


def make_single_plots(plot_data, name, metaname):
    for k, values in plot_data.items():
        if k == 'epoch':
            continue
        make_combined_plot({k: values, 'epoch': plot_data['epoch']}, f'{name}: {k}', metaname)


def make_combined_plot(plot_data, name, metaname):
    plt.figure()
    for k, values in plot_data.items():
        if k == 'epoch':
            continue
        plt.plot(plot_data['epoch'], values, label=k)
    plt.xlabel('Epoch')
    plt.title(name)
    # plt.yscale('log')
    plt.legend()
    plt.text(0.01, -0.08, metaname, fontsize=8, ha='left', va='center', transform=plt.gca().transAxes)
    plt.savefig(f'figures/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{name.lower().replace(" ", "_")}.png')
    plt.close()


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
        matching_values = np.array(values)[plot_indices]

        # Compute correlation
        correlation = np.corrcoef(matching_values, scores)[0, 1]
        correlations[metric]['correlation'] = correlation

        def get_mmrv(val, sc):
            def rank_violation(i, j):
                if (val[i] < val[j]) != (sc[i] < sc[j]):
                    return np.abs(sc[i] - sc[j])
                else:
                    return 0

            return np.mean([np.max([rank_violation(i, j) for j in range(len(sc)) if j != i]) for i in range(len(sc))])

        correlations[metric]['MMRV'] = get_mmrv(matching_values, scores)
        correlations[metric]['negative MMRV'] = get_mmrv(-matching_values, scores)

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
