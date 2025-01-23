import glob
from collections import defaultdict
from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import zarr
from scipy.stats import spearmanr

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


def make_combined_plot(plot_data, name, metaname, smooth_window=1, make_legend=False, do_plot_log=True, figsize=None, try_log_transform=False):
    if figsize is not None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = plt.subplots()

    for k, values in plot_data.items():
        if k == 'epoch':
            continue

        smooth_values = smooth_data(values, smooth_window)
        if try_log_transform and np.all(smooth_values > 0):
            smooth_values = np.log(smooth_values)

        if k == 'huber':
            # skip the plot, but keep the legend
            ax1.plot([], [], label=k)
        else:
            ax1.plot(smooth_data(plot_data['epoch'], smooth_window), smooth_values, label=k)

        if do_plot_log and np.all(smooth_values > 0):
            ax2 = ax1.twinx()
            ax2.plot(smooth_data(plot_data['epoch'], smooth_window),
                     np.log(smooth_values), label=f'log {k}', color='r')


    plt.xlabel('Epoch')
    plt.title(name)
    # plt.yscale('log')
    if make_legend:
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


def get_n_inversions(val, sc):
    return np.sum([(val[i] < val[j]) != (sc[i] < sc[j]) for i in range(len(sc)) for j in range(i + 1, len(sc))])


def shake(data, scale=0.05):
    return [x + np.random.normal(0, scale) for x in data]


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
            if len(matching_values) <= smooth_window:
                continue

            prefix = 'smooth ' if smooth_window > 1 else ''
            smooth_matching_values = smooth_data(matching_values, smooth_window)
            smooth_scores = smooth_data(scores, smooth_window)
            skip_frames = 0

            # Compute correlation
            correlation = np.corrcoef(smooth_matching_values[skip_frames:], smooth_scores[skip_frames:])[0, 1]
            correlations[metric][prefix + 'correlation'] = -correlation

            correlations[metric][prefix + 'negative MMRV'] = get_mmrv(-smooth_matching_values[skip_frames:], smooth_scores[skip_frames:])

            correlations[metric][prefix + 'adv over final'] = get_adv_over_final(-smooth_matching_values[skip_frames:], smooth_scores[skip_frames:])

    return correlations


def get_cross_run_correlations(metrics, scores, score_epochs):
    epoch_mappings = {}

    for k in metrics.keys():
        epoch_mappings[k] = {epoch: i for i, epoch in enumerate(metrics[k]['epoch'])}

    sample_run = list(metrics.keys())[0]

    all_correlation_metrics = dict()

    for metric_name in metrics[sample_run]:
        if metric_name == 'epoch':
            continue

        correlations = defaultdict(list)

        for i, epoch in enumerate(score_epochs):
            score_values = []
            metric_values = []

            for run in metrics:
                if epoch not in epoch_mappings[run]:
                    continue

                score_values.append(scores[run][i])
                metric_values.append(metrics[run][metric_name][epoch_mappings[run][epoch]])

            assert len(score_values) == len(metrics)
            if len(score_values) == 0:
                continue

            correlations['correlation'].append(np.corrcoef(-np.array(metric_values), score_values)[0, 1])
            correlations['MMRV'].append(get_mmrv(np.array(metric_values), score_values))
            correlations['negative MMRV'].append(get_mmrv(-np.array(metric_values), score_values))
            correlations['inversions'].append(get_n_inversions(metric_values, score_values))
            correlations['neg inversions'].append(get_n_inversions(-np.array(metric_values), score_values))
            correlations['Spearman'].append(spearmanr(-np.array(metric_values), score_values)[0])

            # noising
            correlations['inversions'][-1] += np.random.uniform(0, 0.2)
            correlations['neg inversions'][-1] += np.random.uniform(0, 0.2)
            correlations['MMRV'][-1] += np.random.uniform(0, 0.02)
            correlations['negative MMRV'][-1] += np.random.uniform(0, 0.02)

        all_correlation_metrics[metric_name] = correlations

    return all_correlation_metrics


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


def get_action_fields(zarr_group, field_prefix):
    action_fields = []

    for field in zarr_group.array_keys():
        action_fields.append(f'{field_prefix}/{field}')

    for field in zarr_group.group_keys():
        action_fields += get_action_fields(zarr_group[field], f'{field_prefix}/{field}')

    return action_fields


def get_state_from_group(zarr_group):
    state = []

    for field in zarr_group.array_keys():
        state.append(zarr_group[field])

    for field in zarr_group.group_keys():
        state += get_state_from_group(zarr_group[field])

    return state


def convert_zip_to_dict(path):
    with zarr.ZipStore(path, mode="r") as store:
        zarr_group = zarr.group(store)

        action_fields = get_action_fields(zarr_group['ground_truth/action'], '')

        val_prediction_results = {
            'multiple_preds': None,
            'gt_action': np.concatenate([zarr_group[f'ground_truth/action{field}'] for field in action_fields], axis=-1),
            'pred_action': np.squeeze(np.concatenate([zarr_group[f'model_generated/action{field}'] for field in action_fields], axis=-1), axis=1),
            'obs': np.concatenate(get_state_from_group(zarr_group['ground_truth/state']), axis=-1)
        }

        val_data = {
            'action': val_prediction_results['gt_action'],
            'obs': val_prediction_results['obs'],
        }

        data_to_save = {
            'val_data': val_data,
            'per_sample_losses': None,
            'val_prediction_results': val_prediction_results,
        }

    return data_to_save


def pi_experiments():
    return ['4l71mq2q', '9ihrtr3m', 'dtt8wm9u', 'kh1vqrxr', 'ldaug7ak', 'pr2xn6r0', 'wx0gvvmm', 'z0hd44iz', 'zblar8fp']


def get_experiment_data(path, epoch):
    if '/pi_datasets/' not in path:
        file_path = os.path.join(path, f'validation_data_epoch_{epoch}.pkl')
        data = load_pkl_file(file_path)
        batch_size = 256

        return data, batch_size

    data = defaultdict(list)

    files = glob.glob(path + f'_{epoch}_*.zarr.zip')
    if len(files) == 0:
        return None, None

    for i in range(len(files)):
        shard = glob.glob(path + f'_{epoch}_*_{i}.zarr.zip')
        assert len(shard) == 1
        assert shard[0] in files

        dict_shard = convert_zip_to_dict(shard[0])
        for k in dict_shard.keys():
            data[k].append(dict_shard[k])

    return data, data['val_prediction_results'][0]['gt_action'].shape[0]


if __name__ == '__main__':
    process_episode_lengths()
    prepare_table({'l1': {'correlation': 0.7548842625246855, 'negative MMRV': 0.02725119001593361}, 'l2': {'correlation': 0.7495254783613362, 'negative MMRV': 0.02725119001593361}, 'l2_sq': {'correlation': 0.7386109458897673, 'negative MMRV': 0.02725119001593361}, 'max': {'correlation': 0.7474944077200927, 'negative MMRV': 0.0273425331779796}, 'geom': {'correlation': 0.8209172811028643, 'negative MMRV': 0.026296711873917147}, 'huber': {'correlation': 0.7495254897259612, 'negative MMRV': 0.02725119001593361}, 'smooth_prob_pi': {'correlation': 0.8623980228656042, 'negative MMRV': 0.04348259453884881}, 'smooth_prob_5': {'correlation': 0.8623977494296868, 'negative MMRV': 0.04348259453884881}})
