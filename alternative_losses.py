import math
from collections import defaultdict
from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from trajectories_metadata import get_val_trajectories_metadata
from utils import load_pkl_file, make_single_plots, get_correlation_metrics, convert_zip_to_dict, get_experiment_data, \
    make_combined_plot


def huber_loss(x, delta=1):
    if np.abs(x) < delta:
        return 0.5 * x ** 2
    else:
        return delta * (np.abs(x) - 0.5 * delta)


def smooth_probability_loss(x, scale=math.pi):
    return 1 - 1 / ((scale * x)**2 + 1)


def get_alternative_losses(output_dir, start_epoch, end_epoch, step_size=1,
                           do_plot=True, order_per_datapoint=False, plot_samples=False, plot_cum_minimals=False,
                           use_smoothing=True, trajectory_aggregations=None):
    metaname = output_dir.split('/')[-1]
    res = defaultdict(list)
    scores = []
    score_epochs = []
    per_datapoint = []
    pi_scores = None

    val_lengths = get_val_trajectories_metadata(metaname)

    for epoch in range(start_epoch, end_epoch + 1, step_size):
        if epoch % 50 == 0:
            print(f"Processing epoch {epoch}")

        data, batch_size = get_experiment_data(output_dir, epoch)
        if data is None:
            continue

        val_data = data['val_data']
        prediction_results = data['val_prediction_results']

        losses = defaultdict(list)

        for index in range(int(1e5)):
            batch_number = index // batch_size
            index_in_batch = index % batch_size

            if batch_number >= len(val_data) or index_in_batch >= len(val_data[batch_number]['action']):
                break

            gt_action = prediction_results[batch_number]['gt_action'][index_in_batch]
            pred_action = prediction_results[batch_number]['pred_action'][index_in_batch]

            losses['l1'].append(np.linalg.norm(gt_action - pred_action, ord=1))
            l2_distance = np.linalg.norm(gt_action - pred_action, ord=2)
            losses['l2'].append(l2_distance)
            losses['l2_sq'].append(l2_distance ** 2)
            losses['max'].append(np.max(np.abs(gt_action - pred_action)))
            losses['geom'].append(np.mean(np.log(np.abs(gt_action - pred_action) + 1e-6)))
            # losses['huber'].append(huber_loss(l2_distance))
            losses['huber'].append(np.nan)  # Indistinguishable from l2_sq
            losses['smooth_prob_pi'].append(smooth_probability_loss(l2_distance, scale=math.pi))
            losses['smooth_prob_5'].append(smooth_probability_loss(l2_distance, scale=5))

            # losses['per_sample_loss'].append(np.sum(np.abs(data['per_sample_losses'][batch_number][index_in_batch])))
            # losses['torch_mse'].append(torch.nn.functional.mse_loss(torch.tensor(pred_action), torch.tensor(gt_action)).item())

            if index >= len(per_datapoint):
                per_datapoint.append(defaultdict(list))

            for k, v in losses.items():
                per_datapoint[index][k].append(v[-1])

        aggregations = {k: {'min': [], 'max': []} for k in losses.keys()}

        if trajectory_aggregations is not None:
            index = 0
            for l in val_lengths:
                l -= 7  # action horizon offset
                for p_id in range(trajectory_aggregations):
                    part = defaultdict(list)
                    for i in range(l*p_id // trajectory_aggregations, l*(p_id+1) // trajectory_aggregations):
                        for k, v in per_datapoint[index].items():
                            part[k].append(v[-1])
                        index += 1

                    for k, v in part.items():
                        if len(part[k]) == 0:
                            print("Empty part for aggregation, skipping")
                        else:
                            aggregations[k]['min'].append(np.min(part[k]))
                            aggregations[k]['max'].append(np.max(part[k]))

        res['epoch'].append(epoch)
        for k, v in losses.items():
            res[k].append(np.mean(v))

        if trajectory_aggregations is not None:
            for k in aggregations.keys():
                for op in ['min', 'max']:
                    res[f'({op}-{trajectory_aggregations}) {k}'].append(np.mean(aggregations[k][op]))

        if 'test/mean_score' in data:
            scores.append(data['test/mean_score'])
            score_epochs.append(epoch)
        elif '/pi_datasets/' in output_dir:
            if pi_scores is None:
                import csv

                pi_scores = dict()

                with open('pi_results.csv', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                    for row in reader:
                        pi_scores[row[0]] = float(row[5].split(' ')[0]) / float(row[5].split(' ')[-1])

            for k, v in pi_scores.items():
                if f'{metaname}_{epoch}_' in k:
                    scores.append(v)
                    score_epochs.append(epoch)

    if do_plot:
        smooth_window = 5 if use_smoothing else 1
        make_single_plots(res, 'Alternative losses', metaname, smooth_window)
        if scores:
            make_single_plots({'mean_score': scores, 'epoch': score_epochs}, 'Mean score', metaname, smooth_window)

        print(get_correlation_metrics(res, scores, score_epochs))

    if order_per_datapoint:
        for ordering_metric in ['l1', 'l2', 'l2_sq', 'max', 'geom', 'smooth_prob_pi', 'smooth_prob_5']:
            order = []
            for i in range(len(per_datapoint)):
                # get the correlation of l2 loss with scores
                correlation = np.corrcoef(-np.array(per_datapoint[i][ordering_metric]), scores)[0, 1]
                # correlation = spearmanr(-np.array(per_datapoint[i][ordering_metric]), scores)[0]
                # correlation = np.mean(per_datapoint[i][ordering_metric][-10:])
                order.append((correlation, i))

            order = [i for _, i in sorted(order)]

            plots = dict()
            plots['epoch'] = list(range(len(per_datapoint)))

            for metric in per_datapoint[0]:
                plots[metric] = [np.corrcoef(-np.array(per_datapoint[i][metric]), scores)[0, 1] for i in order]
                # plots[metric] = [spearmanr(-np.array(per_datapoint[i][metric]), scores)[0] for i in order]
                # plots[metric] = [np.mean(per_datapoint[i][metric][-10:]) for i in order]

            make_combined_plot(plots, f'Sorted by {ordering_metric}', f'{metaname}, ordered by last values',
                               make_legend=True, do_plot_log=False, smooth_window=15 if use_smoothing else 1, figsize=(16, 12))

    if plot_cum_minimals:
        for metric in ['l1', 'l2', 'l2_sq', 'max', 'geom', 'smooth_prob_pi', 'smooth_prob_5']:
            order = []
            for i in range(len(per_datapoint)):
                final_value = np.mean(per_datapoint[i][metric][-10:])
                order.append((final_value, i))

            order = [i for _, i in sorted(order)]

            plots = dict()
            plots['epoch'] = list(range(len(per_datapoint)))
            plot_name = f'{metric} cumsum correlation'
            plots[plot_name] = []
            cumsum = np.zeros_like(per_datapoint[0][metric])

            for i, data in enumerate(per_datapoint):
                cumsum += np.array(data[metric])
                plots[plot_name].append(np.corrcoef(-cumsum, scores)[0, 1])

            make_combined_plot(plots, f'Cumulative {metric}', f'{metaname}, ordered by last values',
                               make_legend=True, do_plot_log=False, smooth_window=5)

    if plot_samples:
        # plot 10 samples
        plots = dict()
        plots['epoch'] = list(range(len(per_datapoint[0]['l2'])))
        for i in range(0, 100, 10):
            plots[f'{i}'] = [per_datapoint[i]['l2'][j] for j in plots['epoch']]
        make_combined_plot(plots, f'10 first samples', metaname, make_legend=True, do_plot_log=False, smooth_window=5)

    return res, scores, score_epochs, per_datapoint
