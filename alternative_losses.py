import math
from collections import defaultdict
from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import load_pkl_file, make_single_plots, get_correlation_metrics


def huber_loss(x, delta=1):
    if np.abs(x) < delta:
        return 0.5 * x ** 2
    else:
        return delta * (np.abs(x) - 0.5 * delta)


def smooth_probability_loss(x, scale=math.pi):
    return 1 - 1 / ((scale * x)**2 + 1)


def get_alternative_losses(output_dir, start_epoch, end_epoch, step_size=1):
    metaname = output_dir.split('/')[-1]
    res = defaultdict(list)
    scores = []
    score_epochs = []

    for epoch in range(start_epoch, end_epoch + 1, step_size):
        if epoch % 50 == 0:
            print(f"Processing epoch {epoch}")

        file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')

        data = load_pkl_file(file_path)

        val_data = data['val_data']
        prediction_results = data['val_prediction_results']
        per_sample_losses = data['per_sample_losses']

        losses = defaultdict(list)

        for index in range(int(1e5)):
            batch_number = index // 256
            index_in_batch = index % 256

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
            losses['huber'].append(huber_loss(l2_distance))
            losses['smooth_prob_pi'].append(smooth_probability_loss(l2_distance, scale=math.pi))
            losses['smooth_prob_5'].append(smooth_probability_loss(l2_distance, scale=5))
            losses['per_sample_loss'].append(np.sum(np.abs(data['per_sample_losses'][batch_number][index_in_batch])))

            # losses['torch_mse'].append(torch.nn.functional.mse_loss(torch.tensor(pred_action), torch.tensor(gt_action)).item())

        res['epoch'].append(epoch)
        for k, v in losses.items():
            res[k].append(np.mean(v))

        if 'test/mean_score' in data:
            scores.append(data['test/mean_score'])
            score_epochs.append(epoch)

    smooth_window = 5
    make_single_plots(res, 'Alternative losses', metaname, smooth_window)
    if scores:
        make_single_plots({'mean_score': scores, 'epoch': score_epochs}, 'Mean score', metaname, smooth_window)

    print(get_correlation_metrics(res, scores, score_epochs))
