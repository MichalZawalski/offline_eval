from collections import defaultdict
from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import load_pkl_file, make_single_plots, get_correlation_metrics


def get_min_losses(output_dir, start_epoch, end_epoch, step_size=1, do_plot=True):
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
        # multiple_predicted_actions = data['val_data_multiple_preds']

        losses = []

        for index in range(int(1e5)):
            batch_number = index // 256
            index_in_batch = index % 256

            if batch_number >= len(val_data) or index_in_batch >= len(val_data[batch_number]['action']):
                break

            gt_action = prediction_results[batch_number]['gt_action'][index_in_batch]
            pred_action = prediction_results[batch_number]['pred_action'][index_in_batch]
            loss = np.linalg.norm(gt_action - pred_action, ord=2) ** 2
            losses.append(loss)

        sorted_losses = list(sorted(losses))

        res['epoch'].append(epoch)
        res['loss'].append(np.mean(sorted_losses))
        res['top_10%_losses'].append(np.mean(sorted_losses[:int(len(sorted_losses) * 0.1)]))
        res['top_100_losses'].append(np.mean(sorted_losses[:100]))

        if 'test/mean_score' in data:
            scores.append(data['test/mean_score'])
            score_epochs.append(epoch)

    if do_plot:
        smooth_window = 5
        make_single_plots(res, 'Minimal losses', metaname, smooth_window)

        print(get_correlation_metrics(res, scores, score_epochs))

    return res, scores, score_epochs
