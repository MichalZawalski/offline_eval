import os
from collections import defaultdict

import numpy as np
from utils import load_pkl_file, make_single_plots, get_correlation_metrics, find_closest_states, simulate_rollout


def get_closest_losses(output_dir, start_epoch, end_epoch, step_size=1):
    metaname = output_dir.split('/')[-1]
    taskname = output_dir.split('_train_')[-1]
    res = defaultdict(list)
    scores = []
    score_epochs = []

    closest_states = None

    for epoch in range(start_epoch, end_epoch + 1, step_size):
        if epoch % 10 == 0:
            print(f"Processing epoch {epoch}")

        file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')

        data = load_pkl_file(file_path)

        val_data = data['val_data']
        prediction_results = data['val_prediction_results']

        if closest_states is None:
            closest_states = find_closest_states(val_data, 10)

        losses = defaultdict(list)

        for index in range(int(1e5)):
            batch_number = index // 256
            index_in_batch = index % 256

            if batch_number >= len(val_data) or index_in_batch >= len(val_data[batch_number]['action']):
                break

            # gt_action = prediction_results[batch_number]['gt_action'][index_in_batch]
            pred_action = prediction_results[batch_number]['pred_action'][index_in_batch]
            close_losses = []

            for supply_index in closest_states[index]:
                gt_action = prediction_results[supply_index // 256]['gt_action'][supply_index % 256]
                close_losses.append(np.linalg.norm(gt_action - pred_action, ord=2) ** 2)

            losses['closest min'].append(np.min(close_losses))
            losses['closest mean'].append(np.mean(close_losses))
            losses['closest max'].append(np.max(close_losses))

        # res['sim rollout'].append(np.mean([simulate_rollout(closest_states, prediction_results, taskname) for _ in range(10)]))

        res['epoch'].append(epoch)
        for k, v in losses.items():
            res[k].append(np.mean(v))

        if 'test/mean_score' in data:
            scores.append(data['test/mean_score'])
            score_epochs.append(epoch)

    make_single_plots(res, 'Closest losses', metaname)
    # if scores:
    #     make_single_plots({'mean_score': scores, 'epoch': score_epochs}, 'Mean score', metaname)

    print(get_correlation_metrics(res, scores, score_epochs))
