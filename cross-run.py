from collections import defaultdict

import numpy as np

from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from utils import get_cross_run_correlations, make_combined_plot, get_experiment_data, get_vectors_variance, \
    make_histogram


def compare_runs(output_dirs, start_epoch, end_epoch, step_size=1, plot_minimal_sets=False, ignore_epochs=False):
    all_losses = dict()
    all_scores = dict()
    all_per_datapoint = dict()

    for output_dir in output_dirs:
        metaname = output_dir.split('/')[-1]
        all_losses[metaname] = dict()
        print('Processing run', metaname)

        for metric_func in [
            get_alternative_losses,
            get_min_losses,
            # get_closest_losses,
        ]:

            res, scores, score_epochs, per_datapoint = metric_func(output_dir, start_epoch, end_epoch, step_size, do_plot=False)

            # res = {k: v for k, v in res.items() if k == 'epoch' or '(max' in k}

            if ignore_epochs:  # consider only the last checkpoint
                res = {k: [v[-1]] for k, v in res.items() if k != 'epoch'}
                scores = [scores[-1]]
                score_epochs = [0]
                res['epoch'] = score_epochs

            all_losses[metaname] |= res
            all_scores[metaname] = scores

            print(scores, res)

    correlations = get_cross_run_correlations(all_losses, all_scores, score_epochs)

    plots = defaultdict(dict)

    for metric_name in correlations:
        for cor_name, cor_values in correlations[metric_name].items():
            plots[cor_name][metric_name] = cor_values

    for corr_name, plot in plots.items():
        plot['epoch'] = score_epochs
        make_combined_plot(plot, corr_name, 'cross-run', make_legend=True,
                           do_plot_log=False, mark_points=ignore_epochs)

    for metric in ['top_10%_losses', 'l2', 'smooth_prob_5', 'geom']:
        make_combined_plot({'epoch': score_epochs} | {metaname: (all_losses[metaname][metric]) for metaname in all_losses if metric in all_losses[metaname]},
                           metric, 'cross-run', make_legend=True, do_plot_log=False, mark_points=ignore_epochs)

    make_combined_plot(
        {'epoch': score_epochs} | {metaname: all_scores[metaname] for metaname in all_losses},
        'scores', 'cross-run', make_legend=True, do_plot_log=False, mark_points=ignore_epochs)


def get_cross_run_variance(output_dirs, start_epoch, end_epoch, step_size=1):
    all_variances = defaultdict(list)
    combined_variance = []

    for epoch in range(start_epoch, end_epoch + 1, step_size):
        predictions = defaultdict(list)

        if epoch % 50 == 0:
            print(f"Processing epoch {epoch}")

        for output_dir in output_dirs:
            metaname = output_dir.split('/')[-1]

            data, batch_size = get_experiment_data(output_dir, epoch)
            if data is None:
                continue

            val_data = data['val_data']
            prediction_results = data['val_prediction_results']

            for index in range(int(1e5)):
                batch_number = index // batch_size
                index_in_batch = index % batch_size

                if batch_number >= len(val_data) or index_in_batch >= len(val_data[batch_number]['action']):
                    break

                all_predictions = [pred[index_in_batch].flatten() for pred in prediction_results[batch_number]['multiple_preds']]
                predictions[metaname].append(all_predictions)

        epoch_variances = defaultdict(list)
        epoch_combined_variance = []

        for i in range(index):
            for metaname in predictions.keys():
                epoch_variances[metaname].append(get_vectors_variance(predictions[metaname][i]))

            all_predictions = [predictions[metaname][i] for metaname in predictions.keys()]
            all_predictions = np.concatenate(all_predictions, axis=0)
            epoch_combined_variance.append(get_vectors_variance(all_predictions))

        combined_variance.append(np.mean(epoch_combined_variance))

        for metaname in epoch_variances.keys():
            all_variances[metaname].append(np.mean(epoch_variances[metaname]))

        hist, bins = np.histogram(np.log(epoch_combined_variance), bins=np.linspace(-20, 0, 41))
        metanames = list(epoch_variances.keys())
        make_combined_plot({'epoch': ((bins[1:] + bins[:-1]) / 2), 'variance': (hist + 0)},
                           f'Variance distribution (epoch {epoch})', metaname=str(metanames),
                           epoch_label='Log variance', ylimit=(0, 400))

    plot = {
        'epoch': list(range(start_epoch, end_epoch + 1, step_size)),
        'variance': combined_variance
    }
    make_combined_plot(plot, 'Cross-run prediction variance', 'cross-run', make_legend=False,
                       do_plot_log=True, mark_points=False)

    plot = {'epoch': list(range(start_epoch, end_epoch + 1, step_size))} | {k: np.log(v) for k, v in all_variances.items()}
    make_combined_plot(plot, 'Prediction variances', 'log', make_legend=True,
                       do_plot_log=False, mark_points=False)


def plot_minimal_sets(output_dirs, start_epoch, end_epoch, step_size=1, use_best=True):
    all_per_datapoints = dict()

    for output_dir in output_dirs:
        metaname = output_dir.split('/')[-1]

        res, scores, score_epochs, per_datapoint = \
            get_alternative_losses(output_dir, start_epoch, end_epoch, step_size, do_plot=False)

        all_per_datapoints[metaname] = per_datapoint

    combined_plot = dict()

    for metric in ['l1', 'l2', 'l2_sq', 'max', 'geom', 'smooth_prob_pi', 'smooth_prob_5']:
        all_orderings = dict()
        all_places = []

        for metaname, per_datapoint in all_per_datapoints.items():
            order = []
            places = np.zeros(len(per_datapoint))

            for i in range(len(per_datapoint)):
                scale = 1 if use_best else -1
                final_value = np.mean(per_datapoint[i][metric][-10:]) * scale
                order.append((final_value, i))

            order = [i for _, i in sorted(order)]
            all_orderings[metaname] = order

            for i, v in enumerate(order):
                places[v] = i
            all_places.append(places)

        all_places = np.array(all_places)
        cumsum = np.zeros_like(all_places)

        for v in range(all_places.shape[1]):
            for i, pl in enumerate(sorted(all_places[:, v])):
                cumsum[i, int(pl)] += 1

        plots = defaultdict(list)
        plots['epoch'] = list(range(len(per_datapoint)))
        running_sum = np.zeros_like(cumsum[:, 0])

        for i in range(cumsum.shape[1]):
            running_sum += cumsum[:, i]

            for j in [running_sum.shape[0] - 1]:
                # plots[f'int. of {j+1}'].append(running_sum[j])
                plots[f'int. of {j+1}'].append(running_sum[j] / (i + 1))

        make_combined_plot(plots, f'{metric} {"best" if use_best else "worst"} subsets intersections', metaname,
                            make_legend=True, do_plot_log=False, smooth_window=1)
        if len(plots) == 2:
            combined_plot['epoch'] = plots['epoch']
            for k, v in plots.items():
                if k != 'epoch':
                    combined_plot[metric] = v

    if len(combined_plot) > 0:
        make_combined_plot(combined_plot, f'{"best" if use_best else "worst"} subsets intersections',
                           'cross-run', make_legend=True, do_plot_log=False)


if __name__ == '__main__':
    # output_dirs, n_epochs, step_size, ignore_epochs = [  # varying hyperparameters
    #     "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.45.25_train_diffusion_unet_lowdim_tool_hang_lowdim",
    #     "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.14_train_diffusion_unet_lowdim_tool_hang_lowdim",
    #     "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.43_train_diffusion_unet_lowdim_tool_hang_lowdim",
    #     "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.49.23_train_diffusion_unet_lowdim_tool_hang_lowdim",
    #     # "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.55.29_train_diffusion_unet_lowdim_tool_hang_lowdim",  # has more datapoints
    # ], 4500, 200, False
    output_dirs, n_epochs, step_size, ignore_epochs = [  # varying seeds
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2025.01.10/22.27.47_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2025.01.10/22.30.51_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2025.01.10/22.32.50_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2025.01.10/22.33.07_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2025.01.10/22.33.40_train_diffusion_unet_lowdim_tool_hang_lowdim",
    ], 4500, 200, False
    # output_dirs, n_epochs, step_size, ignore_epochs = [
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/pr2xn6r0",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories_final/ldaug7ak",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/wx0gvvmm",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories_final/kh1vqrxr",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories_final/zblar8fp",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/z0hd44iz",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/dtt8wm9u",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/9ihrtr3m",
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/4l71mq2q",
    #     # # "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/3d5ofxqu",  no scores
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/79zk07j7",
    #     # #"/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/f4p1we4b",  no scores
    #     "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/reewbrpb",
    #     # # "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/wbdy7v5h",  no scores
    # ], 1000000, 10000, True

    # compare_runs(output_dirs, 0, n_epochs, step_size, ignore_epochs=ignore_epochs)
    # plot_minimal_sets(output_dirs, 0, 4500, 200, use_best=True)
    get_cross_run_variance(output_dirs, 0, n_epochs, step_size)