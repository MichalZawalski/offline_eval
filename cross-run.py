from collections import defaultdict

import numpy as np

from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from utils import get_cross_run_correlations, make_combined_plot


def compare_runs(output_dirs, start_epoch, end_epoch, step_size=1):
    all_losses = {}
    all_scores = {}

    for output_dir in output_dirs:
        metaname = output_dir.split('/')[-1]
        all_losses[metaname] = dict()

        for metric_func in [
            get_alternative_losses,
            get_min_losses,
            # get_closest_losses,
        ]:

            res, scores, score_epochs = metric_func(output_dir, start_epoch, end_epoch, step_size, do_plot=False)

            all_losses[metaname] |= res
            all_scores[metaname] = scores

    correlations = get_cross_run_correlations(all_losses, all_scores, score_epochs)

    plots = defaultdict(dict)

    for metric_name in correlations:
        for cor_name, cor_values in correlations[metric_name].items():
            plots[cor_name][metric_name] = cor_values

    for corr_name, plot in plots.items():
        plot['epoch'] = score_epochs
        make_combined_plot(plot, corr_name, 'cross-run', make_legend=True, do_plot_log=False)

    # for metric in ['top_10%_losses', 'l2_sq', 'smooth_prob_pi']:
    #     make_combined_plot({'epoch': score_epochs} | {metaname: np.log(all_losses[metaname][metric]) for metaname in all_losses},
    #                        metric, 'cross-run', make_legend=True, do_plot_log=False)
    #
    # make_combined_plot(
    #     {'epoch': score_epochs} | {metaname: all_scores[metaname] for metaname in all_losses},
    #     'scores', 'cross-run', make_legend=True, do_plot_log=False)


if __name__ == '__main__':
    output_dirs = [
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.45.25_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.14_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.43_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.49.23_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.55.29_train_diffusion_unet_lowdim_tool_hang_lowdim",
    ]
    compare_runs(output_dirs, 0, 4500, 200)