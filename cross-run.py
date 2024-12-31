from alternative_losses import get_alternative_losses
from utils import get_cross_run_correlations


def compare_runs(output_dirs, start_epoch, end_epoch, step_size=1):
    all_losses = {}
    all_scores = {}

    for output_dir in output_dirs:
        res, scores, score_epochs = get_alternative_losses(output_dir, start_epoch, end_epoch, step_size, do_plot=False)
        metaname = output_dir.split('/')[-1]

        all_losses[metaname] = res
        all_scores[metaname] = scores

    correlations = get_cross_run_correlations(all_losses, all_scores, score_epochs)


if __name__ == '__main__':
    output_dirs = [
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.45.25_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.14_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.48.43_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.49.23_train_diffusion_unet_lowdim_tool_hang_lowdim",
        "/home/michal/code/offline_validation/new_DP_validation/data/outputs/2024.12.16/17.55.29_train_diffusion_unet_lowdim_tool_hang_lowdim",
    ]
    compare_runs(output_dirs, 0, 1000, 200)