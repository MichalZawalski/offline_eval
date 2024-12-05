from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from oracle import get_oracle_losses
from utils import process_episode_lengths


def main():
    process_episode_lengths()

    # # old dirs
    # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.33.07_train_diffusion_unet_lowdim_kitchen_lowdim"
    # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.34.04_train_diffusion_unet_lowdim_tool_hang_lowdim"
    # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.35.57_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"

    # new dirs
    # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.28/17.36.44_train_diffusion_unet_lowdim_kitchen_lowdim"
    # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.28/17.39.52_train_diffusion_unet_lowdim_tool_hang_lowdim"
    output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.12.02/09.56.38_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"

    # get_oracle_losses(output_dir, start_epoch=0, end_epoch=300, oracle_epoch=30, step_size=10)
    # get_min_losses(output_dir, start_epoch=0, end_epoch=405, step_size=2)
    # get_alternative_losses(output_dir, start_epoch=0, end_epoch=405, step_size=2)
    get_closest_losses(output_dir, start_epoch=0, end_epoch=50, step_size=10)


if __name__ == '__main__':
    main()
