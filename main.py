from datetime import datetime

from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from oracle import get_oracle_losses
from utils import process_episode_lengths


def main():
    process_episode_lengths()

    # final dirs
    # output_dir, n_epochs = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.28/17.36.44_train_diffusion_unet_lowdim_kitchen_lowdim", 4000
    # output_dir, n_epochs = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.12.03/20.27.06_train_diffusion_unet_lowdim_tool_hang_lowdim", 1000
    output_dir, n_epochs = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.12.02/09.56.38_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs", 3500

    print("Processing", output_dir)
    print(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    # get_oracle_losses(output_dir, start_epoch=0, end_epoch=n_epochs, oracle_epoch=30, step_size=10)
    get_min_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=10)
    get_alternative_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=10)
    get_closest_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=10)


if __name__ == '__main__':
    main()
