from datetime import datetime

from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from oracle import get_oracle_losses
from utils import process_episode_lengths


def main():
    process_episode_lengths()

    sim_precision = 2
    pi_precision = 10000

    # final dirs
    # output_dir, n_epochs, precision = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.28/17.36.44_train_diffusion_unet_lowdim_kitchen_lowdim", 2400, sim_precision
    # output_dir, n_epochs, precision = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.12.03/20.27.06_train_diffusion_unet_lowdim_tool_hang_lowdim", 2900, sim_precision
    # output_dir, n_epochs, precision = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.12.02/09.56.38_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs", 4900, sim_precision

    # output_dir, n_epochs, precision = "/home/michal/project_data/offline_validation/datasets/full_train_bc_polar-solver_noise-bits-0_dim-64_id_460696", 4900, sim_precision
    # output_dir, n_epochs, precision = "/home/michal/project_data/offline_validation/datasets/full_train_bc_cartesian-solver_noise-bits-10_dim-32_id_496904", 4900, sim_precision

    output_dir, n_epochs, precision = "/home/michal/project_data/offline_validation/datasets/pi_datasets/2024_12_19_trajectories/ldaug7ak", 1000000, pi_precision

    print("Processing", output_dir)
    print(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    # get_oracle_losses(output_dir, start_epoch=0, end_epoch=n_epochs, oracle_epoch=30, step_size=precision)
    # get_min_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision)
    get_alternative_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision)
    # get_closest_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision)


if __name__ == '__main__':
    main()
