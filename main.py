from datetime import datetime

from alternative_losses import get_alternative_losses
from closest_loss import get_closest_losses
from min_losses import get_min_losses
from oracle import get_oracle_losses
from utils import process_episode_lengths


def main():
    process_episode_lengths()

    sim_precision = 10
    pi_precision = 10000

    DP_DIR = "/home/michal/code/offline_validation/DP_validation/data/outputs/"
    NEW_DP_DIR = "/home/michal/code/offline_validation/new_DP_validation/data/outputs/"
    PI_DIR = "/home/michal/project_data/offline_validation/datasets/"

    # final dirs
    # output_dir, n_epochs, precision = DP_DIR + "2024.11.28/17.36.44_train_diffusion_unet_lowdim_kitchen_lowdim", 2400, sim_precision
    # output_dir, n_epochs, precision = DP_DIR + "2024.12.03/20.27.06_train_diffusion_unet_lowdim_tool_hang_lowdim", 4000, sim_precision
    # output_dir, n_epochs, precision = DP_DIR + "2024.12.02/09.56.38_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs", 4900, sim_precision
    # output_dir, n_epochs, precision = NEW_DP_DIR + "2024.12.09/15.30.53_train_diffusion_unet_lowdim_square_lowdim", 4900, sim_precision
    # output_dir, n_epochs, precision = NEW_DP_DIR + "2024.12.09/15.28.39_train_diffusion_unet_lowdim_lift_lowdim", 4900, sim_precision

    output_dir, n_epochs, precision = NEW_DP_DIR + "2024.12.16/17.45.25_train_diffusion_unet_lowdim_tool_hang_lowdim", 4500, 200
    # output_dir, n_epochs, precision = NEW_DP_DIR + "2025.02.28/17.18.37_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs", 1750, sim_precision
    # output_dir, n_epochs, precision = NEW_DP_DIR + "2025.02.28/17.19.37_train_diffusion_unet_lowdim_kitchen_lowdim", 1750, sim_precision

    # output_dir, n_epochs, precision = PI_DIR + "full_train_bc_polar-solver_noise-bits-0_dim-64_id_460696", 4900, sim_precision
    # output_dir, n_epochs, precision = PI_DIR + "full_train_bc_cartesian-solver_noise-bits-10_dim-32_id_496904", 4900, sim_precision

    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/pr2xn6r0", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories_final/ldaug7ak", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/wx0gvvmm", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories_final/kh1vqrxr", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories_final/zblar8fp", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/z0hd44iz", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/dtt8wm9u", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/9ihrtr3m", 1000000, pi_precision
    # output_dir, n_epochs, precision = PI_DIR + "pi_datasets/2024_12_19_trajectories/4l71mq2q", 1000000, pi_precision


    print("Processing", output_dir)
    print(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    # get_oracle_losses(output_dir, start_epoch=0, end_epoch=n_epochs, oracle_epoch=30, step_size=precision)
    # get_min_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision, use_smoothing=True)
    get_alternative_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision, order_per_datapoint=False, do_plot=True, use_smoothing=False, trajectory_aggregations=None)
    # get_closest_losses(output_dir, start_epoch=0, end_epoch=n_epochs, step_size=precision)


if __name__ == '__main__':
    main()
