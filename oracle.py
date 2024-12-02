from datetime import datetime

import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import load_pkl_file, get_device


# Calculate loss using GPU acceleration
def calculate_loss_gpu_after_oracle(data, data_oracle):
    # load data at this epoch
    val_data = data['val_data']
    predicted_actions = data['val_data_pred']
    multiple_predicted_actions = data['val_data_multiple_preds']
    per_sample_losses = data['per_sample_losses']

    # load data at the oracle epoch
    val_data_oracle = data_oracle['val_data']
    predicted_actions_oracle = data_oracle['val_data_pred']
    multiple_predicted_actions_oracle = data_oracle['val_data_multiple_preds']
    per_sample_losses_oracle = data_oracle['per_sample_losses']

    # Store all states and actions from this epoch in PyTorch tensors
    all_states = []
    all_actions = []
    all_predicted_actions = []
    all_multiple_preds = []

    # total_number_of_states = 13482 # for Push-T
    # total_number_of_states = 2405  # for Block Pushing
    total_number_of_states = 1400  # for Square
    # total_number_of_states = 458 # for Lift

    for index in range(total_number_of_states):
        batch_number = index // 256
        index_in_batch = index % 256
        try:
            state = val_data[batch_number]['obs'][index_in_batch]
            all_states.append(torch.tensor(state[0], device=get_device()))
            action = val_data[batch_number]['action'][index_in_batch]
            all_actions.append(torch.tensor(action[0], device=get_device()))
            predicted_action = predicted_actions[batch_number][index_in_batch]
            all_predicted_actions.append(torch.tensor(predicted_action[0][:action.shape[1]], device=get_device()))

            preds = []

            x = multiple_predicted_actions[batch_number]
            for i in range(len(x)):
                tensor_to_add = x[i][index_in_batch][0]

                # Check and convert to PyTorch tensor if needed
                if isinstance(tensor_to_add, np.ndarray):  # If it's a numpy array
                    tensor_to_add = torch.tensor(tensor_to_add, device=get_device())  # Convert to tensor

                preds.append(tensor_to_add)

            # preds is a list of predicted actions
            # stack them to make a tensor
            multiple_preds = torch.stack(preds)
            all_multiple_preds.append(multiple_preds)

        except:
            print(f"Indexing error: {index}")

    all_states = torch.stack(all_states)
    all_actions = torch.stack(all_actions)
    all_predicted_actions = torch.stack(all_predicted_actions)
    all_multiple_preds = torch.stack(all_multiple_preds)

    all_oracle_states = []
    all_oracle_actions = []
    all_oracle_predicted_actions = []
    all_oracle_multiple_preds = []

    for index in range(total_number_of_states):
        batch_number = index // 256
        index_in_batch = index % 256
        try:
            state = val_data_oracle[batch_number]['obs'][index_in_batch]
            all_oracle_states.append(torch.tensor(state[0], device=get_device()))
            action = val_data_oracle[batch_number]['action'][index_in_batch]
            all_oracle_actions.append(torch.tensor(action[0], device=get_device()))
            predicted_action = predicted_actions_oracle[batch_number][index_in_batch]
            all_oracle_predicted_actions.append(torch.tensor(predicted_action[0][:action.shape[1]], device=get_device()))

            preds = []

            x = multiple_predicted_actions_oracle[batch_number]
            for i in range(len(x)):
                tensor_to_add = x[i][index_in_batch][0]

                # Check and convert to PyTorch tensor if needed
                if isinstance(tensor_to_add, np.ndarray):  # If it's a numpy array
                    tensor_to_add = torch.tensor(tensor_to_add, device=get_device())  # Convert to tensor

                preds.append(tensor_to_add)

            # preds is a list of predicted actions
            # stack them to make a tensor
            multiple_preds = torch.stack(preds)
            all_oracle_multiple_preds.append(multiple_preds)

        except:
            print(f"Indexing error: {index}")

    all_oracle_states = torch.stack(all_oracle_states)
    all_oracle_actions = torch.stack(all_oracle_actions)
    all_oracle_predicted_actions = torch.stack(all_oracle_predicted_actions)
    all_oracle_multiple_preds = torch.stack(all_oracle_multiple_preds)

    # Compute losses
    final_oracle_losses = []
    final_best_of_n_losses = []
    no_of_datapoints_used = 0
    for i in range(len(all_states)):
        s = all_states[i]
        a_pred = all_predicted_actions[i]
        # find the multiple predictions for the current state
        multiple_preds = all_multiple_preds[i]
        # set min_loss = norm between a_pred and all_actions[i]
        min_loss = torch.norm(all_actions[i] - a_pred, p=2) ** 2
        for pred in multiple_preds:
            loss = torch.norm(all_actions[i] - pred, p=2) ** 2
            if loss < min_loss:
                min_loss = loss
        final_best_of_n_losses.append(min_loss)
        # find the multiple predictions for the oracle at the current state
        oracle_multiple_preds = all_oracle_multiple_preds[i]
        # find the lowest loss among the multiple predictions
        oracle_min_loss = float('inf')
        for pred in oracle_multiple_preds:
            loss = torch.norm(all_oracle_actions[i] - pred, p=2) ** 2
            if loss < oracle_min_loss:
                oracle_min_loss = loss
        if oracle_min_loss > min_loss:
            final_oracle_losses.append(min_loss)
            no_of_datapoints_used += 1

            # Calculate average loss
    average_oracle_loss = torch.mean(torch.stack(final_oracle_losses)).item()
    average_best_of_n_loss = torch.mean(torch.stack(final_best_of_n_losses)).item()
    return average_oracle_loss, average_best_of_n_loss, no_of_datapoints_used


# Calculate loss using GPU acceleration
def calculate_loss_gpu_before_oracle(data):
    val_data = data['val_data']
    predicted_actions = data['val_data_pred']
    multiple_predicted_actions = data['val_data_multiple_preds']
    per_sample_losses = data['per_sample_losses']

    # Store all states and actions in PyTorch tensors
    all_states = []
    all_actions = []
    all_predicted_actions = []
    all_multiple_preds = []

    # total_number_of_states = 13482 # for Push-T
    # total_number_of_states = 2405  # for Block Pushing
    total_number_of_states = 1400  # for Square
    # total_number_of_states = 458 # for Lift

    for index in range(total_number_of_states):
        batch_number = index // 256
        index_in_batch = index % 256
        try:
            state = val_data[batch_number]['obs'][index_in_batch]
            all_states.append(torch.tensor(state[0], device=get_device()))
            action = val_data[batch_number]['action'][index_in_batch]
            all_actions.append(torch.tensor(action[0], device=get_device()))
            predicted_action = predicted_actions[batch_number][index_in_batch]
            all_predicted_actions.append(torch.tensor(predicted_action[0][:action.shape[1]], device=get_device()))

            preds = []

            # import ipdb; sb.set_trace()

            x = multiple_predicted_actions[batch_number]
            for i in range(len(x)):
                tensor_to_add = x[i][index_in_batch][0]

                # Check and convert to PyTorch tensor if needed
                if isinstance(tensor_to_add, np.ndarray):  # If it's a numpy array
                    tensor_to_add = torch.tensor(tensor_to_add, device=get_device())  # Convert to tensor

                preds.append(tensor_to_add)

            # import ipdb; ipdb.set_trace()

            # preds is a list of predicted actions
            # stack them to make a tensor
            multiple_preds = torch.stack(preds)
            all_multiple_preds.append(multiple_preds)

        except:
            import ipdb;
            ipdb.set_trace()
            print(f"Indexing error: {index}")

    all_states = torch.stack(all_states)
    all_actions = torch.stack(all_actions)
    all_predicted_actions = torch.stack(all_predicted_actions)
    all_multiple_preds = torch.stack(all_multiple_preds)

    # Compute losses
    final_losses = []

    for i in range(len(all_states)):
        s = all_states[i]
        a_pred = all_predicted_actions[i]

        # find the multiple predictions for the current state
        multiple_preds = all_multiple_preds[i]

        # find the lowest loss among the multiple predictions
        min_loss = torch.norm(all_actions[i] - a_pred, p=2) ** 2
        for pred in multiple_preds:
            loss = torch.norm(all_actions[i] - pred, p=2) ** 2
            if loss < min_loss:
                min_loss = loss

        final_losses.append(min_loss)

    # Calculate average loss
    average_loss = torch.mean(torch.stack(final_losses)).item()

    return average_loss, total_number_of_states


# Main function to process all epochs and generate heatmaps
def get_oracle_losses(output_dir, start_epoch, end_epoch, oracle_epoch, step_size=1):
    all_oracle_losses = []
    all_best_of_n_losses = []
    no_of_datapoints_used_list = []

    for epoch in range(start_epoch, end_epoch + 1, step_size):
        if epoch % 10 == 0:
            print(f"Processing epoch {epoch}")
        file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')

        # Load and process the epoch's losses
        data = load_pkl_file(file_path)

        if epoch < oracle_epoch:
            val_loss, no_of_datapoints_used = calculate_loss_gpu_before_oracle(data)
            all_oracle_losses.append(val_loss)
            all_best_of_n_losses.append(val_loss)
            no_of_datapoints_used_list.append(no_of_datapoints_used)

        elif epoch == oracle_epoch:
            oracle_file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')
            data_oracle = load_pkl_file(oracle_file_path)
            val_loss, no_of_datapoints_used = calculate_loss_gpu_before_oracle(data_oracle)
            all_oracle_losses.append(val_loss)
            all_best_of_n_losses.append(val_loss)
            no_of_datapoints_used_list.append(no_of_datapoints_used)

        elif epoch > oracle_epoch:
            val_oracle_loss, val_best_of_n_loss, no_of_datapoints_used = calculate_loss_gpu_after_oracle(data,
                                                                                                         data_oracle)
            print(f"No of datapoints used: {no_of_datapoints_used}")
            all_oracle_losses.append(val_oracle_loss)
            all_best_of_n_losses.append(val_best_of_n_loss)
            no_of_datapoints_used_list.append(no_of_datapoints_used)

    print(f"Oracle Losses: {all_oracle_losses}")
    print("-------------------")
    print(f"Best of N Losses: {all_best_of_n_losses}")
    print("-------------------")
    print(f"No of Datapoints Used: {no_of_datapoints_used_list}")

    # Plot Oracle Losses
    plt.figure()
    plt.plot(range(start_epoch, end_epoch + 1, step_size), all_oracle_losses, label='Oracle Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Oracle Loss vs Epoch')
    plt.legend()
    plt.savefig(f'figures/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_oracle_loss.png')
    plt.close()

    # Plot Best of N Losses
    plt.figure()
    plt.plot(range(start_epoch, end_epoch + 1, step_size), all_best_of_n_losses, label='Best of N Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Best of N Loss vs Epoch')
    plt.legend()
    plt.savefig(f'figures/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_best_of_n_loss.png')
    plt.close()

    # Plot No of Datapoints Used
    plt.figure()
    plt.plot(range(start_epoch, end_epoch + 1, step_size), no_of_datapoints_used_list, label='No of Datapoints Used')
    plt.xlabel('Epoch')
    plt.ylabel('No of Datapoints Used')
    plt.title('No of Datapoints Used vs Epoch')
    plt.legend()
    plt.savefig(f'figures/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_no_of_datapoints_used.png')
    plt.close()


# # Example usage
# output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.33.07_train_diffusion_unet_lowdim_kitchen_lowdim"
# # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.34.04_train_diffusion_unet_lowdim_tool_hang_lowdim"
# # output_dir = "/home/michal/code/offline_validation/DP_validation/data/outputs/2024.11.12/09.35.57_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"
# main(output_dir, start_epoch=0, end_epoch=50, oracle_epoch=25)  # for Square
# # main(output_dir, start_epoch=0, end_epoch=1559, oracle_epoch=50) # for Lift
# # main(output_dir, start_epoch=0, end_epoch=4895) # for PushT
# # main(output_dir, start_epoch=0, end_epoch=3580, oracle_epoch=50) # for Block Pushing