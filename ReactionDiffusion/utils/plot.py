import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib.lines as mlines
import seaborn as sns

def plot_val_over_epoch(output_path, plot_name, epochs, tasks, task_probs, x_title, y_title, plot_title):
    
    colors = plt.cm.jet(np.linspace(0, 1, len(tasks)))

    fig = plt.figure(figsize=(10, 8))

    for i, task in enumerate(tasks):
        plt.plot(range(len(task_probs[task])), task_probs[task], label='Task ' + str(task), color=colors[i])

    plt.title(plot_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.legend(loc='best')
    # add legend to the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.subplots_adjust(right=0.7) 
    plt.savefig(output_path + '/' + plot_name + '.png')
    

def plot_single_val_over_epoch(output_path, plot_name, epochs, values, x_title, y_title, plot_title):
    

    plt.figure(figsize=(10, 8))

    plt.plot(range(len(values)), values)

    plt.title(plot_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.legend(loc='best')
    plt.show()
    plt.subplots_adjust(right=0.7) 
    plt.savefig(output_path + '/' + plot_name + '.png')



def calculate_disparity(output_path, loss_dict, number, loss_name):
    # sort the loss dictionary by values (loss values)
    sorted_dict = dict(sorted(loss_dict.items(), key=lambda x: x[1][-1]))

    # create a list of keys sorted by values
    sorted_keys = list(sorted_dict.keys())

    # calculate the average loss
    avg_loss = sum([val[-1] for val in list(loss_dict.values())]) / len(loss_dict)


    # select the best and worst performing groups
    best_groups = sorted_keys[:number]
    worst_groups = sorted_keys[-number:]

    # calculate the disparity
    best_avg_loss = sum([loss_dict[g][-1] for g in best_groups]) / len(best_groups)

    worst_avg_loss = sum([loss_dict[g][-1] for g in worst_groups]) / len(worst_groups)
    Disparity_b_w = worst_avg_loss - best_avg_loss
    Disparity_av_w = worst_avg_loss - avg_loss
    Disparity_av_b = avg_loss - best_avg_loss

    # create a dataframe with the results
    data = {'Best performing groups': best_groups,
            'Worst performing groups': worst_groups,
            'Average loss': [avg_loss],
            'Best performing average loss': [best_avg_loss],
            'Worst performing average loss': [worst_avg_loss],
            'Disparity_b_w': [Disparity_b_w],
            'Disparity_av_w': [Disparity_av_w],
            'Disparity_av_b': [Disparity_av_b]}

    
    with open(output_path + '/disparity' + str(number) + '_' + loss_name, "w") as f:
        json.dump(data, f)

    return data


# def plot_loss_convergence(output_path, nus, plot_name, loss_dict, thresholds):
#     epochs = range(1, len(list(loss_dict.values())[0]) + 1)
#     num_tasks = len(loss_dict)
#     colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

#     fig, ax = plt.subplots()

#     text_threshold ={}
#     for t in thresholds:
#         text_threshold[t] = 0
#     for i, (task, loss_values) in enumerate(loss_dict.items()):
#         if task in nus:
#             ax.plot(epochs, loss_values, label=task, color=colors[i])
#             for threshold in thresholds:
#                 epoch_threshold = next((i for i, val in enumerate(loss_values) if val <= threshold), None)
#                 if epoch_threshold:
#                     ax.errorbar(epoch_threshold+1, loss_values[epoch_threshold], yerr=0.05, fmt='none', capsize=5, capthick=1.5, color=colors[i])
                    
#                     if text_threshold[threshold] != epoch_threshold:
#                         ax.text(epoch_threshold+1, loss_values[epoch_threshold]+0.07, f'{threshold:.4f}', rotation=90, va='bottom', ha='right', color=colors[i])
#                         text_threshold[threshold] = epoch_threshold
#                     # ax.axvline(x=epoch_threshold+1, linestyle='--', color=colors[i])

#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title('Loss Convergence Plot')
#     plt.subplots_adjust(right=0.7) 
#     # ax.legend()
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     # plt.show()
#     plt.savefig(output_path + '/' + plot_name + '.png')

def plot_loss_convergence(output_path, nus, plot_name, loss_dict, thresholds):
    epochs = range(1, len(list(loss_dict.values())[0]) + 1)
    num_tasks = len(loss_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (task, loss_values) in enumerate(loss_dict.items()):
        if task in nus:
            ax.plot(epochs, loss_values, label=task, color=colors[i])
            for j, threshold in enumerate(thresholds):
                epoch_threshold = next((epoch for epoch, loss in enumerate(loss_values) if loss <= threshold), None)
                if epoch_threshold is not None:
                    color = plt.cm.tab20(np.linspace(0, 1, len(thresholds)))[j]
                    ax.axhline(y=threshold, linestyle='--', color=color)
                    # abel=f'Threshold: {threshold:.6f}'

                    # if i == 0:
                    #     ax.plot([], [], linestyle='--', color=color, label=f'Threshold: {threshold:.6f}')

                    # if epoch_threshold != 0:
                    #     ax.errorbar(epoch_threshold+1, threshold, yerr=0.05, fmt='none', capsize=5, capthick=1.5, color=color)
                        # ax.text(epoch_threshold+1, threshold-0.05, f'Epoch {epoch_threshold+1}', rotation=0, va='top', ha='center', color=color)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence Plot')
    plt.subplots_adjust(right=0.6, bottom=0.2) 

    # Create a legend for the task names
    task_legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create a legend for the thresholds and the colors assigned to them
    threshold_legend_handles = []
    threshold_legend_labels = []
    for j, threshold in enumerate(thresholds):
        color = plt.cm.tab20(np.linspace(0, 1, len(thresholds)))[j]
        threshold_legend_handles.append(mlines.Line2D([], [], color=color, linestyle='--'))
        threshold_legend_labels.append(f'{threshold:.6f}')
    threshold_legend = ax.legend(threshold_legend_handles, threshold_legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)

    # Add both legends to the plot
    ax.add_artist(task_legend)

    plt.savefig(output_path + '/' + plot_name + '.png')
    plt.show()


def plot_weight_vs_loss(output_path, plot_name, loss_dict, selected_tasks, prob_dict):
    plt.figure(figsize=(8,6))
    for task, losses in loss_dict.items():
        if task in selected_tasks:
            weights = [prob_dict[task][i] for i in range(len(losses))]
            plt.plot(losses, weights, label=task, linewidth=2)
    # plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Weight')
    plt.xlabel('Loss')
    plt.title('Weight vs Loss')
    plt.subplots_adjust(right=0.7) 
    plt.savefig(output_path + '/' + plot_name + '.png')
    

def plot_stacked_bar(output_path, name, loss_dict, prob_dict, tasks):
    fig, ax = plt.subplots()

    num_tasks = len(tasks) + 1
    
    new_loss = {}
    new_probs = {}
    
    epochs = len(prob_dict[tasks[0]])
    sum_loss = np.zeros(epochs)
    sum_prob = np.zeros(epochs)
    for i, nu in enumerate(list(loss_dict.keys())):
        if nu in tasks:
            new_loss[nu] = loss_dict[nu]
            new_probs[nu] = prob_dict[nu]
        else:
            sum_loss = np.add(sum_loss, loss_dict[nu])
            sum_prob = np.add(sum_prob, prob_dict[nu])
    
    new_loss['others'] = sum_loss
    new_probs['others'] = sum_prob
            
        
    
    # Generate colors for each task based on its index
    colors = sns.color_palette("hls", num_tasks).as_hex()

    # colors = sns.color_palette("bright", num_tasks).as_hex()
    # colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, num_tasks)]
    
    # Generate x values for each epoch
    x = np.arange(epochs)
    
    # Initialize bottom values to zero for the first bar
    bottom = np.zeros(epochs)
    
    for i, task in enumerate(new_loss.keys()):
        # Get the probabilities and losses for this task
        probs = new_probs[task]
        losses = new_loss[task]
        
        # Normalize the probabilities
        norm_probs = np.array(probs) / sum(probs)
        
        # Calculate the heights of each segment of the bar
        heights = norm_probs * losses
        
        # Plot the segment for this task's bar
        ax.bar(x, heights, bottom=bottom, color=colors[i], label=task)
        
        # Update the bottom values for the next task's bar
        bottom += heights
        
    # Add a legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set the x and y labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    plt.subplots_adjust(right=0.7) 
    
    plt.show()
    plt.subplots_adjust(right=0.7) 
    plt.savefig(output_path + '/' + name + '.png')


def plot_loss_prob_3d(loss_dict, prob_dict):
    fig = go.Figure()

    # Iterate over tasks and add a trace for each task
    for task, loss in loss_dict.items():
        prob = prob_dict[task]
        fig.add_trace(
            go.Scatter3d(
                x=list(range(len(loss))),
                y=prob,
                z=loss,
                mode='lines',
                name=task
            )
        )

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='Probability',
            zaxis_title='Loss'
        ),
        title='Probability vs Loss over Epochs for Different Tasks'
    )

    # Show the figure
    fig.show()