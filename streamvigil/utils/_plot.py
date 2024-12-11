from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_aucus_result(
    reliabilities_list: List[List[float]],
    detected_list: List[List[int]],
    num_models_list: List[List[int]],
    reliability_colors: List[str],
    detected_colors: List[str],
    num_model_colors: List[str],
    dataset_names: List[str],
    drift_type: str,
):
    plt.rcParams.update({
        "font.size": 14,
    })

    plt.figure(figsize=(12, 6))

    for i in range(len(reliabilities_list)):
        reliabilities = reliabilities_list[i]
        detected = detected_list[i]
        dataset_name = dataset_names[i]
        reliability_color = reliability_colors[i]
        detected_color = detected_colors[i]

        plt.plot(reliabilities, label=f"Reliability ({dataset_name})", color=reliability_color)

        detected_indices = [i for i, x in enumerate(detected) if x == 1]
        plt.scatter(
            detected_indices, [1] * len(detected_indices),
            label=f"Detected ({dataset_name})",
            color=detected_color,
        )

    plt.xlabel("Iterations")
    plt.ylabel("Model Pool Reliability")
    plt.legend()
    plt.title(f"Concept Drift Detection ({drift_type})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Number of Models in Model Pool
    plt.figure(figsize=(12, 6))

    for i in range(len(reliabilities_list)):
        num_models = num_models_list[i]
        dataset_name = dataset_names[i]
        num_model_color = num_model_colors[i]

        plt.plot(num_models, label=f"Number of Models ({dataset_name})", color=num_model_color)

    plt.xlabel("Iterations")
    plt.ylabel("Number of Models")
    plt.legend()
    plt.title(f"Number of Models in Model Pool ({drift_type})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_proposed_result(
    losses_list: List[List[float]],
    detected_list: List[List[int]],
    loss_colors: List[str],
    detected_colors: List[str],
    dataset_names: List[str],
    drift_type: str,
):
    plt.rcParams.update({
        "font.size": 14,
    })

    _, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Drift Detected")
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.1)

    for i in range(len(losses_list)):
        losses = np.array(losses_list[i])
        detected = np.array(detected_list[i])
        loss_color = loss_colors[i]
        detected_color = detected_colors[i]
        dataset_name = dataset_names[i]

        # plot
        ax1.plot(losses, color=loss_color, label=f"Losses ({dataset_name})")

        # scatter plot
        detected_indices = np.where(detected == 1)[0]
        ax2.scatter(
            detected_indices,
            [1] * len(detected_indices),
            color=detected_color,
            label=f"Drift Detected ({dataset_name})",
            alpha=0.6,
        )

        # legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(drift_type)
    plt.tight_layout()
    plt.show()
