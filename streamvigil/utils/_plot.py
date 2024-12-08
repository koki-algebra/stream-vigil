from typing import List

import matplotlib.pyplot as plt


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
    # Concept Drift Detection
    plt.figure(figsize=(12, 6))

    for i in range(3):
        reliabilities = reliabilities_list[i]
        detected = detected_list[i]
        dataset_name = dataset_names[i]
        reliability_color = reliability_colors[i]
        detected_color = detected_colors[i]

        plt.plot(reliabilities, label=f"Reliability ({dataset_name})", color=reliability_color)

        detected_indices = [i for i, x in enumerate(detected) if x == 1]
        plt.scatter(
            detected_indices, [1] * len(detected_indices), label=f"Detected ({dataset_name})", color=detected_color
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

    for i in range(3):
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
