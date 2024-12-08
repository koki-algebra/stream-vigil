from typing import List

from scripts.experiments.gradual.arcus import fmnist, kmnist, mnist
from streamvigil.utils import plot_aucus_result

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

LOSS_COLOR = "#00ADD8"
RELIABILITY_COLOR = "#00A29C"
DETECTED_COLOR = "#CE3262"


def main():
    mnist_result = mnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )

    fmnist_result = fmnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )

    kmnist_result = kmnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )

    # Visualize
    reliabilities_list: List[List[float]] = [
        mnist_result.get("reliabilities"),
        fmnist_result.get("reliabilities"),
        kmnist_result.get("reliabilities"),
    ]
    detected_list: List[List[int]] = [
        mnist_result.get("detected"),
        fmnist_result.get("detected"),
        kmnist_result.get("detected"),
    ]
    num_models_list: List[List[int]] = [
        mnist_result.get("num_models"),
        fmnist_result.get("num_models"),
        kmnist_result.get("num_models"),
    ]
    reliability_colors = ["blue", "teal", "skyblue"]
    detected_colors = ["navy", "purple", "darkgreen"]
    num_model_colors = ["blue", "teal", "skyblue"]
    dataset_names = ["MNIST", "FMNIST", "KMNIST"]
    drift_type = "Gradual Drift"

    plot_aucus_result(
        reliabilities_list,
        detected_list,
        num_models_list,
        reliability_colors,
        detected_colors,
        num_model_colors,
        dataset_names,
        drift_type,
    )


if __name__ == "__main__":
    main()
