from typing import List

from scripts.experiments.concept_drift.abrupt.proposed_method import fmnist, kmnist, mnist
from streamvigil.utils import plot_proposed_result

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

LATEST_WINDOW_SIZE = 500
HISTORICAL_WINDOW_SIZE = 500
LAST_TRAINED_WINDOW_SIZE = 500
WINDOW_GAP = 500
INIT_BATCHES = 50
ALPHA = 0.05


def main():
    mnist_result = mnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        latest_window_size=LATEST_WINDOW_SIZE,
        historical_window_size=HISTORICAL_WINDOW_SIZE,
        last_window_size=LAST_TRAINED_WINDOW_SIZE,
        window_gap=WINDOW_GAP,
        alpha=ALPHA,
        init_batches=INIT_BATCHES,
    )
    fmnist_result = fmnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        latest_window_size=LATEST_WINDOW_SIZE,
        historical_window_size=HISTORICAL_WINDOW_SIZE,
        last_window_size=LAST_TRAINED_WINDOW_SIZE,
        window_gap=WINDOW_GAP,
        alpha=ALPHA,
        init_batches=INIT_BATCHES,
    )
    kmnist_result = kmnist.run(
        random_state=RANDOM_STATE,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        latest_window_size=LATEST_WINDOW_SIZE,
        historical_window_size=HISTORICAL_WINDOW_SIZE,
        last_window_size=LAST_TRAINED_WINDOW_SIZE,
        window_gap=WINDOW_GAP,
        alpha=ALPHA,
        init_batches=INIT_BATCHES,
    )

    # Visualize
    losses_list: List[List[float]] = [
        mnist_result.get("losses"),
        fmnist_result.get("losses"),
        kmnist_result.get("losses"),
    ]
    detected_list: List[List[int]] = [
        mnist_result.get("detected"),
        fmnist_result.get("detected"),
        kmnist_result.get("detected"),
    ]
    loss_colors = ["blue", "teal", "skyblue"]
    detected_colors = ["navy", "purple", "darkgreen"]
    dataset_names = ["MNIST", "FMNIST", "KMNIST"]
    drift_type = "Abrupt Drift"

    plot_proposed_result(
        losses_list,
        detected_list,
        loss_colors,
        detected_colors,
        dataset_names,
        drift_type,
    )


if __name__ == "__main__":
    main()
