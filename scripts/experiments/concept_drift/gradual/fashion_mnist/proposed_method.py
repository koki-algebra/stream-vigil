from logging import getLogger
from logging.config import dictConfig

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil.core import Model, ModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import filter_index, set_seed

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
INIT_BATCHES = 20


def main():
    set_seed(RANDOM_STATE)

    with open("./notebooks/logging.yml", encoding="utf-8") as file:
        config = safe_load(file)
    dictConfig(config)
    logger = getLogger(__name__)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root="./data/pytorch",
        train=True,
        download=True,
        transform=transform,
    )

    concept_a_idx = filter_index(
        train_dataset.targets,
        normal_labels=[1, 2],
        anomaly_labels=[7, 8, 9],
        sample_size=0.75,
    )
    concept_b_idx = filter_index(
        train_dataset.targets,
        normal_labels=[1, 2, 3, 4],
        anomaly_labels=[7, 8, 9],
        sample_size=0.10,
    )
    concept_c_idx = filter_index(
        train_dataset.targets,
        normal_labels=[3, 4],
        anomaly_labels=[7, 8, 9],
        sample_size=0.75,
    )

    result_idx = concept_a_idx + concept_b_idx + concept_c_idx
    train_dataset.data = train_dataset.data[result_idx]
    train_dataset.targets = train_dataset.targets[result_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
    )

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    # Model Pool
    model_pool = ModelPool[Model](
        detector,
        historical_window_size=500,
        latest_window_size=500,
        last_trained_size=500,
        window_gap=500,
    )

    # Training
    for X, y in train_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            print("y:", y)

            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                adapted_model_id = model_pool.find_adapted_model()
                if adapted_model_id is not None:
                    model_pool.current_model_id = adapted_model_id

                    logger.info(f"find adapted model: {adapted_model_id}")
                else:
                    # Add new model
                    new_model_id = model_pool.add_model()
                    model_pool.current_model_id = new_model_id

                    logger.info(f"add new model: {new_model_id}")

        # Train current model
        model_pool.stream_train(X)


if __name__ == "__main__":
    main()
