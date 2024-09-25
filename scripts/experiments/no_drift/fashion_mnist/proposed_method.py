from logging import getLogger
from logging.config import dictConfig

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yaml import safe_load

from streamvigil.core import Model, ModelPool
from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import set_seed

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    # Model Pool
    model_pool = ModelPool[Model](detector)

    # Number of false positives
    fp_cnt = 0

    # Training
    for X, _ in train_loader:
        X = X.view(X.size(0), -1)

        model_pool.update_window(X)

        current_model = model_pool.get_model(model_pool.current_model_id)

        if current_model.num_batches > INIT_BATCHES:
            # Concept Drift detection
            if current_model.is_drift():
                logger.info("concept drift detected!")

                fp_cnt += 1

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

    logger.info(f"Number of false positives: {fp_cnt}")


if __name__ == "__main__":
    main()
