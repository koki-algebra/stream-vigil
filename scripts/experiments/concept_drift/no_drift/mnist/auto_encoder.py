from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchvision import datasets, transforms

from streamvigil.detectors import BasicAutoEncoder, BasicDetector
from streamvigil.utils import filter_index, set_seed, to_anomaly_labels

RANDOM_STATE = 80
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128


def main():
    set_seed(RANDOM_STATE)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data/pytorch",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data/pytorch",
        train=False,
        download=True,
        transform=transform,
    )

    # Filter label
    train_filtered_idx = filter_index(
        train_dataset.targets,
        normal_labels=[1, 2, 3],
        anomaly_labels=[7, 8, 9],
    )
    train_dataset.targets = to_anomaly_labels(
        train_dataset.targets[train_filtered_idx],
        normal_labels=[1, 2, 3],
    )
    train_dataset.data = train_dataset.data[train_filtered_idx]

    test_filterd_idx = filter_index(
        test_dataset.targets,
        normal_labels=[1, 2, 3],
        anomaly_labels=[0, 4, 5],
    )
    test_dataset.targets = to_anomaly_labels(
        test_dataset.targets[test_filterd_idx],
        normal_labels=[1, 2, 3],
    )
    test_dataset.data = test_dataset.data[test_filterd_idx]

    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    )

    # Model
    auto_encoder = BasicAutoEncoder(
        encoder_dims=[784, 588, 392, 196],
        decoder_dims=[196, 392, 588, 784],
        batch_norm=True,
    )
    detector = BasicDetector(auto_encoder)

    # Training
    for X, y in train_loader:
        X = X.view(X.size(0), -1)

        detector.stream_train(X)

    # Evaluation
    auroc = BinaryAUROC()
    auprc = BinaryAUPRC()

    for X, y in test_loader:
        X = X.view(X.size(0), -1)

        scores = detector.predict(X)

        auroc.update(scores, y)
        auprc.update(scores, y)

    print(f"AUROC: {auroc.compute():0.5f}")
    print(f"AUPRC: {auprc.compute():0.5f}")


if __name__ == "__main__":
    main()
