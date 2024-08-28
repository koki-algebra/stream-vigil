import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.utils import resample
from torchvision import datasets, transforms

from streamvigil.utils import set_seed


def generate_output_path(normal_labels: List[int], anomaly_labels: List[int], base_dir="./data"):
    normal_str = ",".join(map(str, normal_labels))
    anomaly_str = ",".join(map(str, anomaly_labels))
    filename = f"mnist_normal={normal_str},anomaly={anomaly_str}.csv"
    return os.path.join(base_dir, filename)


def process_mnist(
    normal_labels: List[int],
    anomaly_labels: List[int],
    base_dir="./data",
    normal_label=0,
    anomaly_label=1,
):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data/pytorch", train=True, download=True, transform=transform)

    # 画像とラベルを取得
    images = mnist_train.data.numpy()
    labels = mnist_train.targets.numpy()

    # 画像を2Dから1Dに変換 (28x28 -> 784)
    images_2d = images.reshape(images.shape[0], -1)

    # 列名を作成
    pixel_columns = [f"pixel_{i+1}" for i in range(images_2d.shape[1])]
    column_names = pixel_columns + ["original_label"]

    # 画像データとラベルを結合
    data = np.column_stack((images_2d, labels))

    # DataFrameを作成
    df = pd.DataFrame(data, columns=column_names)

    # 指定されたラベルのデータを抽出
    selected_labels = normal_labels + anomaly_labels
    df_selected = df[df["original_label"].isin(selected_labels)].copy()

    # 新しいラベルを作成
    df_selected["label"] = normal_label
    df_selected.loc[df_selected["original_label"].isin(anomaly_labels), "label"] = anomaly_label

    # ラベルごとにデータを分割
    df_normal = df_selected[df_selected["label"] == normal_label]
    df_anomaly = df_selected[df_selected["label"] == anomaly_label]

    # 異常データを5%にダウンサンプリング
    df_anomaly_downsampled = resample(df_anomaly, n_samples=int(len(df_normal) * 0.05))

    # データを結合
    df_final = pd.concat([df_normal, df_anomaly_downsampled])

    # データをシャッフル
    df_final = df_final.sample(frac=1).reset_index(drop=True)

    # カラムを指定された順序に並び替え
    columns_order = pixel_columns + ["original_label", "label"]
    df_final = df_final[columns_order]

    # 出力パスを生成
    output_path = generate_output_path(normal_labels, anomaly_labels, base_dir)

    # CSVとして保存
    df_final.to_csv(output_path, index=False)

    print(f"File saved: {output_path}")
    print(f"Total samples: {len(df_final)}")
    print(f"Normal samples (label {normal_label}): {len(df_final[df_final['label'] == normal_label])}")
    print(f"Anomaly samples (label {anomaly_label}): {len(df_final[df_final['label'] == anomaly_label])}")
    print(f"Percentage of anomalies: {len(df_final[df_final['label'] == anomaly_label]) / len(df_final) * 100:.2f}%")


def main():
    random_state = 80
    set_seed(random_state)

    normal_labels = [7, 8]
    anomaly_labels = [9]
    process_mnist(normal_labels, anomaly_labels)


if __name__ == "__main__":
    main()
