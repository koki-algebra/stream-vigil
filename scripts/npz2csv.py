import os

import numpy as np
import pandas as pd


def main():
    input_dir = "./data/ADBench"
    output_dir = "./data/ADBench"

    for filename in os.listdir(input_dir):
        if filename.endswith(".npz"):
            npz_path = os.path.join(input_dir, filename)

            data = np.load(npz_path)

            X_df = pd.DataFrame(data["X"])
            y_df = pd.DataFrame(data["y"])
            df = pd.concat([X_df, y_df], axis=1)

            output_filename = os.path.basename(npz_path).replace(".npz", ".csv.gz")
            output_path = os.path.join(output_dir, output_filename)

            df.to_csv(output_path, header=False, index=False, compression="gzip")

            print(f"Processed {npz_path} and saved to {output_path}")


if __name__ == "__main__":
    main()
