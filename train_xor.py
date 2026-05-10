"""Train the framework-free neural network on the XOR problem."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pdb
import pandas as pd

from neural_net import NeuralNetwork


DATA_DIR = Path(__file__).resolve().parent / "data"


def read_data_csv(filename: str | Path, **read_csv_options: object) -> pd.DataFrame:
    """Read a CSV file from the project's data folder."""
    return pd.read_csv(DATA_DIR / filename, **read_csv_options)


def feature_correlation_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the correlation matrix for numeric feature columns."""
    features = frame.loc[:, list(feature_columns)] if feature_columns else frame
    numeric_features = features.select_dtypes(include="number")
    return numeric_features.corr()


def build_xor_dataframe() -> pd.DataFrame:
    """Return XOR inputs and targets as a pandas dataframe."""
    return pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "target": [0.0, 1.0, 1.0, 0.0],
        }
    )


def main() -> None:
    """Train a tunable neural network and print its final predictions."""
    data = read_data_csv("meteostat_daily_72503_2024.csv")
    corr = feature_correlation_matrix(data)
    pdb.set_trace()
    network = NeuralNetwork(
        layer_sizes=[2, 6, 4, 1],
        hidden_activation="sigmoid",
        output_activation="sigmoid",
        seed=42,
    )
    network.fit_dataframe(
        data,
        target_columns=["target"],
        epochs=10000,
        learning_rate=0.8,
        log_every=1000,
    )

    predictions = network.predict_dataframe(data[["x1", "x2"]])
    result = pd.concat([data, predictions.round(4)], axis=1)
    print("\nFinal predictions:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
