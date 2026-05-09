"""Train the framework-free neural network on the XOR problem."""

from __future__ import annotations

import pandas as pd

from neural_net import NeuralNetwork


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
    data = build_xor_dataframe()

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
