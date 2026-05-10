"""A small neural network implementation using pandas and NumPy.

The code in this module is intentionally framework-free: no TensorFlow,
PyTorch, scikit-learn, or other machine-learning libraries. NumPy handles the
array math, and pandas is used for ergonomic tabular input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd


Array = np.ndarray


def sigmoid(values: Array) -> Array:
    """Return the sigmoid activation for each value in an array."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def sigmoid_derivative(values: Array) -> Array:
    """Return the derivative of sigmoid for each pre-activation value."""
    activated = sigmoid(values)
    return activated * (1.0 - activated)


def tanh(values: Array) -> Array:
    """Return the hyperbolic tangent activation for each value in an array."""
    return np.tanh(values)


def tanh_derivative(values: Array) -> Array:
    """Return the derivative of tanh for each pre-activation value."""
    activated = tanh(values)
    return 1.0 - activated**2


def relu(values: Array) -> Array:
    """Return the rectified linear activation for each value in an array."""
    return np.maximum(0.0, values)


def relu_derivative(values: Array) -> Array:
    """Return the derivative of ReLU for each pre-activation value."""
    return (values > 0.0).astype(float)


def linear(values: Array) -> Array:
    """Return values unchanged for regression-style output layers."""
    return values


def linear_derivative(values: Array) -> Array:
    """Return the derivative of the linear activation."""
    return np.ones_like(values)


def mean_squared_error(predictions: Array, targets: Array) -> float:
    """Return the average squared difference between predictions and targets."""
    return float(np.mean((predictions - targets) ** 2))


def mean_squared_error_derivative(predictions: Array, targets: Array) -> Array:
    """Return the derivative of mean squared error with respect to predictions."""
    return 2.0 * (predictions - targets) / predictions.shape[0]


def train_test_dataframe(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> tuple[Array, Array]:
    """Split a dataframe into numeric feature and target arrays."""
    targets = frame.loc[:, list(target_columns)].to_numpy(dtype=float)
    features = frame.drop(columns=list(target_columns)).to_numpy(dtype=float)
    return features, targets


ACTIVATIONS: dict[str, tuple[Callable[[Array], Array], Callable[[Array], Array]]] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "linear": (linear, linear_derivative),
}


@dataclass
class DenseLayer:
    """Represent one fully connected neural-network layer."""

    input_size: int
    output_size: int
    activation: str = "relu"
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        """Initialize weights, biases, and cached arrays for backpropagation."""
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")

        limit = np.sqrt(2.0 / self.input_size)
        self.weights = self.rng.normal(0.0, limit, size=(self.input_size, self.output_size))
        self.biases = np.zeros((1, self.output_size))
        self.inputs = np.empty((0, self.input_size))
        self.z_values = np.empty((0, self.output_size))

    def forward(self, inputs: Array) -> Array:
        """Calculate and return this layer's output for a batch of inputs."""
        self.inputs = inputs
        self.z_values = inputs @ self.weights + self.biases
        activate, _ = ACTIVATIONS[self.activation]
        return activate(self.z_values)

    def backward(self, output_gradient: Array, learning_rate: float) -> Array:
        """Update parameters from an output gradient and return the input gradient."""
        _, activation_derivative = ACTIVATIONS[self.activation]
        z_gradient = output_gradient * activation_derivative(self.z_values)

        weight_gradient = self.inputs.T @ z_gradient
        bias_gradient = np.sum(z_gradient, axis=0, keepdims=True)
        input_gradient = z_gradient @ self.weights.T

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient


@dataclass
class NeuralNetwork:
    """Train and run a configurable feed-forward neural network."""

    layer_sizes: Sequence[int]
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Build dense layers from the configured layer sizes and activations."""
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must include at least input and output sizes")

        rng = np.random.default_rng(self.seed)
        self.layers: list[DenseLayer] = []
        for index, (input_size, output_size) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            activation = (
                self.output_activation
                if index == len(self.layer_sizes) - 2
                else self.hidden_activation
            )
            self.layers.append(DenseLayer(input_size, output_size, activation, rng))

    def predict_array(self, features: Array) -> Array:
        """Return predictions for a NumPy feature array."""
        outputs = features
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return predictions for all feature columns in a dataframe."""
        predictions = self.predict_array(frame.to_numpy(dtype=float))
        columns = [f"prediction_{index}" for index in range(predictions.shape[1])]
        return pd.DataFrame(predictions, columns=columns, index=frame.index)

    def fit_array(
        self,
        features: Array,
        targets: Array,
        epochs: int = 5000,
        learning_rate: float = 0.1,
        log_every: int = 500,
    ) -> list[float]:
        """Train on NumPy arrays and return the loss history."""
        history: list[float] = []
        for epoch in range(1, epochs + 1):
            predictions = self.predict_array(features)
            loss = mean_squared_error(predictions, targets)
            history.append(loss)

            gradient = mean_squared_error_derivative(predictions, targets)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate)

            if log_every and (epoch == 1 or epoch % log_every == 0):
                print(f"epoch={epoch:5d} loss={loss:.6f}")

        return history

    def fit_dataframe(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        epochs: int = 5000,
        learning_rate: float = 0.1,
        log_every: int = 500,
    ) -> list[float]:
        """Train from a pandas dataframe and return the loss history."""
        features, targets = train_test_dataframe(frame, target_columns)
        return self.fit_array(features, targets, epochs, learning_rate, log_every)
