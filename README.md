# Neural Network With Pandas and NumPy

This project is a small neural network built from scratch with NumPy and pandas.
It does not use ML frameworks.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```powershell
python train_xor.py
```

## How To Operate

1. Put your data into a pandas dataframe.

   Each row should be one training example. Feature columns are the inputs, and
   one or more target columns are the expected outputs.

2. Create the network.

   ```python
   from neural_net import NeuralNetwork

   network = NeuralNetwork(
       layer_sizes=[2, 6, 4, 1],
       hidden_activation="sigmoid",
       output_activation="sigmoid",
       seed=42,
   )
   ```

3. Train the network.

   ```python
   history = network.fit_dataframe(
       frame=data,
       target_columns=["target"],
       epochs=10000,
       learning_rate=0.8,
       log_every=1000,
   )
   ```

   `history` is a list of loss values, one value per epoch.

4. Make predictions.

   ```python
   predictions = network.predict_dataframe(data[["x1", "x2"]])
   print(predictions.round(4))
   ```

## Custom Data Example

```python
import pandas as pd

from neural_net import NeuralNetwork
from train_xor import read_data_csv

data = read_data_csv("my_data.csv")

network = NeuralNetwork(
    layer_sizes=[3, 8, 1],
    hidden_activation="relu",
    output_activation="linear",
    seed=1,
)

network.fit_dataframe(
    frame=data,
    target_columns=["target"],
    epochs=2000,
    learning_rate=0.01,
)

feature_columns = ["feature_a", "feature_b", "feature_c"]
predictions = network.predict_dataframe(data[feature_columns])
print(predictions)
```

For custom data, make sure every feature and target column is numeric. Convert
categories, dates, and text into numeric values before training.

## Tune The Network

Edit `layer_sizes` in `train_xor.py`:

```python
network = NeuralNetwork(layer_sizes=[2, 6, 4, 1])
```

The first number is the input feature count, the last number is the output
count, and the middle numbers are hidden layers. For example:

- `[2, 3, 1]`: one hidden layer with 3 nodes
- `[2, 8, 4, 1]`: two hidden layers with 8 and 4 nodes
- `[2, 16, 8, 4, 1]`: three hidden layers

You can also tune:

- `hidden_activation`: `"relu"`, `"sigmoid"`, or `"tanh"`
- `output_activation`: `"sigmoid"`, `"tanh"`, or `"linear"`
- `epochs`
- `learning_rate`

## Files

- `neural_net.py`: reusable neural network code
- `train_xor.py`: pandas-based XOR training example
- `requirements.txt`: dependencies
