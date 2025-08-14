# style2speed

**style2speed** is a neural network pipeline for analyzing Formula 1 telemetry data. It enables users to:

- Preprocess telemetry laps and extract behavioral parameters
- Train neural network models to classify drivers or regress lap/segment times
- Attribute time or classification differences to specific driver actions
- Visualize telemetry, attribution maps, and model performance

---

## Installation

To install the package:

```bash
pip install style2speed
```

Or, if installing from a local directory or in Google Colab:

```bash
pip install /path/to/style2speed_project.zip
```

---

## Requirements

- Python 3.8+
- torch
- fastf1
- captum
- matplotlib
- numpy
- pandas
- scipy
- seaborn
- scikit-learn
- statsmodels
- tqdm

These dependencies are managed in `setup.py`.

---

## Usage

```python
from style2speed.data import compile_telemetry
from style2speed.models.architectures import TelemetryModelLSTM
from style2speed.models.training import train_model
```

Typical workflow:
1. Load and preprocess telemetry data
2. Assemble a PyTorch dataset and train a model
3. Run inference and calculate parameter attribution
4. Visualize results using the plotting utilities

---

## Project Structure

```
style2speed/
├── data.py                  # Data import, parameter derivation and interpolation
├── stats.py                 # Parameter statistics calculations over shorter circuit segments
├── preprocessing.py         # Data normalization and assembly for training
├── models/architectures.py  # FC, CNN, LSTM, CNN-LSTM PyTorch model definitions
├── models/training.py       # PyTorch model training functions
├── models/inference.py      # PyTorch model inference (predictions and attribution importance)
├── plotting/performance.py  # Plotting functions to visualize model performance
├── plotting/traces.py       # Plotting functions to visualize param or attribution plots on xy plots
├── plotting/traces.py       # Plotting functions to visualize the circuit or param values along the circuit
├── utils.py                 # General helper utilities
├── config.py                # Global variable values
├── __init__.py
setup.py
```

---

## Example Notebooks

You can find sample usage in the following notebooks:
- `demo_miami_2025.ipynb`
- `demo_miami_2025.html` (view only)

---

## License

MIT

---

## Author

Roberta Poceviciute  
GitHub: [rpocevi](https://github.com/rpocevi)