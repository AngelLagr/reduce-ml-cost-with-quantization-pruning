# Weight Usage Analyzer Demo

This project demonstrates the impact of model design choices on both energy consumption and economic cost.
It analyzes the weight importance within a neural network, estimates the total FLOPs required for inference, and explores how quantization and pruning affect both model performance and resource efficiency.

## Project Structure

- **core/**: Contains the main functionality for analyzing weight usage.
  - `weightusageanalyzer.py`: Functions to compute weight importance for TensorFlow/Keras and PyTorch models that come from my repo : https://github.com/AngelLagr/weight-usage-analyser

- **notebooks/**: Jupyter notebook for demonstration purposes.
  - `demo.ipynb`: A notebook that loads the Breast Cancer dataset, trains a simple model, and visualizes weight importance.

- **models/**: Defines the neural network architecture.
  - `simple_model.py`: A simple neural network model with one hidden layer.

- **data/**: Functions for loading and preprocessing datasets.
  - `load_wine_dataset.py`: Functions to load and preprocess the keras wine dataset.

- **quantization/**: Contains functions for model quantization.
  - `quantize.py`: Functions to analyze the impact of quantization on model performance and resource usage.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd weight-usage-analyzer-demo
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter notebook:
   ```
   jupyter notebook notebooks/demo.ipynb
   ```

2. Follow the instructions in the notebook to load the Breast Cancer dataset, train the model, and visualize the weight importance.

## Components Overview

- **Weight Usage Analyzer**: The core functionality for analyzing the importance of weights in neural networks.
- **Simple Model**: A basic neural network architecture to demonstrate the weight usage analysis.
- **Data Loading**: Functions to handle the Breast Cancer dataset, including normalization and splitting.
- **Quantization**: Tools to reduce model size and analyze the impact on performance and energy efficiency.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
