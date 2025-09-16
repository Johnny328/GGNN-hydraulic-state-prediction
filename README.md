# GNN Hydraulic Prediction Code

This repository contains the code for the paper "Graph Neural Networks for Hydraulic Predictions in Water Distribution Networks: Incorporating Sensor Data and Physical Constraints". The project implements a Gated Graph Neural Network (GGNN) model for predicting hydraulic parameters (e.g., pressure) in water distribution networks, integrating sensor data and physical constraints.

## Project Structure

- **Main Scripts**:
  - [`main.py`](main.py): Main training script for the GGNN model with configurable parameters.
  - [`GGNN_regression.py`](GGNN_regression.py): GGNN model definition for regression tasks.
  - [`utils_regression.py`](utils_regression.py): Training utilities and data processing functions.

- **Data Generation and Processing**:
  - [`data_generate.py`](data_generate.py): Generates hydraulic simulation data from water network models.
  - [`adj_gen_distance.py`](adj_gen_distance.py): Generates adjacency matrices for graph-based modeling.

- **Utilities and Metrics**:
  - [`metrics.py`](metrics.py): Evaluation metrics, including physical constraint calculations.
  - [`errors.py`](errors.py): Error computation utilities.
  - [`network_config.json`](network_config.json): Configuration file for network settings (e.g., sensor locations).

## Installation

1. Ensure Python 3.7+ is installed.
2. Install required dependencies:
   ```bash
   pip install torch wntr numpy pandas networkx matplotlib jupyter
   ```
   - PyTorch should support CUDA for GPU acceleration if available.

3. Place benchmark network files (e.g., `.inp` files like `ASnet.inp`) in the appropriate directories as specified in [`network_config.json`](network_config.json).

## Data Preparation

1. **Generate Hydraulic Data**:
   Run the data generation script to simulate and extract hydraulic data:
   ```bash
   python data_generate.py
   ```
   This creates `dataset_all_new1.npz` with simulated head values and masks for sensors and predictions.

2. **Generate Adjacency Matrices**:
   Compute adjacency matrices for the graph representation:
   ```bash
   python adj_gen_distance.py
   ```
   This generates files like `adj_matrices_link_weight_binary_directed.npy` and `adj_matrices_link_distance_binary_undirected.npy` based on network topology and flow directions.

## Running the Model

Train the GGNN model using the main script. Example command for ASnet network with physical constraints:

```bash
python main.py --w 1.1 --init --distance_type 'binary' --distance_direction 'undirected' --epochs 40 --network_name 'ASnet' --loss_type 1 --hidden_size 128 --standard_type 1 --propag_steps 5 --duration 24 --output_size 12 --weight_direction 'directed' --weight_type 'binary'
```

### Key Parameters

- `--w`: Weight for physical constraint loss (e.g., 1.1).
- `--init`: Enable initialization with sensor-based features.
- `--distance_type` / `--distance_direction`: Distance adjacency matrix settings (e.g., 'binary', 'undirected').
- `--epochs`: Number of training epochs (e.g., 40).
- `--network_name`: Network name (e.g., 'ASnet').
- `--loss_type`: Loss function type (0: basic, 1: with constraints).
- `--hidden_size`: Hidden layer size (e.g., 128).
- `--standard_type`: Normalization type (1: global z-score).
- `--propag_steps`: Number of GGNN propagation steps (e.g., 5).
- `--duration`: Input sequence length in steps (e.g., 24).
- `--output_size`: Prediction horizon in steps (e.g., 12).
- `--weight_type` / `--weight_direction`: Weight adjacency matrix settings (e.g., 'binary', 'directed').

Trained models and statistics are saved in subfolders under the network directory (e.g., `ASnet/model/`).

## Results and Evaluation

- Metrics (MAE, RMSE, MAPE) are computed during training and validation.
- Physical constraint violations are penalized in the loss function.
- For visualization, use Jupyter notebooks (e.g., from the parent [`paper`](paper ) folder) to plot results.

## Notes

- This code is for research purposes; ensure compliance with data usage policies.
- GPU acceleration is recommended for large networks.
- Refer to the paper for detailed methodology and results.

For questions or issues, contact the authors.
