# Deep-learning framework for cell-type-specific SnCs And SnGs (DeepSAS)

## Overview

This Python-based framework facilitates the advanced analysis of senescent cells using a variety of deep learning techniques, including graph neural networks and dimensionality reduction via autoencoders. Designed for scalability and robustness, the framework supports various datasets, integrates with modern machine learning tools like PyTorch and scanpy, and offers extensive capabilities for visualizing data.


## Key Features

- **Data Preprocessing**: Standardize and preprocess raw data for further analysis.
- **Dimensionality Reduction**: Use autoencoders to reduce data dimensions effectively, capturing essential features.
- **Graph Neural Networks**: Leverage Graph Attention Networks (GAT) to handle complex data structures typical in biological data.
- **Senescent Cell Identification**: Specialized models to detect and analyze senescent cells and their genetic markers.
- **Experiment Tracking**: Integration with Weights & Biases for real-time tracking of model performance and metrics.
- **Visualization**: Utilize seaborn and matplotlib for detailed visual representations of data insights.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or later installed. This project is developed and tested on Linux and macOS environments.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chthub/sencell.git
   cd sencell/
   git checkout deepsas-v1
   ```

2. **Set Up a uv Environment** (recommended):
   We recommend to use uv for the environment mangement. Check this [link](https://docs.astral.sh/uv/) to install uv.

   ```bash
   uv venv --python 3.8.20
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   
   ```bash
   uv pip install numpy seaborn matplotlib pandas tabulate linetimer scikit-learn ipykernel 'scanpy[leiden]' tqdm 
   ```
   For [Pytorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),  ensure you select the CUDA version that best suits your system. Below is an example from our test environment:
   ```bash
   uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   uv pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 
   ```

4. **Environment Variables**:
   Set necessary environment variables, if any (e.g., PYTHONHASHSEED for reproducibility).

### Usage

Run the main script using the following command with required flags:

```bash
nohup uv run python -u deepsas_v1.py --output_dir ./outputs --exp_name example --device_index 0 --retrain > ./example.log 2>&1 &
```
You can also specify your input for your .h5ad file using the argument --input_data_count

For the visualization of SnCs and SnGs, please follow the tutorial in the [`plot_snc.ipynb`](./plot_snc.ipynb) and [`plot_nsnc.ipynb`](./plot_nsnc.ipynb). 

To generate 3 table of SnGs:
```bash
uv run python -u generate_3tables.py --output_dir ./outputs --exp_name example --device_index 0
```


#### Important Arguments

- `--output_dir`: Directory to store output files and results.
- `--exp_name`: Descriptive name for the experiment.
- `--device_index`: GPU device index if CUDA is available.

## Modules and Functions

- **utils.py**: Contains utility functions for data loading, preprocessing, and transformations.
- **plot_figure.py**: Provides functions for plotting umap, heatmaps, and other visualizations.
- **model_AE.py**, **model_GAT.py**, **model_Sencell.py**: Include model definitions and training procedures for autoencoders, Graph Attention Networks, and senescent cell identification models.
- **deepsas_v1.py**: The main executable script orchestrating data loading, model training, and evaluation processes.
