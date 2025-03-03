# DeepSAS

DeepSAS (**Deep**-learning framework for cell-type-specific **S**nCs **A**nd **S**nGs) is a computational framework designed to identify senescent cells and senescence-associated genes from single-cell RNA sequencing data.

## Overview

Cellular senescence is a state of permanent cell cycle arrest that plays important roles in development, tissue homeostasis, aging, and disease. Identifying senescent cells in heterogeneous tissues is challenging due to the lack of universal markers. DeepSAS leverages graph neural networks and contrastive learning to identify senescent cells and their associated gene signatures from single-cell RNA sequencing data.

## Features

- Identifies senescent cells in heterogeneous tissues
- Discovers senescence-associated genes specific to each cell type
- Leverages both gene expression and cell-cell interactions
- Robust to batch effects and technical variations
- Works across different cell types and senescence induction methods

## Installation

This project is developed and tested on Linux and macOS environments.


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

## Quick Start

To run DeepSAS on example data:

```bash
uv run python -u deepsas_v1.py --exp_name example --device_index 0 --retrain > ./example.log
```

To run in background with logging:

```bash
nohup uv run python -u deepsas_v1.py --exp_name your_experiment --device_index 0 --retrain > ./your_experiment.log 2>&1 &
```

To generate 3 table of SnGs:

```bash
uv run python -u generate_3tables.py --output_dir ./outputs --exp_name example --device_index 0
```
The `generate_3tables.py` needs two inputs: .h5ad file and DeepSAS output. you can use `--input_data_count` and `--exp_name` to load your .h5ad and DeepSAS output respectively.

For the visualization and downstream analysis of SnCs and SnGs, please follow the tutorial in the [`tutorial.ipynb`](./tutorial.ipynb).


## Input Data Format

DeepSAS works with h5ad format (AnnData objects from Scanpy). The input data should include:
- Gene expression matrix (cells × genes)
- Cell type annotations in `adata.obs['clusters']` or another specified column

## Parameters

DeepSAS accepts the following parameters:

### Input/Output Parameters
- `--input_data_count`: Path to input data (h5ad format)
- `--output_dir`: Base output directory
- `--exp_name`: Experiment name (used for output directory naming)
- `--device_index`: CUDA device index to use
- `--retrain`: Whether to retrain models or use saved ones
- `--timestamp`: Timestamp for the experiment (optional)

### Model Configuration
- `--seed`: Random seed for reproducibility
- `--n_genes`: Number of genes to use (3000, 8000 or full)
- `--ccc`: Cell-cell edge type: type1 (binary), type2 (continuous), type3 (none)
- `--gene_set`: Gene set to use (senmayo, fridman, etc.)
- `--emb_size`: Embedding dimension size

### Training Parameters
- `--gat_epoch`: Number of epochs to train the GAT model
- `--sencell_num`: Number of senescent cells to use
- `--sengene_num`: Number of senescence-associated genes to use
- `--sencell_epoch`: Number of epochs to train the Sencell model
- `--cell_optim_epoch`: Number of epochs for cell embedding optimization
- `--learning_rate`: Initial learning rate
- `--batch_id`: Batch ID for processing

## Output Files

DeepSAS generates the following output files in the specified output directory:

- `{exp_name}_new_data.h5ad`: Processed AnnData object
- `{exp_name}_graphnx.data`: NetworkX graph representation
- `{exp_name}_graphpyg.data`: PyTorch Geometric graph representation
- `{exp_name}_GAT.pt`: Trained GAT model
- `{exp_name}_sencellgene-epoch{epoch}.data`: Senescent cells and genes at each epoch


## Citation

If you use DeepSAS in your research, please cite:

```
@article{
   xxxx
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
