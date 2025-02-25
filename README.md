# Deep-learning framework for cell-type-specific SnCs And SnGs (DeepSAS)
In this branch we show how to do subsampling for your anndata object, run DeepSAS, and generate tables/results for each subsampled object


### Usage

1. susampling your anndata object using script subsampling.py. Once you created subsamples you can use 'run_experiments.sh' to loop through each subsample and run DeepSAS

```bash

for i in {1..28}
do
    # Define the file path dynamically
    FILE_PATH="/path/to/your/subsamples/subsample_${i}.h5ad"####the subsampling.py will create subsamples with number depend on the total cell size eg. subsample_1.h5ad, subsample_2.h5ad, ...
    
    # Define the experiment name dynamically
    EXP_NAME="Data_${i}_013025"
    
    # Run your command (replace `your_command` with the actual command you want to execute)
    uv run python -u deepsas_v1.py --input_data_count "$FILE_PATH" --output_dir ./outputs --exp_name "$FILE_PATH" --device_index 4 --timestamp '013025' --retrain

    # Print (optional, for debugging)
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done
```

To generate 3 table of SnGs run 'Generate_table_sampling.sh' :
```bash
for i in {1..28}
do
    # Define the file path dynamically
    FILE_PATH="/path/to/your/subsamples/subsample_${i}.h5ad"
    
    # Define the experiment name dynamically
    EXP_NAME="Data_${i}_022425"
    
    # Run your command (replace `your_command` with the actual command you want to execute)
    uv run python -u generate_3tables.py --input_data_count "$FILE_PATH" --output_dir ./outputs --exp_name "$FILE_PATH" --device_index 4 --timestamp '022425' --retrain

    # Print (optional, for debugging)
    echo "Processing file: $FILE_PATH with experiment name: $EXP_NAME"
done
```
This code will generate a folder for each subsample and each folder contail sncG and sncC results
The `generate_3tables.py` needs two inputs: .h5ad file and DeepSAS output. you can use `--input_data_count` and `--exp_name` to load your .h5ad and DeepSAS output respectively.

For the visualization and downstream analysis of SnCs and SnGs, please follow the tutorial in the [`tutorial.ipynb`](./tutorial.ipynb).


#### Important Arguments

- `--output_dir`: Directory to store output files and results.
- `--exp_name`: Descriptive name for the experiment.
- `--device_index`: GPU device index if CUDA is available.

## Modules and Functions

- **utils.py**: Contains utility functions for data loading, preprocessing, and transformations.
- **plot_figure.py**: Provides functions for plotting umap, heatmaps, and other visualizations.
- **model_AE.py**, **model_GAT.py**, **model_Sencell.py**: Include model definitions and training procedures for autoencoders, Graph Attention Networks, and senescent cell identification models.
- **deepsas_v1.py**: The main executable script orchestrating data loading, model training, and evaluation processes.
