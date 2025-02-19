import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

color_ls = [
    "#3cb44b",
    "#e6194b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffd700",
    "#ff4500",
    "#da70d6",
    "#32cd32",
    "#4682b4",
    "#d2691e",
    "#ff1493",
    "#7fff00",
    "#00ced1",
    "#ff00ff",
    "#1e90ff",
    "#b22222",
    "#adff2f",
    "#a020f0",
    "#00ff00",
    "#4682b4",
    "#f0e68c",
    "#b03060",
    "#0000cd",
    "#808000",
    "#8b4513",
    "#ee82ee",
    "#ff8c00",
    "#556b2f",
    "#00bfff",
    "#dc143c",
    "#fa8072",
    "#ff00ff",
    "#32cd32",
    "#ffff00",
    "#daa520",
    "#1e90ff",
    "#2f4f4f",
    "#ff0000",
    "#483d8b",
    "#afeeee",
    "#dda0dd",
    "#8b0000",
    "#9acd32",
    "#8fbc8f",
    "#98fb98",
    "#f4a460",
    "#228b22",
    "#a9a9a9",
    "#ff1493",
    "#ffe4c4",
    "#00008b",
    "#20b2aa",
    "#800080",
    "#00ffff",
    "#7b68ee",
    "#ffb6c1",
]


def generate_umap(
    adata,
    column_name="clusters",
    filename=None,
    show_text=False,
    bbox_to_anchor=(1.3, 0.5),
):
    """
    Generates a UMAP plot from the given AnnData object and saves it as an image file.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    column_name : str, optional
        The column name in `adata.obs` to use for coloring the UMAP plot. Default is "clusters".
    filename : str, optional
        The name of the file to save the plot. If None, a default name is generated. Default is None.
    show_text : bool, optional
        If True, display the cluster names at the centroid of each cluster. Default is False.
    bbox_to_anchor : tuple, optional
        The bounding box anchor for the legend. Default is (1.3, 0.5).
    Returns:
    --------
    None
    """
    row_numbers_by_category = {}
    for idx, value in enumerate(adata.obs[column_name]):
        row_numbers_by_category.setdefault(value, []).append(idx)

    row_numbers_by_category = dict(
        sorted(row_numbers_by_category.items(), key=lambda item: item[0])
    )

    keys = list(row_numbers_by_category.keys())
    new_row_numbers_by_category = {}
    for i, j in enumerate(keys):
        new_row_numbers_by_category[f"{i}: {j}"] = row_numbers_by_category[j]
    row_numbers_by_category = new_row_numbers_by_category

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    plt.figure(figsize=(15, 10), dpi=300)

    color_count = 0

    h_ls = []
    for cluster, cell_index in row_numbers_by_category.items():
        plt.scatter(
            umap1[cell_index],
            umap2[cell_index],
            s=4,
            color=color_ls[color_count % len(color_ls)],
            label=cluster,
        )
        h_ls.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                color=color_ls[color_count % len(color_ls)],
                linestyle="",
            )
        )
        if show_text:
            # Get the coordinates of the points in this cluster
            x_coords = umap1[cell_index]
            y_coords = umap2[cell_index]
            # Compute the centroid – you can use np.mean or np.median
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            # Add text at the centroid
            plt.text(
                x_center,
                y_center,
                cluster,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

        color_count += 1

    _, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=h_ls,
        labels=legend_labels,
        bbox_to_anchor=bbox_to_anchor,
        loc="right",
        fontsize="large",
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename is None:
        filename = f"high_res_figure_{column_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def generate_umap_snc(
    adata,
    column_name="clusters",
    filename=None,
    show_text=False,
    bbox_to_anchor=(1.3, 0.5),
):
    """
    Generate a UMAP plot for single-cell data with SnC highlighted.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    column_name : str, optional
        The column name in `adata.obs` to use for coloring the clusters (default is "clusters").
    filename : str, optional
        The filename to save the plot. If None, a default filename is used (default is None).
    show_text : bool, optional
        Whether to display the cluster names as text on the plot (default is False).
    bbox_to_anchor : tuple, optional
        The bounding box anchor for the legend (default is (1.3, 0.5)).
    Returns:
    --------
    None
    """
    row_numbers_by_category = {}
    for idx, value in enumerate(adata.obs[column_name]):
        row_numbers_by_category.setdefault(value, []).append(idx)

    row_numbers_by_category = dict(
        sorted(row_numbers_by_category.items(), key=lambda item: item[0])
    )

    keys = list(row_numbers_by_category.keys())
    for i, j in enumerate(keys):
        row_numbers_by_category[f"{i}: {j}"] = row_numbers_by_category[j]
        row_numbers_by_category.pop(j)

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    plt.figure(figsize=(15, 10), dpi=300)

    color_count = 0

    h_ls = []
    for cluster, cell_index in row_numbers_by_category.items():
        if "SnC" in cluster:
            plt.scatter(
                umap1[cell_index], umap2[cell_index], s=5, color="red", label=cluster
            )
            h_ls.append(
                plt.Line2D(
                    [0], [0], marker="o", markersize=10, color="red", linestyle=""
                )
            )
        else:
            plt.scatter(
                umap1[cell_index],
                umap2[cell_index],
                s=4,
                color="#cccccb",
                label=cluster,
            )
            h_ls.append(
                plt.Line2D(
                    [0], [0], marker="o", markersize=10, color="#cccccb", linestyle=""
                )
            )
            color_count += 1

        if show_text:
            # Get the coordinates of the points in this cluster
            x_coords = umap1[cell_index]
            y_coords = umap2[cell_index]
            # Compute the centroid – you can use np.mean or np.median
            x_center = x_coords.mean()
            y_center = y_coords.mean()
            # Add text at the centroid
            plt.text(
                x_center,
                y_center,
                cluster,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

        color_count += 1

    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=h_ls,
        labels=legend_labels,
        bbox_to_anchor=bbox_to_anchor,
        loc="right",
        fontsize="large",
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename is None:
        filename = f"high_res_figure_{column_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def generate_heatmap(adata, 
                     gene_order,
                     cluster_order,
                     column_name="clusters", 
                     filename=None):
    
    # If adata.X is sparse, convert it to a dense array.
    if hasattr(adata.X, "toarray"):
        expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var.index)
    else:
        expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var.index)

    # Add the cluster information from adata.obs.
    expr_df[column_name] = adata.obs[column_name].values

    # Compute the average expression per cluster.
    # This results in a DataFrame with clusters as rows and genes as columns.
    cluster_avg = expr_df.groupby(column_name).mean()

    # Now, create a heatmap DataFrame:
    # We want genes as rows and clusters as columns.
    # First, subset the columns (genes) to those in gene_order.
    # Then, transpose the DataFrame.
    heatmap_df = cluster_avg.loc[:, gene_order].T

    # Reorder the columns (clusters) as desired.
    heatmap_df = heatmap_df[cluster_order]

    # Plot the heatmap.
    plt.figure(figsize=(12, len(gene_order) * 0.15),dpi=300)
    
    # low color, medium and high color seperately
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4F99A9','white','#D65A79'])

    heatmap_df_norm = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)

    sns.heatmap(heatmap_df_norm, cmap=cmap,yticklabels=False)
    # sns.heatmap(heatmap_df, cmap=cmap)


    plt.xlabel("Clusters")
    plt.ylabel("Genes")
    plt.title("Average Expression of Selected Genes by Cluster")
    plt.tight_layout()
    plt.savefig('high_res_figure_heatmap.png', dpi=300)

    plt.show()





def bar_plot_condition(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Status"])
        .size()
        .reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Status", fill_value=0
    )
    print(pivot_table)
    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    new_order = ["cell type", "Healthy", "IPF"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different condition",
        color=["#1f77b4", "#d62728", "#ff9896"],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.19, 1), loc="upper right")
    plt.show()


def bar_plot_location(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Area"]).size().reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Area", fill_value=0
    )
    print(pivot_table)

    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    new_order = ["cell type", "Upper Lobe", "Lower Lobe", "Parenchyma"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different location",
        color=[
            "#d62728",
            "#ff9896",
            "#1f77b4",
        ],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    plt.show()


def bar_plot_age(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Age_Status"])
        .size()
        .reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Age_Status", fill_value=0
    )
    print(pivot_table)

    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    new_order = ["cell type", "Old", "Young"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different age",
        color=["#7a0000", "#d62728", "#ff9896", "#1f77b4", "#daece5"],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    plt.show()


def create_summary_table(adata):
    """
    Creates a summary table of cell counts for each cluster (CT) 
    across various conditions: IPF vs Healthy, UL vs LL vs Parenchyma,
    and Young vs Old. Also calculates the fraction (percentage) of SnC cells.
    
    Assumes:
    - 'clusters' column has the cell type / cluster names.
    - 'Injury' column has 'IPF' or 'Healthy'.
    - 'area' column has 'Upper Lobe', 'Lower Lobe', or 'Parenchyma'.
    - 'Age_Status' column has 'young' or 'old'.
    - 'is_sen' column has 'SnC' or 'normal' (indicating senescent or not).
    """
    # Work off obs DataFrame
    df = adata.obs.copy()

    # All unique clusters
    clusters = df['clusters'].unique().tolist()

    # Define columns in final summary
    columns = [
        'CT',                 # cluster name
        '# cell',             # total cells in that cluster
        '# IPF non-SnC', '# IPF SnC', 'IPF SnC%',
        '# Healthy non-SnC', '# Healthy SnC', 'Healthy SnC%',
        '# UL non-SnC', '# UL SnC', 'UL SnC%',
        '# LL non-SnC', '# LL SnC', 'LL SnC%',
        '# Parenchyma non-SnC', '# Parenchyma SnC', 'Parenchyma SnC%',
        '# Young non-SnC', '# Young SnC', 'Young SnC%',
        '# Old non-SnC', '# Old SnC', 'Old SnC%'
    ]
    
    # Prepare empty summary DataFrame
    summary_df = pd.DataFrame(columns=columns)

    # Helper function to count SnC vs non-SnC for a subset
    def group_counts(sub_df):
        # Number of non-SnC
        n_nonSnC = (sub_df['is_sen'] == 'normal').sum()
        # Number of SnC
        n_SnC = (sub_df['is_sen'] == 'SnC').sum()
        # Percentage of SnC
        total = n_nonSnC + n_SnC
        pct_SnC = (n_SnC / total * 100) if total > 0 else 0.0
        return n_nonSnC, n_SnC, pct_SnC

    for ct in clusters:
        # Subset the DataFrame to the current cluster
        df_ct = df[df['clusters'] == ct]
        
        # Total cells in this cluster
        total_cells = len(df_ct)

        # Count IPF vs Healthy
        df_ct_ipf = df_ct[df_ct['Status'] == 'IPF']
        ipf_nonSnC, ipf_SnC, ipf_pct = group_counts(df_ct_ipf)

        df_ct_healthy = df_ct[df_ct['Status'] == 'Healthy']
        healthy_nonSnC, healthy_SnC, healthy_pct = group_counts(df_ct_healthy)

        # Count Upper/Lower Lobe vs Parenchyma
        df_ct_ul = df_ct[df_ct['Area'] == 'Upper Lobe']
        ul_nonSnC, ul_SnC, ul_pct = group_counts(df_ct_ul)

        df_ct_ll = df_ct[df_ct['Area'] == 'Lower Lobe']
        ll_nonSnC, ll_SnC, ll_pct = group_counts(df_ct_ll)

        df_ct_parenchyma = df_ct[df_ct['Area'] == 'Parenchyma']
        para_nonSnC, para_SnC, para_pct = group_counts(df_ct_parenchyma)

        # Count Young vs Old
        df_ct_young = df_ct[df_ct['Age_Status'] == 'Young']
        young_nonSnC, young_SnC, young_pct = group_counts(df_ct_young)

        df_ct_old = df_ct[df_ct['Age_Status'] == 'Old']
        old_nonSnC, old_SnC, old_pct = group_counts(df_ct_old)

        # Build a row of results
        row = {
            'CT': ct,
            '# cell': total_cells,
            '# IPF non-SnC': ipf_nonSnC, 
            '# IPF SnC': ipf_SnC, 
            'IPF SnC%': ipf_pct,
            
            '# Healthy non-SnC': healthy_nonSnC, 
            '# Healthy SnC': healthy_SnC, 
            'Healthy SnC%': healthy_pct,
            
            '# UL non-SnC': ul_nonSnC, 
            '# UL SnC': ul_SnC, 
            'UL SnC%': ul_pct,
            
            '# LL non-SnC': ll_nonSnC, 
            '# LL SnC': ll_SnC, 
            'LL SnC%': ll_pct,
            
            '# Parenchyma non-SnC': para_nonSnC, 
            '# Parenchyma SnC': para_SnC, 
            'Parenchyma SnC%': para_pct,
            
            '# Young non-SnC': young_nonSnC, 
            '# Young SnC': young_SnC, 
            'Young SnC%': young_pct,
            
            '# Old non-SnC': old_nonSnC, 
            '# Old SnC': old_SnC, 
            'Old SnC%': old_pct
        }

        # Concatenate this new row to summary_df
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([row])],
            ignore_index=True
        )

    return summary_df
