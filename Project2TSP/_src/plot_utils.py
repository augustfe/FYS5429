import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from jax.typing import ArrayLike
import numpy as onp
from pathlib import Path

rcParams = {
    "figure.dpi": 300,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "figure.titlesize": 25,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
}
save_dir = Path(__file__).parent / "figures"
save_dir.mkdir(exist_ok=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from a DataFrame for plotting

    Args:
        df (pd.DataFrame): DataFrame to remove outliers from

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Calculate the IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers from the DataFrame
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    return df_no_outliers


def boxplot_on_ax(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    """Plot a boxplot on a given axis using seaborn

    Args:
        df (pd.DataFrame): DataFrame to plot
        ax (plt.Axes): Axis to plot on
        title (str): Axis title for the plot
    """
    sns.boxplot(data=remove_outliers(df), ax=ax, palette="Set3")
    ax.set_title(title)
    # ax.set_xlabel("Variable")
    ax.set_ylabel("Nodes/Edges")


def single_boxplot(df: pd.DataFrame, title: str, savename: str = None) -> None:
    """Plot a single boxplot

    Args:
        df (pd.DataFrame): DataFrame to plot
        title (str): Title of the plot
    """
    mpl.rcParams.update(rcParams)
    sns.set_style("whitegrid", rc=rcParams)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=remove_outliers(df), ax=ax, palette="Set3")
    ax.set_title(title)
    ax.set_ylabel("Nodes/Edges")

    plt.tight_layout()
    if savename is not None:
        plt.savefig(save_dir / f"{savename}.pdf")
    else:
        plt.show()


def seen_vs_unseen_mvc_graphs(
    dfs: list[pd.DataFrame],
    titles: list[str] = None,
    suptitle: str = "",
    savename: str = None,
) -> None:
    """Compare the final result of the seen and unseen MVC graphs

    Args:
        df_train (pd.DataFrame): DataFrame from the training set
        df_val (pd.DataFrame): DataFrame from the validation set
    """
    num_dfs = len(dfs)
    assert len(titles) == num_dfs or titles is None

    mpl.rcParams.update(rcParams)
    sns.set_style("whitegrid", rc=rcParams)
    fig, axes = plt.subplots(1, num_dfs, figsize=(12, 6), sharey=True)

    for i, df in enumerate(dfs):
        sns.boxplot(data=remove_outliers(df), ax=axes[i], palette="Set3")
        axes[i].set_title(titles[i] if titles is not None else f"Set {i+1}")
        axes[i].set_ylabel("Nodes/Edges")

    fig.suptitle(suptitle)
    plt.tight_layout()

    if savename is not None:
        plt.savefig(save_dir / f"{savename}.pdf")
    else:
        plt.show()


def mvc_plot_graph(
    nx_graph: nx.Graph, pos: dict, covering: ArrayLike, savename: str = None
) -> None:
    """Plot the minimum vertex cover on a graph

    Args:
        nx_graph (nx.Graph): The graph to plot
        pos (dict): Position of the nodes in the graph
        covering (ArrayLike): The proposed minimum vertex cover
    """
    mpl.rcParams.update(rcParams)

    node_list = onp.asarray(list(nx_graph.nodes))
    covering = covering.flatten().astype(bool)

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(nx_graph, pos, ax=ax)
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=node_list[covering],
        node_color="orange",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=node_list[~covering],
        node_color="lightblue",
        ax=ax,
    )

    # Remove border around plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add legend
    orange_patch = mpl.patches.Patch(color="orange", label="Covered")
    blue_patch = mpl.patches.Patch(color="lightblue", label="Not covered")
    ax.legend(handles=[orange_patch, blue_patch])

    ax.set_title(f"Minimum Vertex Cover of size {covering.sum()}")

    plt.tight_layout()

    if savename is not None:
        plt.savefig(save_dir / f"{savename}.pdf")
    else:
        plt.show()
