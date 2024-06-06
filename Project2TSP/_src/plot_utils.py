import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


rcParams = {
    "figure.dpi": 300,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "figure.titlesize": 25,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
}


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


def seen_vs_unseen_mvc_graphs(df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
    """Compare the final result of the seen and unseen MVC graphs

    Args:
        df_train (pd.DataFrame): DataFrame from the training set
        df_val (pd.DataFrame): DataFrame from the validation set
    """
    mpl.rcParams.update(rcParams)
    sns.set_style("whitegrid", rc=rcParams)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    boxplot_on_ax(df_train, axes[0], "Test Set")
    boxplot_on_ax(df_val, axes[1], "Validation Set")

    fig.suptitle("Results from random graphs ($p=0.05$)")
    plt.tight_layout()
    plt.show()
