# style2speed/plotting/performance.py

from typing import Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(
    df_metrics: pd.DataFrame,
    task: str,
    filter_by: Optional[dict[str, list]] = None,
    show_loss: Optional[bool] = None,
    show_accuracy: Optional[bool] = None,
    show_mse: Optional[bool] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot training and validation performance curves from model training history.

    Depending on the task, displays loss, accuracy, and/or MSE over epochs
    for one or more trials. If both `best_trial` and `df_best_trial` are provided,
    `best_trial` takes precedence.

    Args:
        df_metrics (pd.DataFrame): Output from `train_model`, containing performance
            metrics (e.g., loss, accuracy, MSE) across trials and epochs.
        task (str): Task type. Must be one of `{'class', 'reg', 'multi'}`.
            Determines which metrics to display.
        filter_by (dict[str, list], optional): Dictionary for filtering rows in
            `df_metrics` (e.g., `{'ModelName': ['TelemetryModelFC']}`).
        show_loss (bool, optional): If `True`, plot training and validation loss curves.
            If `None`, automatically enabled for all tasks.
        show_accuracy (bool, optional): If `True`, plot validation accuracy curves.
            If `None`, automatically enabled for classification and multi-task.
        show_mse (bool, optional): If `True`, plot validation MSE curves.
            If `None`, automatically enabled for regression and multi-task.
        figsize (tuple[int, int], optional): Figure size. Defaults to `(10, 5)`.
        show (bool): Whether to display the figure. Defaults to `True`.
        save_path (str, optional): If provided, saves the plot to this path.

    Raises:
        ValueError: If `task` is not one of `{'class', 'reg', 'multi'}`.
        ValueError: If filtered data is empty.

    Returns:
        None
    """

    def _pad_trials(trials):
        max_len = max(len(t) for t in trials)
        return np.array([
            np.pad(t, (0, max_len - len(t)), constant_values=np.nan)
            for t in trials
        ])
    
    df_metrics = df_metrics.copy()
    if filter_by is not None:
        for key, vals in filter_by.items():
            df_metrics = df_metrics[df_metrics[key].isin(vals)]

    if df_metrics.loc[df_metrics['BestTrial'] == 1, 'Trial'].nunique() != 1:
        raise ValueError('More than one best trial index is available. '
                         'Filter input dataframe accordingly.')

    best_trial = df_metrics.loc[df_metrics['BestTrial'] == 1, 'Trial'].unique()[0]

    # === Extract and pad metrics ===
    df_metrics.sort_values(by=['Trial', 'Epoch'], inplace=True)
    gb = df_metrics.groupby('Trial')
    train_losses = _pad_trials([group['TrainLoss'].values for _, group in gb])
    valid_losses = _pad_trials([group['ValidLoss'].values for _, group in gb])

    accuracies = None
    if task in ['class', 'multi']:
        accuracies = _pad_trials([group['Accuracy'].values for _, group in gb])
    mses = None
    if task in ['reg', 'multi']:
        mses = _pad_trials([group['MSE'].values for _, group in gb])

    # === Determine default visibility ===
    show_loss = True if show_loss is None else show_loss
    show_accuracy = (
        (task in ['class', 'multi']) if show_accuracy is None else show_accuracy
    )
    show_mse = (task in ['reg', 'multi']) if show_mse is None else show_mse

    num_trials = df_metrics['Trial'].nunique()
    epochs = np.arange(train_losses.shape[1])

    # === Loss Plot ===
    if show_loss:
        plt.figure(figsize=figsize)
        for i in range(num_trials):
            plt.plot(train_losses[i], color='tab:blue', alpha=0.5)
            plt.plot(valid_losses[i], color='tab:orange', alpha=0.5)

        # Add one representative line for legend
        plt.plot([], [], color='tab:blue', label='Training')
        plt.plot([], [], color='tab:orange', label='Validation')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_loss.png'),
                        dpi=300, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()

    # === Accuracy Plot ===
    if show_accuracy and not np.all(np.isnan(accuracies)):
        plt.figure(figsize=figsize)
        for i in range(num_trials):
            color = 'black' if i != best_trial else 'tab:green'
            alpha = 0.2 if i != best_trial else 1.0
            lw = 1 if i != best_trial else 2.5
            label = None if i != best_trial else f'Best Model'
            plt.plot(accuracies[i], color=color, alpha=alpha,
                     linewidth=lw, label=label)

        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy (Best Model Highlighted)')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_accuracy.png'),
                        dpi=300, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()

    # === MSE Plot ===
    if show_mse and not np.all(np.isnan(mses)):
        plt.figure(figsize=figsize)
        for i in range(num_trials):
            color = 'black' if i != best_trial else 'tab:red'
            alpha = 0.2 if i != best_trial else 1.0
            lw = 1 if i != best_trial else 2.5
            label = None if i != best_trial else f'Best Model'
            plt.plot(mses[i], color=color, alpha=alpha,
                     linewidth=lw, label=label)

        plt.xlabel('Epoch')
        plt.ylabel('Validation MSE')
        plt.title('Validation MSE (Best Model Highlighted)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_mse.png'),
                        dpi=300, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()


def plot_acc_across_drivers(
    df_pred: pd.DataFrame,
    filter_by: Optional[dict[str, list]] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes, plt.Axes]]:
    """
    Plot model classification accuracy and lap count for each driver.

    Produces two vertically stacked bar plots:
    - Top: model accuracy per driver (proportion of correct predictions).
    - Bottom: number of laps (samples) per driver.

    Args:
        df_pred (pd.DataFrame): Output from `run_inference`, containing
            at least `'DriverNumber'` and `'Correct'` columns.
        filter_by (dict[str, list], optional): Filter criteria to subset data
            before plotting, for example:
            `{'ModelName': ['TelemetryModelFC'], 'Task': ['class']}`.
        figsize (tuple[int, int]): Size of the figure in inches.
            Defaults to `(10, 5)`.
        show (bool): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, path to save the figure.
        return_objs (bool): If `True`, return the figure and both axes.
            Defaults to `False`.

    Raises:
        ValueError: If no data remains after applying filters.

    Returns (optional):
        If `return_objs` is `True`:
            tuple: (fig, ax1, ax2)
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): Axis for accuracy plot.
            - ax2 (plt.Axes): Axis for lap count plot.
        None otherwise.
    """
    if filter_by is not None:
        df_pred = df_pred.copy()
        for key, vals in filter_by.items():
            df_pred = df_pred[df_pred[key].isin(vals)]

    # Compute per-driver stats
    gb = df_pred.groupby('DriverNumber')
    df_total = gb.size().reset_index(name='Total')
    df_correct = gb['Correct'].sum().reset_index(name='Correct')
    df_accuracy = pd.merge(df_total, df_correct, on='DriverNumber')
    df_accuracy['Accuracy'] = df_accuracy['Correct'] / df_accuracy['Total']
    df_accuracy = df_accuracy.sort_values(by='Accuracy', ascending=False)

    # Extract for plotting
    driver_numbers = df_accuracy['DriverNumber'].astype(str).values
    acc_values = df_accuracy['Accuracy'].values
    lap_values = df_accuracy['Total'].values

    # Plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar chart for accuracy
    ax1.bar(driver_numbers, acc_values, color='skyblue', label='Accuracy')
    ax1.set_ylabel('Prediction Accuracy', color='skyblue')
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(range(len(driver_numbers)))  # Set positions
    ax1.set_xticklabels(driver_numbers, rotation=45, ha='right')  # Set labels

    # Line plot for lap count
    ax2 = ax1.twinx()
    ax2.plot(driver_numbers, lap_values, color='darkorange',
             marker='o', label='Lap Count')
    ax2.set_ylabel('Lap Count', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    # Title and layout
    plt.title('Model Accuracy and Lap Count per Driver')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax1, ax2
    

def plot_confusion_matrix(
    df_pred: pd.DataFrame,
    filter_by: Optional[dict[str, list]] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot a confusion matrix of model predictions versus true driver labels.

    Args:
        df_pred (pd.DataFrame): Output from `run_inference`, containing
            at least `'DriverNumber'` and `'PredictedDriverNumber'` columns.
        filter_by (dict[str, list], optional): Dictionary to filter data before plotting.
            Example: `{'ModelName': ['TelemetryModelLSTM'], 'Task': ['class']}`.
        figsize (tuple[int, int], optional): Size of the figure in inches.
            Defaults to `(10, 8)`.
        show (bool, optional): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, saves the plot to this file path.
        return_objs (bool, optional): If `True`, returns the figure and axes.
            Defaults to `False`.

    Raises:
        ValueError: If no data remains after filtering.

    Returns:
        If `return_objs` is True:
            tuple: (fig, ax)
                fig (plt.Figure): Matplotlib figure object.
                ax (plt.Axes): Matplotlib axes containing the confusion matrix.

        None otherwise.
    """

    if filter_by is not None:
        df_pred = df_pred.copy()
        for key, vals in filter_by.items():
            df_pred = df_pred[df_pred[key].isin(vals)]

    # Prepare data for plotting
    labels = np.sort(df_pred['DriverNumber'].unique())
    cm = confusion_matrix(
        df_pred['DriverNumber'],
        df_pred['PredictedDriverNumber'],
        labels=labels
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax
    

def plot_violins(
    df: pd.DataFrame,
    x_col: str,
    y_col: Union[str, Tuple[str, str]],
    filter_by: Optional[dict[str, list]] = None,
    show_swarm: bool = False,
    ylog: bool = False,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    title: Optional[str] = None,
    violin_color: str = "#d8dee9",
    swarm_color: str = "#4c566a",
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot violin plots (optionally with swarm plots) of a variable grouped by a categorical column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to plot.
        x_col (str): Name of the categorical column for x-axis grouping.
        y_col (str or tuple[str, str]): Name of the numerical column to plot.
            Supports MultiIndex if tuple is provided.
        filter_by (dict[str, list], optional): Dictionary of column filters.
            Rows are retained if the value in each key-column is in the corresponding list.
        show_swarm (bool): If `True`, overlay swarmplot on top of violins.
        ylog (bool): If `True`, apply log scale to y-axis.
        xtitle (str, optional): X-axis label.
        ytitle (str, optional): Y-axis label.
        title (str, optional): Title of the plot.
        violin_color (str): Color for violin plots. Defaults to `'#d8dee9'`.
        swarm_color (str): Color for swarm plots. Defaults to `'#4c566a'`.
        figsize (tuple[int, int]): Figure size in inches. Defaults to `(10, 5)`.
        show (bool): If `True`, display the plot immediately.
        save_path (str, optional): If provided, save the plot to this file path.
        return_objs (bool): If True, return the matplotlib Figure and Axes objects.

    Raises:
        ValueError: If `y_col` is not found in the DataFrame.
        ValueError: If filtering leaves no data.
        ValueError: If all values in `y_col` are NaN after filtering.

    Returns:
        If `return_objs` is `True`:
            tuple: (fig, ax):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): The matplotlib axes.
        None otherwise.
    """

    # Check input data compatibility
    if y_col not in df.columns:
        raise ValueError(f'Column {y_col} not found in DataFrame.')

    # Filter data for plotting
    df_plot = df.copy()
    if filter_by:
        for key, vals in filter_by.items():
            df_plot = df_plot[df_plot[key].isin(vals)]
    if df_plot.empty:
        raise ValueError('No data available after filtering. '
                         'Check your threshold or input dataframe.')

    # Prepare data for plotting
    df_plot = df_plot.dropna(subset=[y_col])
    if df_plot.empty:
        raise ValueError(f'No {y_col} column data to plot after dropping NaNs.')

    # Populate the plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df_plot, x=x_col, y=y_col,
        inner=None, density_norm='area',
        linewidth=1, alpha=0.8,
        color=violin_color, ax=ax
    )

    if show_swarm:
        if len(df_plot) > 500:
            warnings.warn('Swarmplot may be crowded; consider filtering data.')
        sns.swarmplot(
            data=df_plot, x=x_col, y=y_col,
            linewidth=0, size=2, alpha=0.6,
            color=swarm_color, ax=ax
        )

    # Axis and title formatting
    ax.set_xlabel(xtitle if xtitle else x_col, fontsize=14)
    ax.set_ylabel(ytitle if ytitle else y_col, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=10)
    if ylog:
        ax.set_yscale('log')

    # Grid and layout
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax
    

def plot_param_importance_vs_drivers(
    df_attr: pd.DataFrame,
    filter_by: Optional[dict[str, list]] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Visualize parameter importance across drivers from neural network attribution results.

    This function generates a heatmap:
        - X-axis: Drivers
        - Y-axis: Model Parameters (e.g., Speed, Brake)
        - Color: Maximum importance value across the lap

    Args:
        df_attr (pd.DataFrame): Output from `run_inference`, containing at least
            the columns `['DriverNumber', 'Param', 'Importance_LapsMean']`.
        filter_by (dict[str, list], optional): Dictionary to filter data before plotting.
            Example: `{'ModelName': ['TelemetryModelCNNLSTM'], 'Task': ['multi']}`.
        figsize (tuple[int, int]): Size of the plot in inches. Defaults to `(10, 8)`.
        show (bool): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, saves the plot to the given file path.
        return_objs (bool): If `True`, return the figure and axes. Defaults to `False`.

    Raises:
        ValueError: If no data remains after filtering or required columns are missing.

    Returns:
        If `return_objs` is `True`:
            tuple: (fig, ax)
                fig (plt.Figure): Matplotlib figure.
                ax (plt.Axes): Matplotlib axes for the heatmap.
        None otherwise
    """
    if filter_by is not None:
        df_attr = df_attr.copy()
        for key, vals in filter_by.items():
            df_attr = df_attr[df_attr[key].isin(vals)]

    # Prepare data for plotting
    gb = df_attr.groupby(['DriverNumber', 'Param'])
    df = gb['Importance_mean'].agg('max').reset_index()
    # Pivot the data for heatmap format: DriverNumber as rows, Param as columns
    heatmap_data = df.pivot(index='DriverNumber',
                            columns='Param',
                            values='Importance_mean')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu',
                ax=ax, cbar_kws={'label': 'Max of Importance_mean'})
    ax.set_xlabel('Driver Number')
    ax.set_ylabel('Parameter')
    ax.set_title('Parameter Importance Across Drivers')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax