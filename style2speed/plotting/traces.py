# style2speed/plotting/traces.py

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from style2speed.config import TAB10_CUSTOM


def _filter_data(
    df: pd.DataFrame,
    filter_by: Optional[dict[str, list]] = None,
    limit_by: Optional[dict[str, Tuple[float, float]]] = None,
    subsample_by: Optional[dict[str, int]] = None,
    subsample_group_col: Optional[str] = None
) -> pd.DataFrame:
    """Filter a DataFrame based on provided criteria.

    Args:
        df (pd.DataFrame): Input DataFrame.
        filter_by (dict[str, list], optional): Filters for categorical values.
            Rows are retained if values are in the provided lists.
        limit_by (dict[str, tuple[float, float]], optional): Filters for
            continuous values. Rows are retained if values are within range.
        subsample_by (dict[str, int], optional): Subsampling for specific columns.
            Keys are column names; values are sampling rate
            (e.g., keep every n-th value).
        subsample_group_col (str, optional): Grouping column for subsampling
            (e.g., group by driver before subsampling laps).

    Returns:
        pd.DataFrame: Filtered and subsampled DataFrame.
    """
    df = df.copy()

    # Filter by categorical values
    if filter_by is not None:
        for col, vals in filter_by.items():
            if col in df.columns:
                df = df[df[col].isin(vals)]

    # Filter by continuous value limits
    if limit_by is not None:
        for col, (low, high) in limit_by.items():
            if col in df.columns:
                df = df[(df[col] >= low) & (df[col] <= high)]

    # Subsample
    if subsample_by is not None:
        dfs = []
        if subsample_group_col and subsample_group_col in df.columns:
            groups = df[subsample_group_col].unique()
            for g in groups:
                df_group = df[df[subsample_group_col] == g]
                for sub_col, rate in subsample_by.items():
                    if sub_col in df_group.columns:
                        unique_vals = sorted(df_group[sub_col].unique())
                        sampled_vals = unique_vals[::rate]
                        df_group = df_group[df_group[sub_col].isin(sampled_vals)]
                dfs.append(df_group)
        else:
            df_group = df
            for sub_col, rate in subsample_by.items():
                if sub_col in df_group.columns:
                    unique_vals = sorted(df_group[sub_col].unique())
                    sampled_vals = unique_vals[::rate]
                    df_group = df_group[df_group[sub_col].isin(sampled_vals)]
            dfs.append(df_group)

        df = pd.concat(dfs, ignore_index=True)

    return df


def _assign_group_colors(df, group_col, group_colors=None):
    groups = sorted(df[group_col].dropna().unique()) if group_col else [None]
    if group_colors is None:
        palette = sns.color_palette(TAB10_CUSTOM[:len(groups)])
        group_colors = dict(zip(groups, palette)) if group_col else {None: palette[0]}
    return groups, group_colors


def _plot_group_trace(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    mode: str,  # options: 'ind_traces', 'avg_traces'
    label: str,
    color: str,
    alpha_line: float = 0.8,
    alpha_std: float = 0.2,
    linewidth: float = 1.0,
    linestyle:str = '-',
    show_std: bool = True,
    std_col: Optional[str] = 'std',
    subgroup_col: Optional[str] = 'LapNumber'
):
    """Display y_col vs x_col traces on the input axis for a single data group.

    If multiple `y_col` values exist at any `x_col` value, plot either individual
    traces or aggregate as mean ± std, depending on `mode`.
    If only a single `y_col` value is available at any `x_col` value,
    plot available data as is (optionally, include std band).

    Args:
        df (pd.DataFrame): Input data. Must contain columns `x_col` and `y_col`.
        x_col (str): X-axis column in `df`.
        y_col (str): Y-axis column in `df`.
        ax (plt.Axes): Matplotlib axis to plot on.
        mode (str): Plot mode. Options:
            - 'ind_traces': Show individual traces.
            - 'avg_traces': Show mean trace with optional std band.
            Defaults to 'avg_traces'.
        label (str): Trace label.
        color (str): Trace color.
        alpha_line (float): Line alpha value. Defaults to 0.8.
        alpha_std (float): Std alpha value. Defaults to 0.2.
        linewidth (float): Line width. Defaults to 1.0.
        linestyle (str): Line style. Defaults to '-'.
        show_std (bool): Whether to show std values. If multiple `y_col` values
            are available at any `x_col` value, std values are calculated
            internally over those available values. If a single `y_col` value is
            available at any `x_col` value, std values should be present in
            `df` under column `std_col`. If missing, std display is silently
            skipped. Defaults to `True`.
        std_col (str, optional): Column in `df` containing standard deviation
            values for `y_col`. Only used if `show_std == True` and if a single
            `y_col` value is available at any `x_col` value. Defaults to `'std'`.
        subgroup_col (str, optional): Column in `df` to subgroup traces by in
            `mode == 'ind_traces'` mode. Defaults to `'LapNumber'`.

    Raises:
        ValueError: If required columns are not found in the DataFrame.
        ValueError: If no data remains after dropping NaNs.
        ValueError: If unsuported `mode` is provided.
    """
    # Input compatibility
    required_cols = {x_col, y_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'Missing required columns: '
                         f'{required_cols - set(df.columns)}')
    if mode == 'ind_traces' and subgroup_col is not None:
        if subgroup_col not in df.columns:
            raise ValueError(f'Missing required column: {subgroup_col}')

    # Drop NaNs before plotting std
    if show_std and std_col in df.columns:
        df = df.dropna(subset=[x_col, y_col, std_col])

    if df.empty:
        raise ValueError('No data to plot for this group.')

    max_group_size = df.groupby(x_col).size().max()

    if max_group_size > 1:
        # If multiple y_col values at any x_col values, plot ind or avg traces
        if mode == 'ind_traces':
            gb = df.groupby(subgroup_col)
            for i, (_, df_sub) in enumerate(gb):
                if not df_sub.empty:
                    sns.lineplot(data=df_sub, x=x_col, y=y_col, ax=ax, hue=None,
                                label=label if i == 0 else None,
                                color=color, alpha=alpha_line,
                                linewidth=linewidth, linestyle=linestyle)

        elif mode == 'avg_traces':
            gb = df.groupby(x_col)
            stats = gb[y_col].agg(['mean', 'std']).reset_index()
            sns.lineplot(data=stats, x=x_col, y='mean', ax=ax, hue=None,
                        label=label, color=color, alpha=alpha_line,
                        linewidth=linewidth, linestyle=linestyle)
            if show_std:
                ax.fill_between(stats[x_col],
                                stats['mean'] - stats['std'],
                                stats['mean'] + stats['std'],
                                color=color, alpha=alpha_std)

        else:
            raise ValueError(f"Unsupported mode '{mode}'. "
                             f"Choose from ['ind_traces', 'avg_traces'].")

    if max_group_size == 1:
        # If only a single y_col value at any x_col value, plot data as is
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, hue=None,
                    label=label, color=color, alpha=alpha_line,
                    linewidth=linewidth, linestyle=linestyle)
        if show_std and std_col in df.columns:
            ax.fill_between(df[x_col],
                            df[y_col] - df[std_col],
                            df[y_col] + df[std_col],
                            color=color, alpha=alpha_std)
            

def plot_traces(
    df: pd.DataFrame,
    x_col: str,
    y_col: Union[str, Tuple[str, str]],
    group_col: Optional[str] = None,
    group_colors: Optional[dict] = None,
    filter_by: Optional[dict[str, list]] = None,
    limit_by: Optional[dict[str, Tuple[float, float]]] = None,
    subsample_by: Optional[dict[str, int]] = None,
    mode: str = 'avg_traces',
    show_std: bool = True,
    std_col: Optional[Union[str, Tuple[str, str]]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot telemetry-style traces of a variable along a track or time axis.

    Supports individual or averaged traces across groups, with optional filtering,
    standard deviation shading, and subsampling for clarity.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        x_col (str): Column to use for the x-axis (e.g., `'Distance'`).
        y_col (str or tuple[str, str]): Column to use for the y-axis.
            Supports MultiIndex if a tuple is provided.
        group_col (str, optional): Column for grouping traces (e.g., by driver).
            If specific group values should be displayed, include them in
            `filter_by`.
        group_colors (dict, optional): Colors for each group.
            If `None`, a default color palette is used.
        filter_by (dict[str, list], optional): Filters to apply before plotting,
            specified as a dictionary of {column: allowed_values}.
        limit_by (dict[str, tuple[float, float]], optional): Limits to apply
            before plotting, specified as a dictionary of {column: (min, max)}.
        subsample_by (dict[str, int], optional): Subsampling to apply,
            specified as a dictionary of {column: subsample_rate}. If `group_col`
            is provided, subsampling is applied within each group.
        mode (str): Plot mode. Options:
            - 'ind_traces': Show individual traces with optional subsampling
            - 'avg_traces': Show mean trace with optional std band.
            Defaults to 'avg_traces'.
        show_std (bool): If `True`, add standard deviation bands to mean traces.
            If multiple y_col values are available at any x_col value,
            std values are calculated internally over those available values.
            If a single y_col value is available at any x_col value,
            std values should be present in `df` under column `std_col`.
            If missing, std display is silently skipped.
        std_col (str, optional): Column name containing std values for each y value.
            Only used when `mode='avg_traces'` and `show_std=True`, and only if
            each `x_col` value has a single corresponding `y_col` value.
            Supports MultiIndex if a tuple is provided.
            Defaults to `None`.
        x_limits (tuple[float, float], optional): X-axis range to display.
            If `None`, show full extent. Defaults to `None`.
        y_limits (tuple[float, float], optional): Y-axis range to display.
            If `None`, show full extent. Defaults to `None`.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        figsize (tuple[int, int]): Figure size in inches.
        show (bool): If True, display the plot.
        save_path (str, optional): If provided, save the plot to this file path.
        return_objs (bool): If True, return the matplotlib Figure and Axes objects.

    Raises:
        ValueError: If required columns are not found in the DataFrame.
        ValueError: If no data remains after filtering.

    Returns:
        If `return_objs` is True:
            tuple: (fig, ax):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): The matplotlib axes.
        None otherwise.
    """

    # Make y_col and std_col flat if it's a tuple (i.e.., if multi-indexed)
    if isinstance(y_col, tuple):
        y_col_new = '_'.join(y_col)  # New column name
        df[y_col_new] = df[y_col]  # Duplicate column data
        y_col = y_col_new  # Use duplicate column for plotting
    if isinstance(std_col, tuple):
        std_col_new = '_'.join(std_col)
        df[std_col_new] = df[std_col]
        std_col = std_col_new

    # Check input data compatibility
    required_cols = {x_col, y_col}
    required_cols.add(std_col) if std_col else None
    required_cols.add(group_col) if group_col else None
    required_cols.update(filter_by.keys()) if filter_by else None
    required_cols.update(limit_by.keys()) if limit_by else None
    required_cols.update(subsample_by.keys()) if subsample_by else None
    if not required_cols.issubset(df.columns):
        raise ValueError(f'Missing required columns: '
                         f'{required_cols - set(df.columns)}')

    # Filter data for plotting
    df = _filter_data(
        df=df,
        filter_by=filter_by,
        limit_by=limit_by,
        subsample_by=subsample_by,
        subsample_group_col=group_col
    )
    if df.empty:
        raise ValueError('No data available after filtering. '
                         'Check your threshold or input dataframe.')

    # Color palette for each group
    groups, group_colors = _assign_group_colors(df, group_col, group_colors)

    # Populate the plot
    fig, ax = plt.subplots(figsize=figsize)

    for group in groups:

        group_data = df[df[group_col] == group] if group_col else df
        color = group_colors.get(group)
        if color is None:
            print(f"Warning: No color defined for group {group}. Using 'black'.")
            color = 'black'

        _plot_group_trace(df=group_data, x_col=x_col, y_col=y_col, ax=ax,
                          mode=mode, label=group, color=color,
                          show_std=show_std, std_col=std_col)

    # Formatting
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    xlabel = xlabel or x_col
    ylabel = ylabel or y_col
    title = title or f'{y_col} vs {x_col}'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if group_col:  # legend
        ax.legend(title=group_col, bbox_to_anchor=(1.02, 0), loc='lower left')

    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    # Output
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax
    

def overlay_telemetry_and_importance_traces(
    df_tel: pd.DataFrame,
    df_attr: pd.DataFrame,
    y_col_tel: str = 'Brake',
    y_col_attr: str = 'Importance',
    x_col_tel: str = 'Distance',
    x_col_attr: str = 'Distance',
    group_col: Optional[str] = None,
    group_colors: Optional[dict] = None,
    filter_by: Optional[dict[str, list]] = None,
    limit_by: Optional[dict[str, Tuple[float, float]]] = None,
    limit_by_attr: Optional[dict[str, Tuple[float, float]]] = None,
    subsample_by: Optional[dict[str, int]] = None,
    mode: str = 'avg_traces',
    show_std_tel: bool = True,
    show_std_attr: bool = False,
    std_col_tel: Optional[str] = None,
    std_col_attr: Optional[str] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits_tel: Optional[Tuple[float, float]] = None,
    y_limits_attr: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes, plt.Axes]]:
    """
    Overlay telemetry traces with neural network-derived parameter importance
    curves along a shared x-axis (typically circuit distance).

    Args:
        df_tel (pd.DataFrame): Telemetry data. Must contain `x_col_tel`,
            `y_col_tel`, `group_col`, `subsample_col`, and `std_col_tel` columns,
            along with any relevant columns in `filter_by.keys()`.
        df_attr (pd.DataFrame): Parameter importance data, either raw or
            aggregated over driver laps. Must include x_col_attr`, `y_col_attr`,
            `group_col`, `subsample_col`, `std_col_attr`, and `Param` columns,
            along with any relevant columns in `filter_by.keys()`.
        y_col_tel (str): Parameter column in `df_tel` to plot as telemetry trace.
            `y_col_tel` also specifies parameter to filter df_attr.
            Defaults to `'Brake'` as an example.
        y_col_attr (str): Column in `df_attr` to plot as attribution importance.
            Defaults to 'Importance_mean' for `mode == 'avg_traces'`.
            For raw attribution data, set to 'Importance'.
        x_col_tel (str): X-axis column in `df_tel` (e.g., `'Distance'` or
            `'Segment_Start'`). Defaults to `'Distance'`.
        x_col_attr (str): X-axis column in `df_attr`. Should match the units of
            `x_col_tel` for overlaying traces (e.g. [m]). Defaults to `'Distance'`.
        group_col (str, optional): Column in `df_tel` and `df_attr` to group traces by.
            If specific group values should be displayed, include them in `filter_by`.
            Defaults to `None`. 
        group_colors (dict, optional): List of colors for each group.
            If `None`, a default color palette is used. Defaults to `None`.
        filter_by (dict[str, list], optional): Filters to apply before plotting.
            Any `df_attr` specific filters are excluded from filtering `df_tel`.
            Defautls to `None`.
        limit_by (dict[str, tuple[float, float]], optional): Limits to apply
            before plotting to both data frames, specified as a dictionary of
            {column: (min, max)}. Defaults to `None`.
        limit_by_attr (dict[str, tuple[float, float]], optional): Limits to apply
            before plotting, specified as a dictionary of {column: (min, max)},
            on attribution data. If `None`, `limit_by` is used.
        subsample_by (dict[str, int], optional): Subsampling to apply,
            specified as a dictionary of {column: subsample_rate}. If `group_col`
            is provided, subsampling is applied within each group.
            Defaults to `None`.
        mode (str): Display mode. Available options:
            - `'ind_traces'`: Show traces for individual laps.
            - `'avg_traces'`: Show averaged traces.
            Defaults to `'avg_traces'`.
        show_std_tel (bool): Whether to show telemetry std values.
            Defaults to `True`.
        show_std_attr (bool): Whether to show attribution std values.
            Defaults to `False`.
        std_col_tel (str, optional): Column in `df_tel` containing standard
            deviation values for `y_col_tel`. Only used if `mode == 'avg_traces'`,
            `show_std_tel == True` and a single `y_col_tel` value is available
            at any `x_col_tel` value (if multiple values are avaialble, std
            is computed internally). Defaults to `None`.
        std_col_attr (str, optional): Column in `df_attr` containing standard
            deviation values for `y_col_attr`. If provided, attribution std
            values are displayed in `'avg_traces'` mode. Defaults to None.
        x_limits (tuple[float, float], optional): X-axis range to display.
            If `None`, show full extent. Defaults to `None`.
        y_limits_tel (tuple[float, float], optional): Y-axis range for telemetry
            traces. If `None`, show full extent. Defaults to `None`.
        y_limits_attr (tuple[float, float], optional): Y-axis range for attribution
            traces. If `None`, show full extent. Defaults to `None`.
        figsize (tuple[int, int]): Figure size in inches. Defaults to (10, 5).
        title (str, optional): Plot title.
        show (bool): Whether to display the plot. Defaults to True.
        save_path (str, optional): If provided, saves the plot to this file path.
        return_objs (bool): If True, returns the matplotlib figure and axes objects.
            Defaults to False.

    Raises:
        ValueError: If required columns are missing from df_tel or df_attr.
        ValueError: If no data remains after filtering.

    Returns:
        If `return_objs` is True:
            tuple: (fig, ax1, ax2):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): Telemetry axis.
            - ax2 (plt.Axes): Attribution importance axis (as twin axes).
        None otherwise.
    """
    # Check input
    req_cols_tel = {x_col_tel, y_col_tel}
    req_cols_attr = {x_col_attr, y_col_attr,
                    'Param', 'ModelName', 'Task', 'Head'}

    if filter_by is not None:
        for key in filter_by.keys():
            req_cols_tel.add(key) if key not in req_cols_attr else None
            req_cols_attr.add(key)

    req_cols_overlap = set()
    req_cols_overlap.add(group_col) if group_col else None
    req_cols_overlap.update(subsample_by.keys()) if subsample_by else None
    if limit_by is not None and limit_by_attr is None:
        req_cols_overlap.update(limit_by.keys())
    else:
        req_cols_tel.update(limit_by.keys()) if limit_by else None
        req_cols_attr.update(limit_by_attr.keys()) if limit_by_attr else None

    req_cols_tel.update(req_cols_overlap)
    req_cols_attr.update(req_cols_overlap)

    if not req_cols_tel.issubset(df_tel.columns):
        raise ValueError('df_tel is missing required input columns.')
    if not req_cols_attr.issubset(df_attr.columns):
        raise ValueError('df_attr is missing required input columns.')

    # Filter data for plotting
    if isinstance(y_col_tel, tuple):
        param = y_col_tel[0]
    else:
        param = y_col_tel
    filter_by = filter_by.copy() if filter_by else {}
    filter_by['Param'] = [param]

    df_tel = _filter_data(
        df=df_tel, filter_by=filter_by, limit_by=limit_by,
        subsample_by=subsample_by, subsample_group_col=group_col
    )
    df_attr = _filter_data(
        df=df_attr, filter_by=filter_by, limit_by=limit_by,
        subsample_by=subsample_by, subsample_group_col=group_col
    )

    if df_tel.empty:
        raise ValueError('No telemetry data available after filtering. '
                         'Check thresholds or input dataframe.')
    if df_attr.empty:
            raise ValueError('No attribution data available after filtering. '
                             'Check thresholds or input dataframe.')

    # Color palette for each group
    groups, group_colors = _assign_group_colors(df_tel, group_col, group_colors)

    # Plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    for group in groups:

        group_data_tel = (
            df_tel[df_tel[group_col] == group] if group_col else df_tel
        )
        group_data_attr = (
            df_attr[df_attr[group_col] == group] if group_col else df_attr
        )
        color = group_colors.get(group)
        if color is None:
            print(f"Warning: No color defined for group {group}. Using 'black'.")
            color = 'black'

        _plot_group_trace(df=group_data_tel, x_col=x_col_tel, y_col=y_col_tel,
                          ax=ax1, mode=mode, label=group, color=color,
                          linestyle='-', alpha_line=0.5,
                          show_std=show_std_tel, std_col=std_col_tel)

        _plot_group_trace(df=group_data_attr, x_col=x_col_attr, y_col=y_col_attr,
                          ax=ax2, mode=mode, label=None, color=color,
                          linestyle='--', alpha_line=0.7,
                          show_std=show_std_attr, std_col=std_col_attr)

    # Format the plot
    if x_limits is not None:
        ax1.set_xlim(x_limits)
    if y_limits_tel is not None:
        ax1.set_ylim(y_limits_tel)
    if y_limits_attr is not None:
        ax2.set_ylim(y_limits_attr)

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel(f'{param} Telemetry')
    ax2.set_ylabel(y_col_attr)
    if title is not None:
        ax1.set_title(title)

    if group_col:  # legend
        ax1.legend(title=group_col, bbox_to_anchor=(1.1, 0), loc='lower left')

    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    # Output
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax1, ax2
    

def overlay_tel_class_reg_traces(
    df_tel: pd.DataFrame,
    df_attr: pd.DataFrame,
    class_model: str,
    class_task: str,
    reg_model: str,
    reg_task: str,
    y_col_tel: str = 'Brake',
    y_col_attr: str = 'Importance_mean',
    x_col_tel: str = 'Distance',
    x_col_attr: str = 'Distance',
    mode: str = 'avg_traces',
    show_std_tel: bool = True,
    show_std_attr: bool = True,
    std_col_tel: Optional[str] = None,
    std_col_attr: Optional[str] = 'Importance_std',
    filter_by: Optional[dict[str, list]] = None,
    limit_by: Optional[dict[str, Tuple[float, float]]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits_tel: Optional[Tuple[float, float]] = None,
    y_limits_attr: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes, plt.Axes]]:
    """
    Overlay telemetry traces with neural network-derived parameter importance
    curves along a shared x-axis (typically circuit distance) for both
    classification and regression tasks.

    Args:
        df_tel (pd.DataFrame): Telemetry data. Must contain columns `x_col_tel`,
            `y_col_tel`, and any values in `filter_by.keys()`.
        df_attr (pd.DataFrame): Attribution data (raw or stats). Must include
            columns `x_col_attr`, `y_col_attr`, `'ModelName'`, `'Task'`, `'Head'`,
            `'Param'`, and any values in `filter_by.keys()`.
        class_model (str): Model name for classification task.
        class_task (str): Task name for classification task, either `'class'`
            or `'multi'`.
        reg_model (str): Model name for regression task.
        reg_task (str): Task name for regression task, either `'reg'` or `'multi'`.
        y_col_tel (str): Parameter column in `df_tel` to plot as telemetry trace.
            `df_attr` is filtered by this column. Defaults to `'Brake'` as an example.
        y_col_attr (str): Column in `df_attr` to plot as attribution importance.
            Defaults to 'Importance_mean' for attribution stats data. For raw
            attribution data, set to 'Importance'.
        x_col_tel (str): X-axis column in `df_tel` (e.g., `'Distance'` or
            `'Segment_Start'`). Defaults to `'Distance'`.
        x_col_attr (str): X-axis column in `df_attr`. The dimensions of
            `x_col_tel` and `x_col_attr` should match (e.g. [m] or [sec]).
            Defaults to `'Distance'`.
        mode (str): Display mode. Available options:
            - `'ind_traces'`: Show traces for individual laps.
            - `'avg_traces'`: Show averaged traces.
            Defaults to `'avg_traces'`.
        show_std_tel (bool): Whether to show telemetry std values. If multiple
            telemetry values are available at any `x_col_tel` value, std values
            are calculated internally over those available values. If a single
            telemetry value is available at any `x_col_tel` value, std values
            are calculated internally from those available values. If missing,
            telemetry std display is silently skipped. Defaults to `True`.
        show_std_attr (bool): Whether to show attribution std values. If multiple
            attribution values are available at any `x_col_attr` value, std values
            are calculated internally over those available values. If a single
            attribution is available at any `x_col_attr` value, std values
            should be present in `df_attr` under column `std_col_attr`. If missing,
            attr std display is silently skipped. Defaults to `True`.
        std_col_tel (str, optional): Column in `df_tel` containing standard
            deviation values for `y_col_tel`. Defaults to `None`.
        std_col_attr (str, optional): Column in `df_attr` containing standard
            deviation values for `y_col_attr`. Only used if `show_attr_str == True`
            and if a single attribution value is available at any `x_col_attr`
            value. Defaults to `'Importance_std'`.
        filter_by (dict[str, list], optional): Filters to apply before plotting
            on df_tel and df_attr_stats.
        limit_by (dict[str, tuple[float, float]], optional): Limits to apply
            before plotting, specified as a dictionary of {column: (min, max)}.
        x_limits (tuple[float, float], optional): X-axis range to display.
            If `None`, show full extent. Defaults to `None`.
        y_limits_tel (tuple[float, float], optional): Y-axis range for telemetry
            traces. If `None`, show full extent. Defaults to `None`.
        y_limits_attr (tuple[float, float], optional): Y-axis range for attribution
            traces. If `None`, show full extent. Defaults to `None`.
        figsize (tuple[int, int]): Figure size in inches. Defaults to `(10, 5)`.
        show (bool): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, saves the plot to this file path.

    Raises:
        ValueError: If required columns are missing from df_tel or df_attr.
        ValueError: If invalid class_task or reg_task is provided.
        ValueError: If no data remains after filtering.

    Returns:
        If `return_objs` is True:
            tuple: (fig, ax1, ax2):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): Telemetry axis.
            - ax2 (plt.Axes): Attribution importance axis (as twin axes).
        None otherwise.
    """

    # Check inputs
    tel_columns = {x_col_tel, y_col_tel}
    attr_columns = {x_col_attr, y_col_attr, 'Param', 'ModelName', 'Task', 'Head'}
    if filter_by is not None:
        tel_columns.update(filter_by.keys())
        attr_columns.update(filter_by.keys())
    if not tel_columns.issubset(df_tel.columns):
        raise ValueError('df_tel is missing required input columns.')
    if not attr_columns.issubset(df_attr.columns):
        raise ValueError('df_attr is missing required input columns.')
    allowed_class_tasks = {'class', 'multi'}
    allowed_reg_tasks = {'reg', 'multi'}
    if class_task not in allowed_class_tasks:
        raise ValueError(f'Invalid class_task: {class_task}.')
    if reg_task not in allowed_reg_tasks:
        raise ValueError(f'Invalid reg_task: {reg_task}.')

    # Filter data for plotting
    if isinstance(y_col_tel, tuple):
        param = y_col_tel[0]
    else:
        param = y_col_tel
    filter_by = filter_by.copy() if filter_by else {}
    filter_by['Param'] = [param]

    df_tel = _filter_data(
        df=df_tel, filter_by=filter_by, limit_by=limit_by
    )
    df_attr = _filter_data(
        df=df_attr, filter_by=filter_by, limit_by=limit_by
    )

    # Split attribution data into classification and regression branches
    df_class = df_attr[
        (df_attr['ModelName'] == class_model) &
        (df_attr['Task'] == class_task) &
        (df_attr['Head'] == 'class')
    ]
    df_reg = df_attr[
        (df_attr['ModelName'] == reg_model) &
        (df_attr['Task'] == reg_task) &
        (df_attr['Head'] == 'reg')
    ]

    # Check data after filtering
    if df_tel.empty:
        raise ValueError('No telemetry data available after filtering. '
                         'Check thresholds or input dataframe.')
    if df_class.empty or df_reg.empty:
        raise ValueError('No attribution data available after filtering. '
                         'Check thresholds or input dataframe.')

    # Plot with dual axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Telemetry trace (group all remaining values after filtering)
    _plot_group_trace(df=df_tel, x_col=x_col_tel, y_col=y_col_tel, ax=ax1,
                      mode=mode, label='telemetry', color='grey',
                      show_std=show_std_tel, std_col=std_col_tel)

    # Classification importance trace
    _plot_group_trace(df=df_class, x_col=x_col_attr, y_col=y_col_attr, ax=ax2,
                      mode=mode, label='class', color='red',
                      show_std=show_std_attr, std_col=std_col_attr)

    # Regression importance trace
    _plot_group_trace(df=df_reg, x_col=x_col_attr, y_col=y_col_attr, ax=ax2,
                      mode=mode, label='reg', color='blue',
                      show_std=show_std_attr, std_col=std_col_attr)

    # Formating
    if x_limits is not None:
        ax1.set_xlim(x_limits)
    if y_limits_tel is not None:
        ax1.set_ylim(y_limits_tel)
    if y_limits_attr is not None:
        ax2.set_ylim(y_limits_attr)

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel(f'{param} Telemetry')
    ax2.set_ylabel(y_col_attr)
    ax2.legend(title='Task', bbox_to_anchor=(1.1, 0), loc='lower left')
    ax1.set_title(f'{param} Telemetry vs Class & Reg Importance')

    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    # Output
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax1, ax2