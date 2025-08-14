# style2speed/plotting/circuit.py

from typing import Literal, Optional, Tuple, Union
import warnings

import fastf1 as ff1
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style2speed.config import GROUP_COLS


def visualize_circuit(
    df_telemetry: Optional[pd.DataFrame] = None,
    session: Optional[ff1.core.Session] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = (10, 5),
    interval: int = 500,
    interval_mark_color: str = 'crimson',
    show_corners: bool = True,
    corner_label_offset: int = 500,
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Visualize an F1 circuit from telemetry data or a FastF1 session.

    Args:
        df_telemetry (pd.DataFrame, optional): Telemetry data containing at least
            `'DriverNumber'`, `'LapNumber'`, `'X'`, `'Y'`, and `'Distance'` columns.
        session (ff1.core.Session, optional): FastF1 session object. If provided,
            telemetry is extracted automatically. Either `df_telemetry` or
            `session` must be provided, not both.
        title (str, optional): Title of the plot. If `session` is provided and
            `title` is `None`, the session's event location is used.
        figsize (tuple, optional): Figure size in inches. Defaults to `(10, 5)`.
        interval (int): Distance interval in meters for marking the circuit.
            Defaults to 500.
        interval_mark_color (str): Color used to draw interval markings.
            Defaults to `'crimson'`.
        show_corners (bool): Whether to label corner numbers (requires session).
            Defaults to `True`.
        corner_label_offset (int): Distance in meters to offset corner labels
            from the track. Defaults to 500.
        show (bool): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, the plot is saved to this file path.
        return_objs (bool): If `True`, return the figure and axes objects.
            Defaults to `False`.

    Raises:
        ValueError: If neither `df_telemetry` nor `session` is provided.
        ValueError: If both `df_telemetry` and `session` are provided.
        ValueError: If telemetry is empty or missing required columns.

    Returns:
        If `return_objs` is `True`:
            tuple: (fig, ax):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): The matplotlib axes.
        None otherwise.
    """

    # Check input:
    if session is None and df_telemetry is None:
        raise ValueError('Provide either df_telemetry or session.')
    if session is not None and df_telemetry is not None:
        raise ValueError('Only one of df_telemetry or session can be provided.')

    # Get example trajectory along circuit
    if df_telemetry is not None:
        required_cols = set(GROUP_COLS).union({'Distance', 'X', 'Y'})
        if not required_cols.issubset(df_telemetry.columns):
            raise ValueError(f'df_telemetry must include columns: {required_cols}')
        gb = df_telemetry.groupby(GROUP_COLS)
        df = gb.get_group(gb.size().idxmax())  # lap w/ most data points
    elif session is not None:
        df = session.laps.pick_fastest().get_telemetry()  # fastest lap

    if df.empty:
        raise ValueError('No data available in example trajectory.')
    required_cols = {'Distance', 'X', 'Y'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'Required data is missing from input: '
                         f'{required_cols}')

    df = df.sort_values('Distance')
    track = df[['X', 'Y']].to_numpy()
    distance = df['Distance'].to_numpy()

    # Generate distance interval marks and the corresponding indices
    if distance[-1] < interval:
        marks = [0]
    else:
        marks = np.arange(0, distance[-1], interval)
    marks_indices = [np.argmin(np.abs(distance - d)) for d in marks]

    # Plot the circuit
    with plt.rc_context({'font.family': 'DejaVu Sans'}):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(track[:, 0], track[:, 1],
                linewidth=3, color='black', alpha=0.8)

        for i, dist in zip(marks_indices, marks):
            x, y = track[i]
            ax.text(
                x, y, f'{int(dist)}m',
                fontsize=10, color=interval_mark_color,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7,
                          boxstyle='round,pad=0.2'))

        # If session is provided, add corner labels as well
        if show_corners and session:

            circuit_info = session.get_circuit_info()

            offset_vector = [corner_label_offset, 0]
            for _, corner in circuit_info.corners.iterrows():

                txt = f"{corner['Number']}{corner['Letter']}"  # Corner label

                # Generate position for corner label on the plot
                angle = corner['Angle'] / 180 * np.pi  # Corner angle in rad
                offset_x, offset_y = np.dot(
                    [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]],
                    offset_vector
                )
                track_x, track_y = corner['X'], corner['Y']
                text_x, text_y = track_x + offset_x, track_y + offset_y

                # Draw a circle next to the track
                ax.scatter(text_x, text_y, color='black', s=200)

                # Draw a line from the track to this circle
                ax.plot([track_x, text_x], [track_y, text_y], color='black')

                # Add corner label
                ax.text(text_x, text_y, txt,
                        fontsize=10, color='white',
                        va='center_baseline', ha='center')

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor('#f7f7f7')

        if session is not None and title is None:
            title = f"{session.event['Location']} Circuit"
        if title is not None:
            ax.set_title(title, fontsize=14, weight='bold')

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax
    

def visualize_along_circuit_ind_values(
    df: pd.DataFrame,
    z_col: Union[str, Tuple[str, str]],
    filter_by: Optional[dict[str, list]] = None,
    agg_func: str = 'mean',
    mode: Literal['line', 'point'] = 'line',
    linewidth: int = 3,
    pointsize: int = 50,
    cmap_name: str = 'viridis',
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
    cbar_label: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Visualize individual values along XY positions of a racing circuit.

    Supports both line and scatter modes. If multiple values remain at the same
    XY position after filtering, the aggregation function is applied.

    Args:
        df (pd.DataFrame): Input dataframe. Must contain `'Distance'`, `'X'`, `'Y'`,
            and the `z_col` column(s).
        z_col (str or tuple[str, str]): Column (or MultiIndex tuple) to visualize
            along the circuit. If a tuple is passed, `df` must have MultiIndex columns.
        filter_by (dict[str, list], optional): Dictionary specifying filtering criteria
            (e.g., `{'Driver': [1, 11]}`).
        agg_func (str): Aggregation function to apply when multiple values
            exist per XY location. Defaults to `'mean'`.
        mode (Literal['line', 'point']): Whether to draw a continuous line
            or scatter plot. Defaults to `'line'`.
        linewidth (int): Line width (only used in `'line'` mode). Defaults to 3.
        pointsize (int): Point size (only used in `'point'` mode).Defaults to 50.
        cmap_name (str): Name of the colormap to use. Defaults to `'viridis'`.
        cbar_min (float, optional): Minimum value for the colorbar. If `None`,
            determined from the data. Defaults to `None`.
        cbar_max (float, optional): Maximum value for the colorbar. If `None`,
            determined from the data. Defaults to `None`.
        cbar_label (str, optional): Label for the colorbar. Defaults to `None`.
        title (str, optional): Title of the plot. Defaults to `None`.
        figsize (tuple[int, int]): Figure size in inches. Defaults to `(10, 5)`.
        show (bool): If `True`, display the plot. Defaults to `True`.
        save_path (str, optional): If provided, save the plot to this file path.
        return_objs (bool): If `True`, return the figure and axis objects.
            Defaults to `False`.

    Raises:
        ValueError: If `z_col` is a tuple but df lacks MultiIndex columns.
        ValueError: If `z_col` is not found in the dataframe.
        ValueError: If required trajectory columns are missing.
        ValueError: If the mode is not one of `{'line', 'point'}`.
        ValueError: If no data remains after filtering.
        ValueError: If no valid values remain for coloring the circuit.

    Returns:
        If `return_objs` is `True`:
            tuple: (fig, ax):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): The matplotlib axes.
        None otherwise.
    """

    # Check input data
    if isinstance(z_col, tuple):
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError(f'Tuple {z_col} provided for z-col, '
                             f'but df_segment has no MultiIndex.')

    if z_col not in df.columns:
        raise ValueError(f'Column {z_col} not found in input dataframe')

    trajectory_cols = {'Distance', 'X', 'Y'}
    if not trajectory_cols.issubset(df.columns):
        raise ValueError(f'Missing one or more required trajectory columns: '
                         f'{trajectory_cols}')

    if mode not in {'line', 'point'}:
        raise ValueError("mode must be either 'line' or 'point'")

    # Filter data for plotting
    df = df.copy()
    if filter_by:
        for key, vals in filter_by.items():
            df = df[df[key].isin(vals)]
    if df.empty:
        raise ValueError('No data available after filtering. '
                         'Check your threshold or input dataframe.')

    # Check if multiple values can remain after filtering (suppress X/Y noise)
    gb = df.groupby('Distance')
    if gb.size().max() > 1:

        warnings.warn(
            f'After filtering, more than one {z_col} value available '
            f'at a given Distance. Using {agg_func} for aggregation.'
        )

        # Aggregate z_col and compute representative X/Y for plotting
        df_agg = gb[z_col].agg(agg_func).reset_index()
        df_xy = gb[['X', 'Y']].mean().reset_index()

        # Merge X/Y back into main DataFrame
        df = pd.merge(df_agg, df_xy, on='Distance', how='left')

    else:
        df = df[['Distance', 'X', 'Y', z_col]]

    # Extract values for consistent plotting
    df = df.sort_values('Distance')
    x, y, z = df['X'].values, df['Y'].values, df[z_col].values

    if mode == 'line':

        # Create xy line segments to show track
        segments = [[(x[i], y[i]), (x[i+1], y[i+1])] for i in range(len(x)-1)]

        # Use average values between two consecutive points to color track
        colors = np.nanmean(np.column_stack((z[:-1], z[1:])), axis=1)

        # Separate valid and NaN values
        valid_mask = ~np.isnan(colors)
        segments_valid = [s for s, v in zip(segments, valid_mask) if v]
        segments_nan = [s for s, v in zip(segments, valid_mask) if not v]
        colors_valid = colors[valid_mask]

    elif mode == 'point':

        colors = z
        valid_mask = ~np.isnan(colors)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        colors_valid = colors[valid_mask]

    if colors_valid.size == 0:
        raise ValueError('No valid color values to display on circuit.')

    # Set colormap scale & color scheme
    if cbar_min is None:
        cbar_min = np.nanmin(colors_valid)
    if cbar_max is None:
        cbar_max = np.nanmax(colors_valid)
    norm = plt.Normalize(cbar_min, cbar_max)
    cmap = plt.get_cmap(cmap_name)

    # Populate plot
    with plt.rc_context({'font.family': 'DejaVu Sans'}):
        fig, ax = plt.subplots(figsize=figsize)

        if mode == 'line':
            lc_valid = mc.LineCollection(segments_valid, array=colors_valid,
                                        cmap=cmap, norm=norm,
                                         linewidth=linewidth)
            lc_nan = mc.LineCollection(segments_nan, colors='lightgrey',
                                      linestyle='dotted', linewidth=linewidth)
            ax.add_collection(lc_valid)  # show valid values in color
            ax.add_collection(lc_nan)  # show NaN values in grey
            cbar = plt.colorbar(lc_valid, ax=ax)
        elif mode == 'point':
            sc = ax.scatter(x_valid, y_valid, c=colors_valid, cmap=cmap, norm=norm,
                            s=pointsize)
            cbar = plt.colorbar(sc, ax=ax)

        # Format plot
        ax.set_facecolor('#f7f7f7')

        ax.autoscale()
        ax.set_aspect('equal')

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title is None:
            if isinstance(z_col, tuple):
                title = f"{' / '.join(z_col)} along circuit"
            else:
                title = f'{z_col} along circuit'
        ax.set_title(title, fontsize=14, weight='bold')

        if cbar_label is None:
            if isinstance(z_col, tuple):
                cbar_label = ' / '.join(z_col)
            else:
                cbar_label = z_col
        cbar.set_label(cbar_label)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax
    

def visualize_along_circuit_segment_vals(
    df_segment: pd.DataFrame,
    z_col: Union[str, Tuple[str, str]] = 'rmsse',
    df_telemetry: Optional[pd.DataFrame] = None,
    session: ff1.core.Session = None,
    filter_by: Optional[dict[str: list]] = None,
    agg_func: str = 'mean',
    linewidth: int = 3,
    cmap_name: str = 'viridis',
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
    cbar_label: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None,
    return_objs: bool = False
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Visualize segment-level statistics along a racing circuit.

    Unlike per-point visualization, this function visualizes data aggregated over
    track segments (e.g., lap-time RMSE per segment, p-values of driver differences).

    Provide either `df_telemetry` or `session` to extract the trajectory.

    Args:
        df_segment (pd.DataFrame): DataFrame with one of the following:
            - Param statistics per driver/lap/segment (e.g., mean, std, RMSE).
            - Statistical values (e.g., p-values or effect sizes) per
              param/stat/segment combination.
            Must include `'Segment'` and a value column matching `z_col`.
        z_col (str or tuple[str, str]): Column in `df_segment` to visualize
              (can be a MultiIndex if needed). Defaults to `'rmsse'`.
        df_telemetry (pd.DataFrame, optional): Telemetry data used to extract
            the circuit trajectory (must include `'DriverNumber'`, `'LapNumber'`,
            `'Distance'`, `'X'`, `'Y'`).
        session (ff1.core.Session, optional): FastF1 session object to extract
            circuit layout.
        filter_by (dict[str, list], optional): Dictionary for filtering `df_segment`
            by specific values (e.g.,`{'Driver': ['1', '11']}`).
        agg_func (str): Aggregation function to apply to `z_col` if multiple
            values remain per segment after filtering. Defaults to `'mean'`.
        linewidth (int): Line width of the segment trace. Defaults to 3.
        cmap_name (str): Name of the colormap to use. Defaults to `'viridis'`.
        cbar_min (float, optional): Minimum colorbar value. If `None`, inferred from data.
        cbar_max (float, optional): Maximum colorbar value. If `None`, inferred from data.
        cbar_label (str, optional): Label for the colorbar. Defaults to `None`.
        title (str, optional): Plot title. Defaults to `None`.
        figsize (tuple[int, int]): Figure size in inches. Defaults to `(10, 5)`.
        show (bool): Whether to display the plot. Defaults to `True`.
        save_path (str, optional): If provided, saves the plot to the given file path.
        return_objs (bool): If `True`, return the figure and axis objects.
            Defaults to False.

    Raises:
        ValueError: If `z_col` is not found in `df_segment`.
        ValueError: If required segment columns are missing.
        ValueError: If neither `df_telemetry` nor `session` is provided.
        ValueError: If both `df_telemetry` and `session` are provided.
        ValueError: If no data remains after filtering.
        ValueError: If multiple segments are mapped to the same trajectory distance.
        ValueError: If all values to visualize are NaN or missing.

    Returns:
        If `return_objs` is `True`:
            tuple: (fig, ax):
            - fig (plt.Figure): The matplotlib figure.
            - ax1 (plt.Axes): The matplotlib axes.
        None otherwise.
    """

    def assign_value(distance, df_values, col) -> float:
        """Assign a value to a position along circuit distance to the value of
        the circuit segment the position belongs to.
        """
        df_row = df_values[
            (df_values['Segment_Start'].values <= distance) &
            (df_values['Segment_End'].values > distance)]
        if len(df_row) > 1:
            raise ValueError(f'Multiple segments found for distance {distance}')
        return df_row[col].iloc[0] if not df_row.empty else np.nan

    if z_col not in df_segment.columns:
        raise ValueError(f'Column {z_col} not found in input dataframe')

    required_cols = {'Segment_Start', 'Segment_End'}
    if not required_cols.issubset(df_segment.columns):
        raise ValueError(f'Missing one or more required segment columns: '
                         f'{required_cols}')

    if df_telemetry is None and session is None:
        raise ValueError('Provide either df_telemetry or session.')
    if df_telemetry is not None and session is not None:
        raise ValueError('Only one of df_telemetry or session can be provided.')

    # Filter data for plotting
    df = df_segment.copy()
    if filter_by:
        for key, vals in filter_by.items():
            df = df[df[key].isin(vals)]
    if df.empty:
        raise ValueError('No data available after filtering. '
                         'Check your threshold or input dataframe.')

    # Group by circuit segments and aggregate
    gb = df.groupby(['Segment_Start', 'Segment_End'])
    df_agg = gb[[z_col]].agg(agg_func).reset_index()

    # Get track trajectory and combine data
    if df_telemetry is not None:
        gb_ = df_telemetry.groupby(['DriverNumber', 'LapNumber'])
        _, group = next(iter(gb_))
        df_track = group[['Distance', 'X', 'Y']]
    else:
        lap = session.laps.pick_fastest()
        df_track = lap.get_telemetry()[['Distance', 'X', 'Y']]
    df_track = df_track.copy()
    df_track[z_col] = df_track['Distance'].apply(
        lambda d: assign_value(d, df_agg, z_col))
    df_track = (
        df_track.dropna(subset=[z_col])
        .sort_values('Distance')
        .reset_index(drop=True)
    )

    x, y, z = df_track['X'].values, df_track['Y'].values, df_track[z_col].values

    # Create xy line segments to show track
    segments = [[(x[i], y[i]), (x[i+1], y[i+1])] for i in range(len(x)-1)]

    # Use average values between two consecutive points to color track
    colors = np.nanmean(np.column_stack((z[:-1], z[1:])), axis=1)

    # Separate valid and NaN values
    valid_mask = ~np.isnan(colors)
    segments_valid = [s for s, v in zip(segments, valid_mask) if v]
    segments_nan = [s for s, v in zip(segments, valid_mask) if not v]
    colors_valid = colors[valid_mask]
    if colors_valid.size == 0:
        raise ValueError('No valid color values to display on circuit.')

    # Set colormap scale & color scheme
    if cbar_min is None:
        cbar_min = np.nanmin(colors_valid)
    if cbar_max is None:
        cbar_max = np.nanmax(colors_valid)
    norm = plt.Normalize(cbar_min, cbar_max)
    cmap = plt.get_cmap(cmap_name)

    # Populate plot
    with plt.rc_context({'font.family': 'DejaVu Sans'}):
        fig, ax = plt.subplots(figsize=figsize)
        lc_valid = mc.LineCollection(segments_valid, array=colors_valid,
                                    cmap=cmap, norm=norm, linewidth=linewidth)
        lc_nan = mc.LineCollection(segments_nan, colors='lightgrey',
                                   linestyle='dotted', linewidth=linewidth)
        ax.add_collection(lc_valid)  # show valid values in color
        ax.add_collection(lc_nan)  # show NaN values in grey
        cbar = plt.colorbar(lc_valid, ax=ax)

        # Format plot
        ax.set_facecolor('#f7f7f7')

        ax.autoscale()
        ax.set_aspect('equal')

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title is None:
            if isinstance(z_col, tuple):
                title = f"{' / '.join(z_col)} along circuit"
            else:
                title = f'{z_col} along circuit'
        ax.set_title(title, fontsize=14, weight='bold')

        if cbar_label is None:
            if isinstance(z_col, tuple):
                cbar_label = ' / '.join(z_col)
            else:
                cbar_label = z_col
        cbar.set_label(cbar_label)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show() if show else plt.close()

    if return_objs:
        return fig, ax