# style2speed/preprocessing.py
 
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mode 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from style2speed.config import GROUP_COLS
from style2speed.config import NN_PARAMS
from style2speed.config import PARAMS_CON_POS, PARAMS_CON_SIGNED, PARAMS_ORD_POS, PARAMS_ORD_SIGNED, PARAMS_INV, PARAM_BASELINE_STRATEGY


def calc_segment_times(
    df_tel: pd.DataFrame,
    group_cols: Optional[list[str]] = None,
    dlimits: Optional[tuple[float, float]] = None
) -> pd.DataFrame:
    """Compute segment times for each group (e.g., driver/lap combination).

    If `dlimits` is provided, times are computed only within the specified
    distance interval. Otherwise, the full lap distance is used.

    Args:
        df_tel (pd.DataFrame): Telemetry data with at least `'Distance'` and
            `'Timer'` columns, plus any additional columns in `group_cols`.
        group_cols (list[str], optional): Columns to group by before computing
            segment times. Defaults to `GROUP_COLS`.
        dlimits (tuple[float, float], optional): Start and end distances
            (in meters) that define the segment of interest.
            If None, the full distance range is used.

    Raises:
        ValueError: If `dlimits` is not a tuple of two values or is out of bounds.

    Returns:
        pd.DataFrame: Segment times and metadata for each group.
    """

    df = df_tel.copy()

    # Handle optional arguments
    if group_cols is None:
        group_cols = GROUP_COLS

    available_range = [df['Distance'].min(), df['Distance'].max()]
    if dlimits is not None:
        if not (isinstance(dlimits, tuple) and
                len(dlimits) == 2 and
                all(isinstance(x, (int, float)) for x in dlimits)):
            raise ValueError('dlimits must be a tuple of 2 floats (start, end).')

        start, end = dlimits
        if start < available_range[0] or end > available_range[1]:
            raise ValueError(
                f'Input dlimits {dlimits} exceed available distance range {available_range}.'
            )
        df = df[(df['Distance'] >= start) & (df['Distance'] <= end)]
    else:
        start, end = available_range

    # Segment times for each group
    df_times = (
        df.groupby(group_cols)['Timer']
        .agg(lambda x: x.max() - x.min())
        .reset_index(name='SegmentTime')
    )

    # Segment metadata
    df_times['Segment'] = f'{int(start)}-{int(end)}'
    df_times['Segment_Start'] = start
    df_times['Segment_End'] = end

    return df_times


def get_scaler(method: str):
    """Get sklearn.preprocessing scaler object based on descriptive method string."""
    if method == 'min_max':
        return MinMaxScaler()
    elif method == 'standard' or method == 'zscore':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f'Unknown scaling method: {method}')
    

def get_scaling_params(scaler, method):
    """Get relevant scaling parameters for a sklearn.preprocessing scaler object."""
    params = {'method': method, 'scaler': scaler}
    if hasattr(scaler, 'mean_'):
        params['mean'] = scaler.mean_
    if hasattr(scaler, 'scale_'):
        params['iqr' if method == 'robust' else 'std'] = scaler.scale_
    if hasattr(scaler, 'data_min_'):
        params['min'] = scaler.data_min_
    if hasattr(scaler, 'data_max_'):
        params['max'] = scaler.data_max_
    if hasattr(scaler, 'center_'):
        params['center'] = scaler.center_
    return params


def normalize_telemetry(
    df_tel: pd.DataFrame,
    method_con_pos: str = 'min_max',
    method_con_signed: str = 'zscore',
    method_ord_pos: str = 'min_max',
    method_ord_signed: str = 'zscore',
    method_inv: str = 'min_max',
    scale_per: Optional[list[str]] = None
) -> pd.DataFrame:
    """Normalize telemetry data, optionally per group (e.g., by driver or lap).

    Categorical variables (e.g., Brake, BrakeChange, DRS) are excluded from normalization.

    Args:
        df_tel (pd.DataFrame): Input telemetry DataFrame.
        method_con_pos (str): Scaling method for continuous positive variables.
            One of: `'min_max', 'standard'/'zscore', 'robust'`.
            Defaults to `'min_max'`.
        method_con_signed (str): Scaling method for continuous signed variables.
            Defaults to `'zscore'`.
        method_ord_pos (str): Scaling method for ordinal positive variables.
            Defaults to `'min_max'`.
        method_ord_signed (str): Scaling method for ordinal signed variables.
            Defaults to `'zscore'`.
        method_inv (str): Scaling method for inverse variables
            (e.g., `'FollowingDistanceInv'`). Defaults to `'min_max'`.
        scale_per (list[str], optional): Columns to group by for per-group scaling.
            If None, global scaling is applied. Defaults to None.

    Raises:
        ValueError: If an invalid scaling method is provided.

    Returns:
        pd.DataFrame: Normalized telemetry DataFrame.
    """

    df = df_tel.copy()

    # Map parameters to normalization methods
    method_map = {
        **{param: method_con_pos for param in PARAMS_CON_POS},
        **{param: method_con_signed for param in PARAMS_CON_SIGNED},
        **{param: method_ord_pos for param in PARAMS_ORD_POS},
        **{param: method_ord_signed for param in PARAMS_ORD_SIGNED},
        **{param: method_inv for param in PARAMS_INV},
    }

    # Check inputs
    allowed_methods = {'min_max', 'standard', 'zscore', 'robust'}
    for method in method_map.values():
        if method not in allowed_methods:
            raise ValueError(f'Unsupported scaling method: {method}')

    params = list(method_map.keys())  # all params to scale

    if scale_per:  # scale each group independently
        dfs = []
        gb = df.groupby(scale_per)
        for _, group in gb:
            group_scaled = group.copy()
            for param in params:
                if param in group.columns:
                    scaler = get_scaler(method_map[param])
                    group_scaled[param] = scaler.fit_transform(group[[param]])
            dfs.append(group_scaled)
        df = pd.concat(dfs).sort_index()

    else:   # global scaling
        for param in params:
            if param in df.columns:
                scaler = get_scaler(method_map[param])
                df[param] = scaler.fit_transform(df[[param]])

    return df


def normalize_times(
    df_times: pd.DataFrame,
    time_col: str = 'LapTime',
    method: str = 'standard',
    scale_per: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """Normalize lap time or segment time data.

    Args:
        df_times (pd.DataFrame): Input DataFrame with lap or segment times.
        time_col (str): Name of the column to normalize. Defaults to `'LapTime'`.
            Other examples: `'SegmentTime'`.
        method (str): Scaling method. One of: `'standard'`/`'z-score'`,
            `'min_max'`, or `'robust'`. Defaults to `'standard'`.
        scale_per (str, optional): Column name to group by before scaling
            (e.g., `'Location'`, `'Year'`, `'DriverNumber'`, `'LapNumber'`).
            If None, global scaling is applied.

    Raises:
        ValueError: If an invalid scaling method is provided.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple where:
            - The first element is the normalized DataFrame.
            - The second is a dictionary of scaling parameters. For global scaling,
              the key is 'global'. For group-based scaling, keys correspond to
              group values. Each value is a dictionary with keys like 'method',
              'scaler', 'mean', 'std', 'min', 'max', 'center', 'iqr',
              depending on the scaling method.
    """

    # Check inputs
    allowed_methods = {'min_max', 'standard', 'zscore', 'robust'}
    if method not in allowed_methods:
        raise ValueError(f"Unsupported scaling method: {method}")

    df = df_times.copy()
    scaling_params = {}

    if scale_per:  # scale each group independently
        gb = df.groupby(scale_per, sort=False)
        dfs = []
        for group_key, group in gb:
            group_scaled = group.copy()
            scaler = get_scaler(method)  # new instance per group
            group_scaled[time_col] = scaler.fit_transform(group[[time_col]])
            dfs.append(group_scaled)
            scaling_params[group_key] = get_scaling_params(scaler, method)

        df = pd.concat(dfs).sort_index()

    else:  # global scaling
        scaler = get_scaler(method)
        df[time_col] = scaler.fit_transform(df[[time_col]])
        scaling_params['global'] = get_scaling_params(scaler, method)

    return df, scaling_params


def inverse_normalize(series, scaler):
    return scaler.inverse_transform(series.values.reshape(-1, 1)).flatten()


def get_baseline(
    X: np.ndarray,
    params,
    num_params,
    num_steps,
    strategies: Optional[dict[str, str]] = None
) -> np.ndarray:
    """Compute per-step, per-parameter baseline values.
    - For discrete params, use mode.
    - For continuous params, use median.

    Args:
        X (np.ndarray): Array of shape (num_datasets, num_steps, num_params).
        params (list[str]): List of parameter names in the same order as in `X`.
        num_params (int): Number of parameters.
        num_steps (int): Number of steps along the circuit.
        strategies (dict[str, str], optional): Strategy for computing baseline
            for each param, with keys representing param names and values -
            baseline strategies (one of `'mode'`, `'median'`, `'mean'`, `'zero'`).
            Defaults to `PARAM_BASELINE_STRATEGY`.

    Raises:
        ValueError: If parameter list is missing.
        ValueError: If input shape is invalid.
        ValueError: If a baseline strategy is missing for any param.
        ValueError: If unknown baseline strategy is provided.

    Returns:
        baseline (np.ndarray): Array of shape (num_steps, num_params)
    """
    if params is None:
        raise ValueError('Parameter list must be provided.')

    if X.shape[2] != num_params:
        raise ValueError('Mismatch between X.shape[2] and number of params.')

    if strategies is None:
        strategies = PARAM_BASELINE_STRATEGY

    for param in params:
        if param not in strategies:
            raise ValueError(f'Missing baseline strategy for parameter: {param}')

    for strategy in strategies.values():
        if strategy not in ['mode', 'median', 'mean', 'zero']:
            raise ValueError(f"Invalid baseline strategy: {strategy}")

    baseline = np.zeros((num_steps, num_params))

    for i in range(num_steps):
        for j, param in enumerate(params):
            values = X[:, i, j]
            strategy = strategies[param]

            if strategy == 'mode':
                baseline[i, j] = mode(values, keepdims=False).mode
            elif strategy == 'median':
                baseline[i, j] = np.median(values)
            elif strategy == 'mean':
                baseline[i, j] = np.mean(values)
            elif strategy == 'zero':
                baseline[i, j] = 0
            else:
                raise ValueError(f"Unknown baseline strategy '{strategy}' "
                                 f"for parameter '{param}'. "
                                 f"Allowed: 'mode', 'median', 'mean', 'zero'.")

    return baseline


def build_torch_dataset(
    df_tel: pd.DataFrame,
    drivers: Optional[list[int]] = None,
    params: Optional[list[str]] = None,
    dlimits: Optional[Tuple[float, float]] = None,
    group_cols: Optional[list[str]] = None,
    df_times: Optional[pd.DataFrame] = None,
    time_col: Optional[str] = 'LapTime',
    scaling_method: Optional[str] = 'standard',
    scale_per: Optional[str] = None,
    strategies: Optional[dict[str, str]] = None,
) -> Tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, pd.DataFrame, dict, int, int, int, Optional[dict]]:
    """Build PyTorch-ready dataset from telemetry data for both classification
    and regression tasks.

    If df_times is not provided, also extracts and normalizes lap or segment times.

    Input data must be interpolated and, if desired, normalized prior to calling.

    Args:
        df_tel (pd.DataFrame): Telemetry data for each driver/lap combo.
            Must be pre-interpolated and normalized if needed.
        drivers (list[int], optional): Permanent numbers of drivers to include.
            If None, all drivers in `df_tel` are used.
        params (list[str], optional): Telemetry parameters to extract. Defaults to
            `NN_PARAMS`, filtered by availability in `df_tel`.
        dlimits (tuple[float, float], optional): Distance limits (start, end) of
            circuit segment. If None, full lap is used.
        group_cols (list[str], optional): Columns to group by to identify unique
            datasets. Defaults to `GROUP_COLS`.
        df_times (pd.DataFrame, optional): Segment or lap time values. If None,
            times are computed from telemetry using min/max of `'Timer'`.
            By default, times are also normlaized.
        time_col (str, optional): Column in `df_times` to use. Defaults to `'LapTime'`.
        scaling_method (str, optional): Method used to normalize time values
            if `df_times` is computed internally. Defaults to `'standard'`.
        scale_per (str, optional): Column to group by before normalizing times.
            Ignored if `df_times` is provided.
        strategies (dict[str, str], optional): Strategies for computing baseline
            for each param, with keys representing param names and values -
            baseline strategies (one of `'mode'`, `'median'`, `'mean'`, `'zero'`).
            Defaults to `PARAM_BASELINE_STRATEGY`.

    Raises:
        ValueError: If input params are missing from the dataframe.
        ValueError: If input `group_cols` are missing from the dataframe.
        ValueError: If no telemetry data remains after filtering.
        ValueError: If `'time_col'` is missing in `df_times`.
        ValueError: If inconsistent group sizes or duplicate times found.

    Returns:
        Tuple:
            - X (np.ndarray): Telemetry data.
                Shape: [n_datasets, n_steps, n_params].
            - y (list[np.ndarray]): Labels and times.
                y[0] → driver labels, y[1] → lap/segment times.
            - baseline (np.ndarray): Per-step, per-parameter baseline values.
            - trajectory (np.ndarray): Sample trajectory
                (`'Distance'`, `'X'`, `'Y'`).
            - df_meta (pd.DataFrame): Dataset metadata (`'Dataset_ID'`,
                `'DriverNumber'`, `'LapNumber'`, etc.)
            - label_to_driver (dict): Maps label index to driver number.
            - n_drivers (int): Number of unique drivers.
            - n_steps (int): Number of steps per sample.
            - n_params (int): Number of telemetry features used.
            - scaling_params (dict | None): Normalization params if `df_times`
                was computed and normalized internally.
    """

    df = df_tel.copy()

    if df_times is not None and time_col not in df_times.columns:
        raise ValueError(f'Missing segment/lap time column: {time_col}')

    # Handle optional arguments: group columns
    if group_cols is None:
        group_cols = GROUP_COLS
    if not all(col in df.columns for col in group_cols):
        raise ValueError('Input DataFrame is missing required columns.')
    driver_idx = group_cols.index('DriverNumber')
    lap_idx = group_cols.index('LapNumber')

    # Handle optional arguments: Segment of interest (i.e. dlimits)
    if dlimits:
        df = df[(df['Distance'] >= dlimits[0]) & (df['Distance'] <= dlimits[1])]

    # Handle optional arguements: Drivers
    available_drivers = df['DriverNumber'].unique().tolist()
    if drivers:
        missing_drivers = set(drivers) - set(available_drivers)
        if missing_drivers:
            warnings.warn(f'Missing data for drivers: {sorted(missing_drivers)}')
        drivers = [d for d in drivers if d in available_drivers]
        df = df[df['DriverNumber'].isin(drivers)]
    else:
        drivers = available_drivers
    n_drivers = len(drivers)

    if df.empty:
        raise ValueError('No telemetry data after filtering'
                         ' by drivers and distance limits.')

    driver_labels = list(range(len(drivers)))  # map driver number to label
    label_to_driver = dict(zip(driver_labels, drivers))
    driver_to_label = dict(zip(drivers, driver_labels))

    # Handle optional arguments: Telemetry parameters to include
    if params is None:
        params = [p for p in NN_PARAMS if p in df.columns]
    else:
        missing = [p for p in params if p not in df.columns]
        if missing:
            raise ValueError(f'Missing telemetry parameters: {missing}')
    n_params = len(params)

    # Handle optional arguments: df_times
    if df_times is None:
        df_times = calc_segment_times(df, group_cols=group_cols)
        time_col = 'SegmentTime'
        df_times, scaling_params = normalize_times(df_times, time_col=time_col,
                                                   method=scaling_method)
    else:
        scaling_params = None

    # Handle optional arguments: Baselining strategies (for attributions)
    if strategies is None:
        strategies = PARAM_BASELINE_STRATEGY

    # Get number of time points (should be the same across all sets)
    sizes = set(df.groupby(group_cols).size())
    if len(sizes) > 1:
        raise ValueError('All sets must have the same number of time points')
    n_steps = sizes.pop()

    # Sort both dataframes and assign id's to datasets
    df = df.sort_values(by=group_cols + ['Distance']).reset_index(drop=True)
    df_times = df_times.sort_values(by=group_cols).reset_index(drop=True)
    df_times['Dataset_ID'] = np.arange(len(df_times))

    # Create a ataframe with dataset metadata
    df_meta = df_times.drop(columns=time_col)

    # Initialize output
    X, y_labels, y_times = [], [], []
    trajectory = None

    # Iterate over each driver and lap combination
    for i, (name, group) in enumerate(df.groupby(group_cols)):

        group = group.sort_values(by='Distance').reset_index(drop=True)
        X.append(group[params].values)
        y_labels.append(driver_to_label[name[driver_idx]])

        if i == 0:  # save sample trajectory for future plotting
            trajectory = group[['Distance', 'X', 'Y']].to_numpy()

        df_row = df_times.copy()
        for i, col in enumerate(group_cols):
            df_row = df_row[df_row[col] == name[i]]

        if df_row.empty:
            raise ValueError(
                f'Missing {time_col} for Driver {name[driver_idx]}, Lap {name[lap_idx]}.'
            )
        if len(df_row) != 1:
            raise ValueError(
                f'Got >1 row for Driver {name[driver_idx]}, Lap {name[lap_idx]} in df_times.'
            )
        y_times.append(df_row[time_col].values[0])

    if not all(x.shape == X[0].shape for x in X):
        raise ValueError('Mismatch in telemetry sample shapes. '
                         'Expected uniform step count.')

    # Convert lists to arrays for faster conversion to tensors
    X = np.stack(X)  # [n_sets, min_stamps, n_features]
    y_labels = np.array(y_labels)  # [n_sets]
    y_times = np.array(y_times)  # [n_sets]
    y = [y_labels, y_times]  # [2, n_sets]

    baseline = get_baseline(X, params, n_params, n_steps, strategies=strategies)

    return (
        X, y, baseline, trajectory, df_meta, label_to_driver,
        n_drivers, n_steps, n_params, scaling_params
    )