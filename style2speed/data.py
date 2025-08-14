# style2speed/data.py

import math
from typing import Optional, Tuple, Union

import fastf1 as ff1
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from style2speed.utils import download_file_if_colab

from style2speed.config import IMPORT_PARAMS
from style2speed.config import GROUP_COLS
from style2speed.config import PARAMS_CATEGORICAL, PARAMS_CONTINUOUS, PARAMS_TRAJECTORY
from style2speed.config import PARAMS_TO_DERIVE_CATEG, PARAMS_TO_DERIVE_CONT, PARAMS_TO_INVERSE
from style2speed.config import EPSILON


def compile_telemetry(
    session: ff1.core.Session,
    thresh_from_fastest: float = 6.0,
    min_data_points: int = 500,
    verbose: bool = False,
    save_path_tel: Optional[str] = '/content/',
    save_path_time: Optional[str] = '/content/',
    download = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compile telemetry data and lap times from a single session for each
    driver/lap combination.

    Exlude drivers who did not finish the race, laps with insufficient data,
    and laps that are uncharacteristically slow.

    Modify DRS values so that:
        1: DRS enabled
        0: DRS not enabled or not allowed

    Args:
        session (ff1.core.Session): Loaded F1 session from fastf1 module.
        thresh_from_fastest (float): laps slower than this number of seconds
            will be excluded from the dataset. Defaults to 6.0 seconds.
        min_data_points (int): Laps with less than this number of data points
            will be excluded from the dataset. Defaults to 500.
        verbose (bool): If True, print messages to console. Defaults to False.
        save_path_tel (str, optional): If provided, saves telemetry data to
            the specified CSV path. Defaults to `'/content/'`.
        save_path_time (str, optional): If provided, saves lap time data to
            the specified CSV path. Defaults to `'/content/'`.
        download (bool): If provided, CSV files are downloaded to local machine.

    Raises:
        RuntimeError: If failed to determine available parameters.

    Returns:
        pd.DataFrame with clean telemetry data for each driver/lap combination.
        pd.DataFrame with lap times for each driver/lap combination.
    """

    data = session.laps
    drivers = data['DriverNumber'].unique().tolist()
    total_laps = data['LapNumber'].max()
    fastest_time = data.pick_fastest()['LapTime'].total_seconds()

    # Probe available telemetry parameters
    params = IMPORT_PARAMS
    try:
        sample_data = data.pick_fastest().get_telemetry()
        available_params = sample_data.columns.intersection(params).tolist()
    except Exception as e:
        raise RuntimeError('Failed to determine available parameters') from e

    ordered_columns = ['DriverNumber', 'LapNumber', 'Timer'] + available_params

    dfs_tel = []
    dfs_laptime = []
    num_skipped_drivers = 0
    num_skipped_laps = 0

    for driver in tqdm(drivers, desc='Processing Drivers'):
        driver_data = data[data['DriverNumber'] == driver]
        total_driver_laps = int(driver_data['LapNumber'].max())

        # Skip driver if too few laps (not full race)
        if total_driver_laps < (total_laps - 2):
            num_skipped_drivers += 1
            if verbose:
                print(f'Driver {driver} skipped: did not finish the race.')
            continue

        for lap in range(1, total_driver_laps + 1):
            try:
                lap_data = driver_data[driver_data['LapNumber'] == lap].iloc[0]
                lap_time = lap_data['LapTime']
                if pd.isnull(lap_time):
                    num_skipped_laps += 1
                    continue
                lap_time_sec = lap_time.total_seconds()
                if lap_time_sec > (fastest_time + thresh_from_fastest):
                    num_skipped_laps += 1
                    if verbose:
                        print(f'Driver {driver} Lap {lap} skipped: slow lap.')
                    continue

                # Extract telemetry
                lap_telemetry = (
                    lap_data.get_telemetry().sort_values('Distance').copy()
                )

                if len(lap_telemetry) < min_data_points:
                    num_skipped_laps += 1
                    if verbose:
                        print(f'Driver {driver} Lap {lap} skipped: not enough data.')
                    continue

                # Add a timer that starts at zero for each lap
                lap_telemetry['Timer'] = (
                    lap_telemetry['SessionTime'] - lap_data['LapStartTime']
                ).dt.total_seconds()

                # Store telemetry data
                lap_telemetry['DriverNumber'] = driver
                lap_telemetry['LapNumber'] = lap
                dfs_tel.append(lap_telemetry[ordered_columns])

                # Store lap time data
                df_laptime = pd.DataFrame({
                    'DriverNumber': [driver],
                    'LapNumber': [lap],
                    'LapTime': [lap_time_sec]
                })
                dfs_laptime.append(df_laptime)

            except Exception as e:
                num_skipped_laps += 1
                if verbose:
                    print(f'Skipping Driver {driver} Lap {lap}: {e}.')
                continue

    df_tel = pd.concat(dfs_tel, ignore_index=True)
    df_laptimes = pd.concat(dfs_laptime, ignore_index=True)

    # Convert DriverNumber dtype from str to int
    df_tel = df_tel[df_tel['DriverNumber'].str.isnumeric()]
    df_tel['DriverNumber'] = df_tel['DriverNumber'].astype(int)
    df_laptimes = df_laptimes[df_laptimes['DriverNumber'].str.isnumeric()]
    df_laptimes['DriverNumber'] = df_laptimes['DriverNumber'].astype(int)

    # Customize nGear, DRS and following distance columns
    df_tel = df_tel.rename(columns={
        'nGear': 'Gear',
        'DRS': 'DRS_non_binary',
        'DistanceToDriverAhead': 'FollowingDistance'
    })
    df_tel['DRS'] = df_tel['DRS_non_binary'].map(lambda x: 1 if x == 12 else 0)
    df_tel['FollowingDistance'] = df_tel['FollowingDistance'].clip(upper=100)
    df_tel['FollowingDistance'] = df_tel['FollowingDistance'].fillna(100)

    df_tel = (
        df_tel.sort_values(by=['DriverNumber', 'LapNumber', 'Timer'])
        .reset_index(drop=True)
    )

    print(f'[INFO] {num_skipped_drivers} drivers skipped.')
    print(f'[INFO] For remaining drivers, {num_skipped_laps} total laps skipped.')

    # Add event metadata
    location = session.event.Location
    year = session.event.year
    event = session.session_info['Type']

    df_tel['Location'] = location
    df_tel['Year'] = year
    df_tel['Event'] = event
    df_laptimes['Location'] = location
    df_laptimes['Year'] = year
    df_laptimes['Event'] = event

    # Enforce order
    tel_order = GROUP_COLS + ['Timer', 'Distance']
    tel_order += [
        col for col in df_tel.columns
        if col not in GROUP_COLS + ['Timer', 'Distance']
    ]
    df_tel = df_tel[tel_order]
    df_laptimes = df_laptimes[GROUP_COLS + ['LapTime']]

    if save_path_tel:
        df_tel.reset_index(drop=True).to_csv(save_path_tel, index=False)
        if download:
            download_file_if_colab(save_path_tel)
    if save_path_time:
        df_laptimes.reset_index(drop=True).to_csv(save_path_time, index=False)
        if download:
            download_file_if_colab(save_path_time)

    return df_tel, df_laptimes


def interpolate_telemetry(
    df_tel: pd.DataFrame,
    group_cols: Optional[list[str]] = None,
    ref_col: str = 'Distance',
    fix_by: str = 'num_points',
    num_points: Union[str, int] = 'max',
    spacing: float = 5.0,
    min_points: int = 5,
    max_gap: float = 100.0,
    method_categorical: str = 'nearest',
    method_continuous: str = 'linear',
    method_trajectory: str = 'linear',
    method_other: str = 'nearest',
    start: Optional[float] = None,
    end: Optional[float] = None
) -> pd.DataFrame:
    """Interpolate telemetry data for all driver/lap combinations to a fixed
    number of evenly spaced samples between start and end reference points.

    Args:
        df_tel (pd.DataFrame): Input telemetry data with at least `'DriverNumber'`,
            `'LapNumber'`, and `ref_col` columns, plus telemetry parameters to
            interpolate.
        group_cols (list[str], optional): Columns to group by before
            interpolation. Defaults to `GROUP_COLS`.
        ref_col (str): Reference column for interpolation. One of: `'Distance'`,
            `'Timer'`. Defaults to `'Distance'`.
        fix_by (str): Mode to fix interpolation granularity. Options:
            - `'num_points'`: Fix the number of points per group.
            - `'spacing'`: Fix the spacing between points.
            Defaults to `'num_points'`.
        num_points (int or str): Target number of points per group, or
            `'min'`/`'max'` to auto-select based on group sizes.
            Used only if `fix_by == 'num_points'`. Defaults to `'max'`.
        spacing (float): Target spacing [in units of `ref_col`] between points.
            Used only if `fix_by = 'spacing'`. Defaults to 5.0.
        min_points (int): Minimum number of points required to interpolate
            a group. Groups with fewer points are skipped. Defaults to 5.
        max_gap (float): Maximum allowed gap [in units of  `ref_col`] between
            consecutive points in a group. Groups with gaps larger than this
            are skipped. Defaults to 100.0.
        method_categorical (str): Interpolation method for categorical parameters.
            One of: `'nearest'`, `'zero'`, `'linear'`, `'slinear'`, `'quadratic'`,
            `'cubic'`, `'previous'`, `'next'`. Defaults to `'nearest'`.
        method_continuous (str): Interpolation method for continuous parameters.
            Defaults to `'linear'`.
        method_trajectory (str): Interpolation method for trajectory parameters.
            Defaults to `'linear'`.
        method_other (str): Interpolation method for uncategorized parameters.
            Defaults to `'nearest'`.
        start (float, optional): Start value of `ref_col`. If None, uses the
            maximum of the per-group minimum values.
        end (float, optional): End value of `ref_col`. If None, uses the
            minimum of the per-group maximum values.

    Raises:
        ValueError: If `ref_col` is invalid.
        ValueError: If `group_cols` are missing from input dataframe.
        ValueError: If `fix_by` is invalid.
        ValueError: If `num_points` is an invalid string.

    Returns:
        pd.DataFrame: Interpolated telemetry data.
    """

    # Check input
    allowed_ref_columns = ['Distance', 'Timer']
    if ref_col not in allowed_ref_columns:
        raise ValueError(f'ref_col must be one of {allowed_ref_columns}.')

    # Handle optional arguments: columns to group by
    if group_cols is None:
        group_cols = GROUP_COLS
    if not all(col in df_tel.columns for col in group_cols):
        raise ValueError('Some group cols are missing from input dataframe')
    gb = df_tel.groupby(group_cols)

    # Handle optional arguments: start and end of ref_col values
    if start is None:
        start_positions = gb[ref_col].min()  # Starting points across laps
        start = start_positions.max()  # Latest starting point
    if end is None:
        end_positions = gb[ref_col].max()  # Last points across laps
        end = end_positions.min()  # Earliest ending point

    # Handle optional arguments: number of points to interpolate to
    if fix_by == 'num_points':
        if isinstance(num_points, str):
            sizes = gb.size()
            if num_points == 'max':
                num_points = sizes.max()
            elif num_points == 'min':
                num_points = sizes.min()
            else:
                raise ValueError(f'Invalid string for num_points: {num_points}')
    elif fix_by == 'spacing':
        num_points = math.ceil((end - start) / spacing)
    else:
        raise ValueError(
            f"Invalid fix_by value: {fix_by}."
            "Choose from 'num_points' or 'spacing'."
        )

    target_positions = np.linspace(start, end, num_points)

    # Params to interpolate & interpolation methods
    non_params = group_cols + [ref_col]
    params = [col for col in df_tel.columns if col not in non_params]
    method_map = {
        **{param: method_categorical for param in PARAMS_CATEGORICAL},
        **{param: method_continuous for param in PARAMS_CONTINUOUS},
        **{param: method_trajectory for param in PARAMS_TRAJECTORY},
    }

    # Interpolate each group
    dfs = []
    num_drops = 0
    for name, group in gb:

        group = group.sort_values(ref_col).drop_duplicates(subset=ref_col)

        # Check data quality
        if len(group) < min_points:
            num_drops += 1
            continue  # skip short or empty data segments
        if np.max(np.diff(group[ref_col])) > max_gap:
            num_drops += 1
            continue  # skip segments with large gaps between points

        # Interpolate each parameter
        interpolated = {}
        for param in params:

            method = method_map.get(param, method_other)
            f = interp1d(group[ref_col], group[param],
                         kind=method, fill_value='extrapolate')
            interpolated[param] = f(target_positions)

        df_i = pd.DataFrame(interpolated)
        df_i[ref_col] = target_positions

        for i, col in enumerate(group_cols):
            df_i[col] = name[i]

        dfs.append(df_i)

    if num_drops > 0:
        print(f'[INFO] Dropped {num_drops} laps due to insufficient data.')

    return pd.concat(dfs, ignore_index=True)


def derive_params(
    df_telemetry: pd.DataFrame,
    smooth: bool = True,
    sigma: Optional[float] = 1.0,
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Derive geometric and dynamic parameters along the circuit.

    Args:
        df_telemetry (pd.DataFrame): Interpolated telemetry data for each
            driver/lap combination.
        smooth (bool): Whether to apply Gaussian smoothing to XYZ positions.
            Defaults to True.
        sigma (float): Standard deviation for Gaussian smoothing if enabled.
            Defaults to 1.0.
        group_cols (list[str], optional): Columns to group by before
            deriving parameters. Defaults to `GROUP_COLS`.

    Raises:
        ValueError: If input DataFrame is missing required columns.

    Returns:
        pd.DataFrame: Input telemetry data with additional columns:
            - Derivatives (e.g., `'dSpeed'`, `'dThrottle'`, `'GearChange'`)
            - Additional trajectory parameters (`'Curvature'`, `'Radius'`)
            - Inverted parameters (e.g., `'FollowingDistanceInv'`)
    """

    # Check input
    if group_cols is None:
        group_cols = GROUP_COLS
    required_cols = group_cols + ['Distance', 'Timer']
    if not all(col in df_telemetry.columns for col in required_cols):
        raise ValueError('Input DataFrame is missing required columns.')
    merge_cols = group_cols + ['Distance']

    # Derive parameters for each group
    dfs = []
    gb = df_telemetry.groupby(group_cols)
    for name, group in gb:

        # Extract and prepare
        group = group.sort_values('Distance').copy()

        # Continuous params: derive time-based changes
        derivatives_cont = {
            f'd{param}': np.gradient(group[param].values, group['Timer'].values)
            for param in PARAMS_TO_DERIVE_CONT
        }

        # Categorical params: derive stepwise changes
        derivatives_categ = {
            f'{param}Change': np.diff(group[param].values, prepend=group[param].values[0])
            for param in PARAMS_TO_DERIVE_CATEG
        }

        # Parameters to inverse
        inverses = {}
        for param in PARAMS_TO_INVERSE:
            if param in group.columns:
                values = group[param].values
                inv = 1 / (values + EPSILON)
                inverses[f'{param}Inv'] = np.clip(inv, 0, None)
            else:
                inverses[f'{param}Inv'] = np.full(len(group), np.nan)

        # Trajectory params: derive curvature, radius, & lateral acceleration
        x, y, z = group['X'].values, group['Y'].values, group['Z'].values
        positions = np.stack([x, y, z], axis=1)

        if smooth:
            positions = gaussian_filter1d(positions, sigma=sigma,
                                          axis=0, mode='wrap')

        dp = np.gradient(positions, axis=0)
        ddp = np.gradient(dp, axis=0)

        dp_norm = np.linalg.norm(dp, axis=1)
        cross = np.cross(dp, ddp)
        cross_norm = np.linalg.norm(cross, axis=1)

        curvature = cross_norm / (dp_norm**3 + EPSILON)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        radius = 1 / (curvature + EPSILON)
        speed = group['Speed'].values
        lateral_acc = speed ** 2 * curvature

        # Move to dataframe
        df_derived = pd.DataFrame({
            'Distance': group['Distance'].values,
            **derivatives_cont,
            **derivatives_categ,
            **inverses,
            'Curvature': curvature,
            'Radius': radius,
            'LatAcc': lateral_acc,
        })
        for i, col in enumerate(group_cols):
            df_derived[col] = name[i]

        dfs.append(df_derived)

    df_derived_master = pd.concat(dfs, ignore_index=True)
    return df_telemetry.merge(df_derived_master, on=merge_cols)