# style2speed/stats.py

from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests

from style2speed.config import GROUP_COLS, STATS_PARAMS, STATS


def get_circuit_dominance(
    df_tel: pd.DataFrame,
    driver: int,
    ref_driver: int,
    method: str = "avg_speed",          # or 'fastest_lap'
    df_laptimes: Optional[pd.DataFrame] = None,
    eps: float = 0.0                    # tolerance in m/s to ignore tiny diffs
) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    """
    Compute driver dominance versus a reference driver along circuit distance.

    This function compares the speed profiles of two drivers and determines
    which driver is dominant (i.e., faster) at each point along the circuit.
    Dominance can be computed either from average speed across all laps
    or from the fastest lap of each driver.

    Args:
        df_tel (pd.DataFrame): 
            Telemetry data containing at least 'DriverNumber', 'Distance', 
            and 'Speed' columns. Optionally includes 'LapNumber' if 
            `method='fastest_lap'`.
        driver (int): 
            Driver number of interest for whom dominance is evaluated.
        ref_driver (int): 
            Reference driver number to compare against.
        method (str): 
            Circuit dominance evaluation method, either `'avg_speed'` or `'fastest_lap'`.
            - `'avg_speed'`: Dominance is evaluated based on average speed across all laps.
            - `'fastest_lap'`: Dominance is evaluated using the fastest lap of each driver.
        df_laptimes (pd.DataFrame, optional): 
            Data frame with lap times across drivers and laps. Must include
            'DriverNumber', 'LapNumber', and 'LapTime' columns.
            Required if `method == 'fastest_lap'`.
        eps (float, optional): 
            Difference in speed (m/s) regarded as a tie, i.e. 
            |speed_driver − speed_ref| ≤ eps → dominance = 0.5. Default is 0.0.

    Raises:
        ValueError: If an invalid method is provided.
        ValueError: If `df_laptimes` is not provided for `'fastest_lap'` method.
        ValueError: If `df_laptimes` is missing rows for the input drivers.
        ValueError: If `df_tel` is missing speed values for the input drivers.
        ValueError: If the input drivers lack sufficient overlapping 
            'Distance' values for comparison.

    Returns:
        dominance (pd.Series): 
            Series indexed by 'Distance' (meters). Values:
            - 1.0 → `driver` dominates (`speed_driver - speed_ref > eps`)
            - 0.0 → `ref_driver` dominates (`speed_driver - speed_ref < -eps`)
            - 0.5 → tie (`|speed_driver - speed_ref| ≤ eps`)
        dlimits (list[tuple[float, float]]): 
            List of maximal contiguous distance intervals (start_m, end_m) 
            where the dominance label is constant.
    """
    
    # --- Check inputs ---
    if method not in {"avg_speed", "fastest_lap"}:
        raise ValueError('method must be "avg_speed" or "fastest_lap".')
    if method == "fastest_lap" and df_laptimes is None:
        raise ValueError("Provide df_laptimes for fastest_lap method.")
    
    # --- Select telemetry rows to compare ---
    if method == "avg_speed":
        df = df_tel.loc[
            df_tel["DriverNumber"].isin([driver, ref_driver]),
            ["DriverNumber", "Distance", "Speed"]
        ].copy()

    else:  # fastest_lap
        def _fastest_lap_num(dn: int) -> int:
            sub = df_laptimes.loc[df_laptimes["DriverNumber"] == dn]
            if sub.empty:
                raise ValueError(f"No df_laptimes rows for driver {dn}.")
            row = sub.sort_values("LapTime", ascending=True).iloc[0]
            return int(row["LapNumber"])

        fast_lap = _fastest_lap_num(driver)
        fast_lap_ref = _fastest_lap_num(ref_driver)

        df = df_tel.loc[
            ((df_tel["DriverNumber"] == driver) & (df_tel["LapNumber"] == fast_lap)) |
            ((df_tel["DriverNumber"] == ref_driver) & (df_tel["LapNumber"] == fast_lap_ref)),
            ["DriverNumber", "Distance", "Speed"]
        ].copy()

    # --- Aggregate by distance per driver ---
    df_agg = (
        df.groupby(["DriverNumber", "Distance"], as_index=False)["Speed"].mean()
    )
    df_agg = df_agg.sort_values(["Distance", "DriverNumber"])  # ensure sorted

    # --- Pivot wide & only keep rows where both drivers have a value
    df_wide = df_agg.pivot(index="Distance", columns="DriverNumber", values="Speed")

    if driver not in df_wide.columns or ref_driver not in df_wide.columns:
        raise ValueError("One of the drivers has no speed data after aggregation.")
    
    df_wide = df_wide[[driver, ref_driver]].dropna().sort_index()

    if df_wide.empty or len(df_wide) < 2:
        raise ValueError("Insufficient overlapping distance samples for comparison.")
    
    # --- Dominance calculation ---
    # 1: driver is faster
    # 0: ref_driver is faster
    # 0.5: tie (speed difference is within eps)
    delta = df_wide[driver] - df_wide[ref_driver]  # >0 => driver faster
    dom = pd.Series(np.where(delta > eps, 1.0, np.where(delta < -eps, 0.0, 0.5)),
                    index=df_wide.index, name="Dominance")
    
    # --- Find change points (run-lengths over dominance) ---
    # Change point is where dominance value differs from previous
    # Treat ties (0.5) as their own label; they’ll form their own segments.
    dvals = dom.values
    change_idx = np.flatnonzero(np.diff(dvals)) + 1
    split_points = np.r_[0, change_idx, (len(dom) - 1)]  # add start and end

    # --- Construct circuit segments based on dominance
    distances = dom.index.values
    dlimits: List[Tuple[float, float]] = []
    for i in range(len(split_points) - 1):
        start = distances[split_points[i]]
        end = distances[split_points[i + 1]]
        if end < start:
            continue
        dlimits.append((float(start), float(end)))

    return dom, dlimits


def calc_param_stats(
    df: pd.DataFrame,
    segment_length: Optional[float] = 250.0,
    dlimits: Optional[list[Tuple[float, float]]] = None,
    include_full_lap: bool = False,
    group_cols: Optional[list[str]] = None,
    params: Optional[list[str]] = None,
    stats: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute telemetry param stats for each driver/lap/segment combination.

    By default, compute statistics for 250m long segments of the circuit.
    Alternatively, if segment_length is not provided and dlimits is provided,
    compute statistics for specified distance limits of the circuit.
    Optionally, include full lap as segment.
    If neither segment_length nor dlimits is provided, compute statistics
    for the entire circuit only.

    Args:
        df (pd.DataFrame): telemetry data for each driver/lap combination.
        segment_length (float, optional): If provided, divide the entire circuit
            into segments of the specified length and compute statistics
            for each segment. Defaults to 250.0 m.
        dlimits (list[tuple[float, float]], optional): If provided, compute
            parameter statistics for specified distance limits of the circuit.
            Defaults to None.
        include_full_lap (bool, optional): If `True`, include full lap statistics.
            Defaults to `False`. If neither `segment_length` nor `dlimits` is provided,
            compute full lap statistics even if `include_full_lap` is `False`.
        group_cols (list[str], optional): columns to group by before calculating
            param stats (excluding columsn that denote segment - that are
            created in this script and included automatically).
            Defaults to `GROUP_COLS`.
        params (list[str], optional): telemetry parameters to evaluate.
            Defaults to `STATS_PARAMS`.
        stats (list[str], optional): statistics to compute.
            Defaults to `STATS`.
            Note applicable to lap/segment time.

    Raises:
        ValueError: If neither `segment_length` nor `dlimits` is provided.
        ValueError: If both `segment_length` and `dlimits` are provided.

    Returns:
        pd.DataFrame with various telemetry parameter (`'Speed'`, etc.) statistics
        (mean, min, max, etc.) for each driver/lap combination
    """

    # Handle optional arguments
    if params is None:
        params = STATS_PARAMS
    params = [p for p in params if p in df.columns]
    if stats is None:
        stats = STATS
    if group_cols is None:
        group_cols = GROUP_COLS
    group_cols_ = group_cols + ['Segment', 'Segment_Start', 'Segment_End']
    required_cols = group_cols + ['Distance', 'Timer'] + params

    # Confirm that input df has all required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Define circuit segments to calculate param stats over
    total_distance = df['Distance'].max()
    if segment_length is not None:
        if dlimits is not None:
            raise ValueError('Provide `dlimits` or `segment_length`, not both.')
        edges = np.arange(0, total_distance, segment_length)
        if edges[-1] < total_distance:
            edges = np.append(edges, total_distance)
    else:
        if dlimits is None:
            dlimits = [(0.0, float(total_distance))]
        elif include_full_lap:
            dlimits = [(0.0, float(total_distance))] + dlimits
        edges = list({bound for dl in dlimits for bound in dl})
        edges = sorted(set(edges))

    # Assign segment based on position along track distance
    df = df.copy()  # so as not to overwrite original
    label_bounds = {
        f'{int(start)}-{int(end)}': (start, end)
        for start, end in zip(edges[:-1], edges[1:])
    }
    labels = list(label_bounds.keys())
    df['Segment'] = pd.cut(df['Distance'], bins=edges, labels=labels,
                           include_lowest=True, right=True)
    df['Segment_Start'] = df['Segment'].map(
        lambda x: label_bounds.get(x, (np.nan, np.nan))[0]).astype(float)
    df['Segment_End'] = df['Segment'].map(
        lambda x: label_bounds.get(x, (np.nan, np.nan))[1]).astype(float)

    # Compute segment statistics
    gb = df.groupby(group_cols_, observed=True)

    # Param stats
    df_stats = (
        gb[params].agg(stats).reset_index()
    )
    # Segment time
    df_times = (
        gb['Timer'].agg(lambda x: x.max() - x.min())
        .reset_index(name='SegmentTime')
    )
    # Delay from best time
    best_times = (
        df_times.groupby('Segment', observed=True)['SegmentTime']
        .transform('min')
    )
    df_times['DeltaFromBest'] = df_times['SegmentTime'] - best_times

    # Make df_times columns consistent with df_stats columns
    df_times = df_times.rename(
        columns={'SegmentTime': ('Timer', 'SegmentTime'),
                 'DeltaFromBest': ('Timer', 'DeltaFromBest')}
    )
    df_times.columns = [
        (col if isinstance(col, tuple) else (col, ''))
        for col in df_times.columns
    ]
    df_times.columns = pd.MultiIndex.from_tuples(df_times.columns)

    # Merge the two data frames
    merge_keys = [(col, '') for col in group_cols_]

    df_stats_all = df_stats.merge(df_times, on=merge_keys)

    # Enforce column order
    other_cols = [c for c in df_stats_all.columns if c not in merge_keys]
    df_stats_combined = df_stats_all[merge_keys + sorted(other_cols)]

    # Flatten metadata columns
    df_stats_all.columns = [
        col[0] if col in merge_keys else col
        for col in df_stats_all.columns
    ]

    return df_stats_all


def calc_pval_rmsse(
    data: list[np.ndarray],
    test: str = 'kruskal',
) -> Tuple[float, float]:
    """Calculate p-value and effect size for input data.

    To calculate pvalue, use anova, welch, or kruskal test.

    To evaluate effect size, use root mean square standardized effect (RMSSE),
    which quantifies between-group variability relative to the overall standard
    deviation. This is conceptually similar to Cohen's f, extending Cohen's d
    to multiple groups.

    Args:
        data (list[np.ndarray]): dataset of interest, with each np.ndarray in
            the list corresponding to a single data group.
        test (str, optional): the test to use for statistical significance
            evaluation.
            Defaults to `'kruskal'`. Other options are `'anova'` and `'welch'`.

            - anova: assumes independent, normally distributed, homoscedastic
              (equal variance), and continuous data.
            - welch: assumes independent, normally distributed data
              with **unequal** variances.
            - kruskal: assumes independent, ordinal or continuous data,
              with **similarly shaped** group distributions (non-parametric).

    Raises:
        ValueError: If a valid test was not selected.
        ValueError: If data properties are invalid.

    Returns:
        Tuple(float, float): p-value and effect size.
    """

    # Confirm that a valid test was selected
    valid_tests = {'anova', 'kruskal', 'welch'}
    if test not in valid_tests:
       raise ValueError(f'Invalid test: {test}. Choose from {valid_tests}.')

    # Confirm data properties
    if any(np.isnan(group).any() for group in data):
        raise ValueError('NaN values present')
    if len(data) < 2:
        raise ValueError('Less than two groups')
    if any(group.size == 0 for group in data):
        raise ValueError('Empty group')
    if all(np.std(group) == 0 for group in data):
        raise ValueError('All groups are constant')

    # Statistical significance
    if test == 'anova':
        _, pval = f_oneway(*data)
    elif test == 'kruskal':
        _, pval = kruskal(*data)
    elif test == 'welch':
        y = np.concatenate(data) # Flatten data and build group labels
        groups = np.concatenate([[i] * len(group) for i, group in enumerate(data)])
        result = anova_oneway(y, groups=groups, use_var='unequal')
        pval = result.pvalue

    # Root mean square standardized effect (RMSSE)
    means = np.array([np.mean(g) for g in data])
    data_flat = np.concatenate(data)
    grand_mean = np.mean(data_flat)
    grand_std = np.std(data_flat, ddof=1)
    if grand_std == 0:
        raise ValueError('Pooled standard deviation is zero; '
                         'effect size is undefined.')
    dof = len(data) - 1  # Degrees of freedom
    # Mean square between (MSB) — variance of group means around the grand mean
    msb = np.sum((means - grand_mean)**2) / dof
    rmsse = np.sqrt(msb) / grand_std

    return pval, rmsse


def calc_significance(
    df_stats: pd.DataFrame,
    test: str = 'kruskal',
    pval_correction: str = 'fdr_bh',
    params: Optional[list[str]] = None,
    stats: Optional[list[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute significance of driver differences across available track segments.

    For each track segment, evaluate significance based on various
    parameter/statistic combinations across driver laps.

    Args:
        df_stats (pd.DataFrame): DataFrame with MultiIndex columns representing
            telemetry parameter statistics for each driver/lap/segment.
            Metadata columns are expected to be single-level (e.g.
            `'DriverNumber'`, `'LapNumber'`, `'Segment'`).
            Param stats columns are expected to be MultiIndex (e.g.
            `('Speed', 'mean')`, `('Speed', 'max')`).
            `'Segment'` column must be formatted as 'start-end' (e.g., `'0-250'`).
        test (str): the test to use for statistical significance evaluation.
            Defaults to `'kruskal'`. Other options are `'anova'` and `'welch'`.
        pval_correction (str): method for multiple comparisons correction.
            Defaults to `'fdr_bh'`.
        params (list[str], optional): telemetry parameters to evaluate.
            Defaults to `STATS_PARAMS`.
        stats  (list[str], optional): parameter statistics to evaluate.
            Defaults to `STATS`.

    Raises:
        ValueError: If a valid test was not selected.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] where
        - the 1st DataFrame contains p- and rmsse-values for each
            Segment/Parameter/Statistic combination.
        - the 2nd DataFrame contains skipped combinations for which
            p-val and rmsse could not be calculated plus with reason for skipping.
    """
    def try_append_sig_stats(gb, param, stat, segment_info):
        """Try calculating pval and rmsse for each dataset & format data output.
        This uses 'test', which is defined in the outer function"""
        try:
            data = [group[(param, stat)].values for _, group in gb]
            pval, rmsse = calc_pval_rmsse(data, test)
            return {
                **segment_info,
                'Parameter': param,
                'Statistic': stat,
                'pval': pval,
                'rmsse': rmsse
            }, None
        except ValueError as e:
            return None, {
                **segment_info,
                'Parameter': param,
                'Statistic': stat,
                'Reason': str(e)
            }

    # Confirm that a valid test was selected
    valid_tests = {'anova', 'welch', 'kruskal'}
    if test not in valid_tests:
        raise ValueError(f'Invalid test: {test}. Choose from {valid_tests}.')

    # Define parameters and statistics to consider
    if params is None:
        params = STATS_PARAMS
    if stats is None:
        stats = STATS

    results = []
    skipped = []

    df_stats['Segment'] = df_stats['Segment'].astype(str)
    segments = df_stats['Segment'].unique()
    for segment in segments:

        try:
            segment_start, segment_end = map(float, segment.split('-'))
        except ValueError:
            warnings.warn(f"Invalid 'Segment' format: {segment}")
            continue
        segment_info = {
            'Segment': segment,
            'Segment_Start': segment_start,
            'Segment_End': segment_end
        }

        df_segment = df_stats[df_stats['Segment'] == segment]
        gb = df_segment.groupby('DriverNumber')

        # Significance of different param/stat combinations
        for param in params:
            for stat in stats:

                res, skip = try_append_sig_stats(gb, param, stat, segment_info)
                if res:
                    results.append(res)
                else:
                    skipped.append(skip)

        # Segment time significance
        res, skip = try_append_sig_stats(gb, 'Timer', 'SegmentTime', segment_info)
        if res:
            results.append(res)
        else:
            skipped.append(skip)

        # Delay to best time significance
        res, skip = try_append_sig_stats(gb, 'Timer', 'DeltaFromBest', segment_info)
        if res:
            results.append(res)
        else:
            skipped.append(skip)

    df_results = pd.DataFrame(results)

    # Account for multiple comparisons
    pvals_raw = df_results['pval'].values
    _, pvals_corrected, _, _ = multipletests(pvals_raw, method=pval_correction)
    df_results['pval_adj'] = pvals_corrected

    # Add testing metadata
    df_results['Test'] = test
    df_results['CorrectionMethod'] = pval_correction

    # Tidy up the output
    df_results = df_results[[
        'Segment', 'Segment_Start', 'Segment_End', 'Parameter', 'Statistic',
        'rmsse', 'pval', 'pval_adj', 'Test', 'CorrectionMethod'
    ]]
    df_results = df_results.sort_values(
        ['Segment_Start', 'Parameter', 'Statistic']
    )

    if len(skipped) > 0:
        df_skipped = pd.DataFrame(skipped)
        df_skipped = df_skipped.sort_values(
            ['Segment', 'Parameter', 'Statistic']
        )
    else:
        df_skipped = None

    return df_results, df_skipped