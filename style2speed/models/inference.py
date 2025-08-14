# style2speed/models/inferene.py

import glob
import os
import sys
from typing import Callable, Optional, Tuple, Union
import warnings

from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from style2speed.utils import download_file_if_colab
from style2speed.models.architectures import TelemetryDataset
from style2speed.models.training import train_model

from style2speed.config import NN_PARAMS, STATS, GROUP_COLS_ATTR

warnings.filterwarnings('ignore', message='.*deprecated*', module='torch')


def _run_inference_class(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    label_to_driver: dict
) -> pd.DataFrame:
    """Run driver classification model and summarize predictions for each lap.

    Args:
        model (nn.Module): Trained PyTorch model for F1 driver classification.
        device (torch.device): Computation device (CPU or GPU).
        dataloader (DataLoader): Batched telemetry data for inference.
        label_to_driver (dict): Maps class label → driver number (str).

    Returns:
        pd.DataFrame: One row per lap/segment with columns:
        - Dataset_ID: Index of the dataset in input.
        - DriverNumber: Ground truth driver (str).
        - PredictedDriverNumber: Model-predicted driver (str).
        - TrueLabel: Ground truth class label (int).
        - PredictedLabel: Predicted class label (int).
        - Correct: Whether prediction was correct (bool).
    """

    # Run model to generate driver predictions
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            targets.append(yb.cpu())

    # Summarize results
    predictions = torch.cat(predictions).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    correct = (predictions == targets)

    driver_numbers = [label_to_driver[t] for t in targets]
    predicted_driver_numbers = [label_to_driver[p] for p in predictions]

    # Move to dataframe (one row for each dataset or driver/lap combination)
    df_predictions = pd.DataFrame({
        'Dataset_ID': range(len(targets)),
        'DriverNumber': driver_numbers,
        'PredictedDriverNumber': predicted_driver_numbers,
        'TrueLabel': targets,
        'PredictedLabel': predictions,
        'Correct': correct,
    })

    # Add metadata
    df_predictions['ModelName'] = model.__class__.__name__
    df_predictions['Task'] = 'class'

    return df_predictions


def _run_inference_reg(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    y_labels: np.ndarray,
    label_to_driver: dict
) -> pd.DataFrame:
    """Run regression model and summarize predictions per lap or segment.

    Args:
        model (nn.Module): Trained PyTorch regression model.
        device (torch.device): Computation device (CPU or GPU).
        dataloader (DataLoader): Input data for inference.
        y_labels (np.ndarray): Ground-truth driver class labels
            (used for mapping to `'DriverNumber'`).
        label_to_driver (dict): Maps class label → driver number (str).

    Returns:
        pd.DataFrame: One row per dataset with:
        - Dataset_ID: Index of the input dataset.
        - DriverNumber: Corresponding driver number (str).
        - TrueTime: Ground truth lap or segment time (float).
        - PredictedTime: Model-predicted time (float).
        - Error: Prediction error (float).
    """

    # Run model of time regression
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for xb, yb in dataloader:  # iterate over batches
            xb = xb.to(device)
            pred = model(xb)
            predictions.append(pred.cpu())
            targets.append(yb.cpu())

    predictions = torch.cat(predictions).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    driver_numbers = [label_to_driver[label] for label in y_labels]

    df_predictions = pd.DataFrame({
        'Dataset_ID': range(len(targets)),
        'DriverNumber': driver_numbers,
        'TrueTime': targets,
        'PredictedTime': predictions,
        'Error': predictions - targets
    })

    # Add metadata
    df_predictions['ModelName'] = model.__class__.__name__
    df_predictions['Task'] = 'reg'

    return df_predictions


def _run_inference_multi(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    label_to_driver: dict
) -> pd.DataFrame:
    """Run multi-task model to predict driver and lap/segment time per dataset.

    Args:
        model (nn.Module): Trained PyTorch model for multi-task learning.
        device (`torch.device`): Computation device (CPU or GPU).
        dataloader (`DataLoader`): Input data.
        label_to_driver (dict): Maps class label → driver number (str).

    Returns:
        pd.DataFrame: One row per dataset with:
            - Dataset_ID: Dataset index.
            - DriverNumber: Ground truth driver number.
            - PredictedDriverNumber: Predicted driver number.
            - TrueLabel: Ground truth driver class label.
            - PredictedLabel: Predicted class label.
            - Correct: Boolean for classification correctness.
            - TrueTime: Ground truth lap/segment time.
            - PredictedTime: Predicted lap/segment time.
            - Error: Prediction error (PredictedTime - TrueTime).
    """

    model.eval()
    predictions_cls, predictions_reg = [], []
    targets_cls, targets_reg = [], []

    with torch.no_grad():
        for xb, (yb_cls, yb_reg) in dataloader:
            xb = xb.to(device)
            pred_cls, pred_reg = model(xb)
            predictions_cls.append(torch.argmax(pred_cls, dim=1).cpu())
            predictions_reg.append(pred_reg.cpu())
            targets_cls.append(yb_cls.cpu())
            targets_reg.append(yb_reg.cpu())

    predictions_cls = torch.cat(predictions_cls).cpu().numpy()
    predictions_reg = torch.cat(predictions_reg).cpu().numpy()
    targets_cls = torch.cat(targets_cls).cpu().numpy()
    targets_reg = torch.cat(targets_reg).cpu().numpy()
    correct = predictions_cls == targets_cls

    df_predictions = pd.DataFrame({
        'Dataset_ID': range(len(targets_cls)),
        'DriverNumber': [label_to_driver[t] for t in targets_cls],
        'PredictedDriverNumber': [label_to_driver[p] for p in predictions_cls],
        'TrueLabel': targets_cls,
        'PredictedLabel': predictions_cls,
        'Correct': correct,
        'TrueTime': targets_reg,
        'PredictedTime': predictions_reg,
        'Error': predictions_reg - targets_reg
    })

    # Add metadata
    df_predictions['ModelName'] = model.__class__.__name__
    df_predictions['Task'] = 'multi'

    return df_predictions


class OutputSelectorWrapper(nn.Module):
    def __init__(self, model: nn.Module, head: str = 'class'):
        super().__init__()
        self.model = model
        assert head in ['class', 'reg'], f'Unsupported head: {head}'
        self.head = head

    def forward(self, x):
        pred_class, pred_reg = self.model(x)
        return pred_class if self.head == 'class' else pred_reg
    

def _compute_attributions(
    model: nn.Module,
    device: torch.device,
    task: str,
    head: str,
    X: np.ndarray,
    y: list[np.ndarray],
    baseline: np.ndarray,
    trajectory: np.ndarray,
    label_to_driver: dict,
    params: Optional[list[str]] = None
) -> Tuple[pd.DataFrame, list]:
    """Compute parameter importance along the lap/segment for each dataset.

    Attribution scores are aggregated across all laps per driver, not per lap.

    Args:
        model (nn.Module): Trained PyTorch model.
        device (torch.device): Torch device.
        task (str): Task type. One of `{'class', 'reg', 'multi'}`.
        head (str): Model head to attribute (`'class'` or `'reg'`) for multi-task.
        X (np.ndarray): Telemetry input. Shape: [num_sets, num_steps, num_params].
        y (list[np.ndarray]): Ground truth targets [class_labels, times].
        baseline (np.ndarray): Baseline param values along circuit [num_steps, num_params].
        trajectory (np.ndarray): Circuit trace [num_steps, 3] (Distance, X, Y).
        label_to_driver (dict): Maps model label → driver number (str).
        params (list[str]): Telemetry parameter names in same order as X.

    Raises:
        ValueError: If params are not provided.
        ValueError: If unsorted task or head are passed.
        ValueError: If the length of positions does not match the number of
            steps in the input.

    Returns:
        Tuple[pd.DataFrame, list[float]]:
            - df_attr: Long-form DataFrame of param importance across datasets
                and steps along the circuit.
            - deltas: Convergence deltas from IntegratedGradients."""
    # Check inputs
    if params is None:
        raise ValueError('`params` must be explicitly provided to match X shape.')

    if task not in {'class', 'reg', 'multi'}:
        raise ValueError(f"Invalid task '{task}'")

    if task == 'class':
        head = 'class'
    elif task == 'reg':
        head = 'reg'
    elif task == 'multi':
        if head not in {'class', 'reg'}:
            raise ValueError(f"Invalid head '{head}'")

    model_name = model.__class__.__name__  # save original model name
    if task == 'multi':
        model = OutputSelectorWrapper(model, head=head)

    # Set baseline for regression task (shape: [1, num_steps, num_params])
    baseline_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(device) 

    # Attributions (one row each driver/lap/step/param combination)
    model.train()
    ig = IntegratedGradients(model)
    attr_rows, deltas = [], []
    n_datasets = len(y[0]) if isinstance(y, (list, tuple)) else len(y)

    for i in range(n_datasets):

        # Determine target for attribution
        if task == 'class' or (task == 'multi' and head == 'class'):
            target = int(y[0][i])
        elif task == 'reg' or (task == 'multi' and head == 'reg'):
            target = None  # correct for regression if model outputs a single value

        # Input sample (batch of 1)
        input_tensor = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
        attributions, delta = ig.attribute(input_tensor,
                                           baselines=baseline_tensor,
                                           target=target,
                                           return_convergence_delta=True)
        attr_values = attributions[0].cpu().numpy()  # shape [num_steps, num_params]

        if trajectory.shape[0] != attr_values.shape[0]:
            raise ValueError(f'Trajectory steps ({trajectory.shape[0]}) '
                             f'do not match attribution steps ({attr_values.shape[0]})')

        deltas.append(delta.item())
        driver = label_to_driver.get(y[0][i])

        for step in range(attr_values.shape[0]):
            for param_idx, param in enumerate(params):
                attr_rows.append({
                    'Dataset_ID': i,
                    'DriverNumber': driver,
                    'Step': step,
                    'Param': param,
                    'Importance': attr_values[step, param_idx]
                })

    if not attr_rows:
        raise ValueError('No attribution data collected. Check inputs.')

    df_attr = pd.DataFrame(attr_rows)

    # Attach trajectory info
    df_traj = pd.DataFrame({
        'Step': np.arange(len(trajectory)),
        'Distance': trajectory[:, 0],
        'X': trajectory[:, 1],
        'Y': trajectory[:, 2]
    })
    df_attr = df_attr.merge(df_traj, on='Step', how='left')
    df_attr = df_attr.drop(columns='Step')

    # Add metadata
    df_attr['ModelName'] = model_name
    df_attr['Task'] = task
    df_attr['Head'] = head

    return df_attr, deltas


def run_inference(
    model: nn.Module,
    task: str,
    X: np.ndarray,
    y: list[np.ndarray],
    baseline: np.ndarray,
    trajectory: np.ndarray,
    df_meta: pd.DataFrame,
    label_to_driver: dict,
    params: Optional[list[str]] = None,
    group_cols: Optional[list[str]] = None,
    stats: Optional[list[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run model on entire dataset and store per-lap predictions and attributions.

    Args:
        model (nn.Module): Trained PyTorch model.
        task (str): Task type. Options: `'class'`, `'reg'`, `'multi'`.
        X (np.ndarray): Telemetry data [num_sets, num_steps, num_params].
        y (list[np.ndarray]): Ground truth values [class_labels, lap_times].
        baseline (np.ndarray): Baseline param values along circuit [num_steps, num_params].
        trajectory (np.ndarray): Circuit trace [num_steps, 3] → Distance, X, Y.
        df_meta (pd.DataFrame): Metadata for each dataset along with segment/lap
            times.
        label_to_driver (dict): Maps predicted label → driver number.
        params (list[str], optional): Telemetry parameter names.
            Defaults to `NN_PARAMS` if None.
        group_cols (list[str], optional): Columns to group by df_attr to compute
            attribution statistics. Defaults to `GROUP_COLS_ATTR`.
        stats (list[str], optional): Statistics to compute for `df_attr`.
            Defaults to `STATS`.

    Raises:
        ValueError: If the number of params does not match the input shape.
        ValueError: If the task is not one of `{'class', 'reg', 'multi'}`.
        ValueError: If `y` length is not 2.
        ValueError: If the the number of datasets in `X` and `y` is not the same.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - df_pred: One row per dataset with model prediction summary.
            For `'class'` or `'multi'`: Dataset_ID, DriverNumber,
            PredictedDriverNumber, TrueLabel, PredictedLabel, Correct,
            Delta_class (along with df_meta columns).
            For `'reg'` or `'multi'`: Dataset_ID, DriverNumber, PredictedTime,
            TrueTime, Error, Delta_reg (along with df_meta columns).
        - df_attr: Long-form attribution table with param importance for each
            dataset and distance along the circuit. Columns: Dataset_ID,
            DriverNumber, Param, Distance, X, Y, Importance, Head (along with
            df_meta columns).
        - df_attr_stats: Long-form attribution table with param importance
            statistics for each data group and distance along the circuit.
            Columns: DriverNumber, Param, Distance, X, Y, along with stats
            columsn and df_meta columns.
    """
    # Handle optional arguments
    if params is None:
        params = NN_PARAMS
    if group_cols is None:
        group_cols = GROUP_COLS_ATTR
    if stats is None:
        stats = STATS

    # Check input
    if task not in {'class', 'reg', 'multi'}:
        raise ValueError(f"Invalid task '{task}'")
    if not isinstance(y, list) or len(y) != 2:
        raise ValueError('Expected y to be a list of [class_labels, lap_times]')
    if len(y[0]) != len(X) or len(y[1]) != len(X):
        raise ValueError('Mismatch between X and y lengths (num_datasets)')
    if len(params) != X.shape[2]:
        raise ValueError(f'Expected {len(params)} params, got {X.shape[2]}')

    # Set up to run the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if task == 'class':
        dataset = TelemetryDataset(X, y[0], task=task)
    elif task == 'reg':
        dataset = TelemetryDataset(X, y[1], task=task)
    elif task == 'multi':
        dataset = TelemetryDataset(X, y, task=task)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # --- Inference summary ---
    if task == 'class':
        df_pred = _run_inference_class(model, device, dataloader,
                                       label_to_driver)
    elif task == 'reg':
        df_pred = _run_inference_reg(model, device, dataloader, y[0],
                                     label_to_driver)
    else:  # 'multi'
        df_pred = _run_inference_multi(model, device, dataloader,
                                       label_to_driver)

    # --- Attribution ---
    if task in {'class', 'reg'}:
        df_attr, deltas = _compute_attributions(
            model, device, task, task, X, y, baseline,
            trajectory, label_to_driver, params
        )
        df_pred[f'Delta_{task}'] = deltas

    else:  # 'multi'
        df_attr_class, deltas_class = _compute_attributions(
            model, device, task, 'class', X, y, baseline,
            trajectory, label_to_driver, params
        )
        df_pred['Delta_class'] = deltas_class

        df_attr_reg, deltas_reg = _compute_attributions(
            model, device, task, 'reg', X, y, baseline,
            trajectory, label_to_driver, params
        )
        df_pred['Delta_reg'] = deltas_reg

        df_attr = pd.concat([df_attr_class, df_attr_reg], ignore_index=True)

    # Add back metadata
    df_pred = df_pred.merge(df_meta, on=['Dataset_ID', 'DriverNumber'],
                                  how='left')
    df_attr = df_attr.merge(df_meta, on=['Dataset_ID', 'DriverNumber'],
                            how='left')

    # Compute attribution statistics (Dataset_ID is dropped)
    df_attr_stats = (
        df_attr.groupby(group_cols)
        .agg({'Importance': stats})
        .reset_index()
    )
    df_attr_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                         for col in df_attr_stats.columns]

    return df_pred, df_attr, df_attr_stats


def run_nn_pipeline(
    model_class: Callable[..., nn.Module],
    task: str,
    X: np.ndarray,
    y: list[np.ndarray],
    baseline: np.ndarray,
    trajectory: np.ndarray,
    df_meta: pd.DataFrame,
    label_to_driver: dict,
    num_drivers: int,
    num_steps: int,
    num_params: int,
    f: float = 0.8,
    batch_size: int = 64,
    n_trials: int = 5,
    n_epoch: int = 50,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    weight_class: float = 1.0,
    weight_reg: float = 1.0,
    verbose: bool = False,
    enable_tqdm: bool = True,
    save_dir: str = '/content/',
    download: bool = False,
    include_lap_attr: bool = False,
    model_kwargs: dict = {}
) -> Tuple[nn.Module, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train a model and run inference on it.

    Args:
        model_class (Callable[..., nn.Module]): PyTorch model class.
        task (str): Task type. Options: `'class'`, `'reg'`, `'multi'`.
        X (np.ndarray): Telemetry data [num_sets, num_steps, num_params].
        y (list[np.ndarray]): Ground truth values [class_labels, lap_times].
        baseline (np.ndarray): Baseline param values along circuit [num_steps, num_params].
        trajectory (np.ndarray): Circuit path [num_steps, 3] (Distance, X, Y).
        df_meta (pd.DataFrame): Metadata for each dataset along with segment/lap
            times.
        label_to_driver (dict): Class index → driver number.
        num_drivers (int): Number of drivers (for classification head).
        num_steps (int): Telemetry time steps per input.
        num_params (int): Number of parameters per step.
        f, batch_size, n_trials, n_epoch, lr, early_stopping_patience:
            Training hyperparameters.
        weight_class, weight_reg (float): Loss weights (for 'multi' task).
        verbose (bool): Print progress if True.
        enable_tqdm (bool): If True, use tqdm to track progress of training
            model classes. Defaults to True.
        save_dir (str): Directory to save output to. Defaults to '/content/'.
        download (bool): If `True`, download all outputs (except for raw
            attribution values). Defaults to `False`.
        include_lap_attr (bool): If `True`, save and/or download attributions
            for each driver/lap/param combination. Defaults to `False`.
        model_kwargs (dict): Additional model kwargs.

    Returns:
        Tuple containing:
            - model (nn.Module): Trained model.
            - df_metrics (pd.DataFrame): Per-epoch training/validation metrics.
            - df_pred (pd.DataFrame): One-row-per-set model predictions.
            - df_attributions (pd.DataFrame): Parameter importances by step.

    Saves (and optionally downloads):
        - model (nn.Module): Trained model.
        - df_metrics (pd.DataFrame as csv): Per-epoch training/validation metrics
            for each model/task.
        - df_pred (pd.DataFrame as csv): One-row-per-set model predictions
            for each model/task.
        - df_attr (pd.DataFrame): Parameter importances along the circuit for
            each model/task (not aggregated over driver laps).
        - df_attr (pd.DataFrame as csv): Parameter importances along the circuit
            for each model/task (aggregated over driver laps).
        - df_best_trial (pd.DataFrame as csv): Best trial number per model/task.
    """

    assert len(y) == 2, 'y must contain [class_labels, times]'
    assert X.shape[0] == y[0].shape[0] == y[1].shape[0], 'Mismatch in dataset lengths'
    assert task in ['class', 'reg', 'multi'], f'Invalid task: {task}'

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if task == 'class':
        dataset = TelemetryDataset(X, y[0], task)
    elif task == 'reg':
        dataset = TelemetryDataset(X, y[1], task)
    else:
        dataset = TelemetryDataset(X, y, task)

    # Train the model
    model, model_kwargs_, df_metrics = train_model(
        dataset=dataset,
        model_class=model_class,
        task=task,
        num_drivers=num_drivers,
        num_steps=num_steps,
        num_params=num_params,
        f=f,
        batch_size=batch_size,
        n_trials=n_trials,
        n_epoch=n_epoch,
        lr=lr,
        early_stopping_patience=early_stopping_patience,
        weight_class=weight_class,
        weight_reg=weight_reg,
        verbose=verbose,
        enable_tqdm=enable_tqdm,
        model_kwargs=model_kwargs
    )

    # Run inference on the trained model
    df_pred, df_attr, df_attr_stats = run_inference(
        model=model,
        task=task,
        X=X,
        y=y,
        baseline=baseline,
        trajectory=trajectory,
        df_meta=df_meta,
        label_to_driver=label_to_driver,
        params=None
    )

    model_name = model_class.__name__

    # Save data
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = f'{model_name}_{task}'

        metrics_path = os.path.join(save_dir, f'{base_name}_metrics.csv')
        pred_path = os.path.join(save_dir, f'{base_name}_predictions.csv')
        attr_stats_path = os.path.join(save_dir, f'{base_name}_attr_stats.csv')
        model_path = os.path.join(save_dir, f'{base_name}.pt')

        df_metrics.reset_index(drop=True).to_csv(metrics_path, index=False)
        df_pred.reset_index(drop=True).to_csv(pred_path, index=False)
        df_attr_stats.reset_index(drop=True).to_csv(attr_stats_path, index=False)
        torch.save({
            'state_dict': model.state_dict(),
            'model_class': model_class.__name__,
            'model_kwargs': model_kwargs_,
        }, model_path)

        if download:
            download_file_if_colab(metrics_path)
            download_file_if_colab(pred_path)
            download_file_if_colab(attr_stats_path)
            download_file_if_colab(model_path)

        if include_lap_attr:
            attr_path = os.path.join(save_dir, f'{base_name}_attr.csv')
            df_attr.reset_index(drop=True).to_csv(attr_path, index=False)
            if download:
                download_file_if_colab(attr_path)

    return model, df_metrics, df_pred, df_attr, df_attr_stats


def combine_outputs(
    data_dir: str,
    file_endings: Optional[list[str]] = None,
    download: bool = False
):
    """Combine related csv files into a single master csv file.

    Args:
      data_dir (str): Directory with csv files.
      file_endings (list[str], optional): File endings to combine. Defaults to
          `['metrics.csv', 'predictions.csv', 'attr_stats.csv', 'best_trial.csv']`.
      download (bool): If `True`, download the combined files. Defaults to `False`.
    """

    if file_endings is None:
        file_endings = ['metrics.csv', 'predictions.csv', 'attr_stats.csv']

    for ending in file_endings:
        file_paths = glob.glob(os.path.join(data_dir, f'*{ending}'))
        dfs = []
        for path in file_paths:
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(data_dir, f'{ending}')
        df.reset_index(drop=True).to_csv(combined_path, index=False)
        if download:
            download_file_if_colab(combined_path)


def compare_models(
    model_classes: list[Callable[..., nn.Module]],
    tasks: list[str],
    X: np.ndarray,
    y: list[np.ndarray],
    baseline: np.ndarray,
    trajectory: np.ndarray,
    df_meta: pd.DataFrame,
    label_to_driver: dict,
    num_drivers: int,
    num_steps: int,
    num_params: int,
    f: float = 0.8,
    batch_size: int = 64,
    n_trials: int = 5,
    n_epoch: int = 30,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    weight_class: float = 1.0,
    weight_reg: float = 1.0,
    verbose: bool = False,
    enable_tqdm: bool = True,
    save_dir: str = '/content/',
    download: bool = False,
    include_lap_attr: bool = False,
    combine: bool = True,
    model_kwargs_master: dict = {}
) -> pd.DataFrame:
    """Train and compare multiple models across classification, regression,
    and multi-task settings.

    Args:
        model_classes (list): List of model classes to compare.
        tasks (list[str]): List of tasks to train the models for.
            Options: class, reg, multi.
        X (np.ndarray): Telemetry [num_sets, num_steps, num_params].
        y (list[np.ndarray]): [class_labels, lap_times], each [num_sets].
        baseline (np.ndarray): Baseline param values along circuit [num_steps, num_params].
        trajectory (np.ndarray): Circuit path [num_steps, 3] (Distance, X, Y).
        df_meta (pd.DataFrame): Metadata for each dataset along with segment/lap
            times.
        label_to_driver (dict): Class index → driver number.
        num_drivers (int): Number of drivers (for classification head).
        num_steps (int): Telemetry time steps per input.
        num_params (int): Number of parameters per step.
        f, batch_size, n_trials, n_epoch, lr, early_stopping_patience:
            Training hyperparameters.
        weight_class, weight_reg (float): Loss weights (for 'multi' task).
        verbose (bool): Print progress if `True`.
        enable_tqdm (bool): If `True`, use tqdm to track progress of training
            model classes. Defaults to `True` (however, `False` is passed to nested
            functions).
        save_dir (str): Directed to save output to. Defaults to '/content/'.
        download (bool): If `True`, download all outputs (except for raw attribution
            values). Defaults to `False`.
        include_lap_attr (bool): If `True`, save and/or download attribution values
            for each driver/lap/param combination. Defaults to `False`.
        combine (bool): If `True`, metrics, combine summaries, and attribution stats
            for different models into a single file. Defaults to `True`.
        model_kwargs_master (dict): Master dictionary of model kwargs where
            keys are model names and values are `model_kwargs`.

    Returns:
        None

    Saves (and optionally downloads):
        - model (nn.Module): Trained model.
        - df_metrics (pd.DataFrame as csv): Per-epoch training/validation metrics
            for each model/task.
        - df_summary (pd.DataFrame as csv): One-row-per-set model predictions
            for each model/task.
        - df_attr (pd.DataFrame): Parameter importances along the circuit for
            each model/task (not aggregated over driver laps).
        - df_attr (pd.DataFrame as csv): Parameter importances along the circuit
            for each model/task (aggregated over driver laps).
        - df_best_trial (pd.DataFrame as csv): Best trial number per model/task.
    """
    assert len(y) == 2, 'y must contain [class_labels, lap_times]'
    assert X.shape[0] == y[0].shape[0] == y[1].shape[0], 'Mismatch in dataset lengths'
    allowed_tasks = ['class', 'reg', 'multi']
    for task in tasks:
        assert task in allowed_tasks, f'Invalid task: {task}'

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if enable_tqdm:
        model_loop = tqdm(model_classes, desc='Training models',
                          leave='True', file=sys.stderr)
    else:
        model_loop = model_classes

    for model_class in model_loop:
        for task in tasks:

            if model_class.__name__ in model_kwargs_master.keys():
                model_kwargs = model_kwargs_master[model_class.__name__]
            else:
                model_kwargs = {}

            _, _, _, _, _ = run_nn_pipeline(
                model_class=model_class,
                task=task,
                X=X,
                y=y,
                baseline=baseline,
                trajectory=trajectory,
                df_meta=df_meta,
                label_to_driver=label_to_driver,
                num_drivers=num_drivers,
                num_steps=num_steps,
                num_params=num_params,
                f=f,
                batch_size=batch_size,
                n_trials=n_trials,
                n_epoch=n_epoch,
                lr=lr,
                early_stopping_patience=early_stopping_patience,
                weight_class=weight_class,
                weight_reg=weight_reg,
                verbose=verbose,
                enable_tqdm=False,
                save_dir=save_dir,
                download=download,
                include_lap_attr=include_lap_attr,
                model_kwargs=model_kwargs,
            )

    if combine:
        combine_outputs(save_dir)

    return