# style2speed/models/training.py

import copy
from typing import Callable, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

warnings.filterwarnings('ignore', message='.*deprecated*', module='torch')


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    weight_class: float = 1.0,
    weight_reg: float = 1.0,
    loss_fn_class: Optional[Callable] = None,
    loss_fn_reg: Optional[Callable] = None
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train a single epoch of a PyTorch model and return performance metrics.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        valid_dataloader (DataLoader): Dataloader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        device (torch.device): Device to run the model on (e.g., cuda or cpu).
        task (str): Task type. Options: `'class'`, `'reg'`, `'multi'`.
        weight_class: Classification loss weight (for `'multi'` task).
        weight_reg: Regression loss weight (for `'multi'` task).
        loss_fn_class (Callable, optional): Loss function for classification.
            Required for `'class'`/`'multi'`. Defaults to `nn.CrossEntropyLoss()`.
        loss_fn_reg (Callable, optional): Loss function for regression.
            Required for `'reg'`/`'multi'`. Defaults to `nn.MSELoss`.

    Raises:
        ValueError: For invalid task, missing loss functions, or weight errors.
        ValueError: For `'multi'` task, if `total_weight` is 0.
        ValueError: For `'multi'` task, if `yb` is not a tuple with 2 elements.

    Returns:
        model (nn.Module): The trained model.
        df_metrics_epoch (pd.DataFrame): Epoch metrics (loss, accuracy, MSE).
    """

    # --- Input validation ---
    valid_tasks = {'class', 'reg', 'multi'}
    if task not in valid_tasks:
        raise ValueError(f'Invalid task type: {task}. Choose from {valid_tasks}.')
    if task in {'class', 'multi'} and loss_fn_class is None:
        loss_fn_class = nn.CrossEntropyLoss()
    if task in {'reg', 'multi'} and loss_fn_reg is None:
        loss_fn_reg = nn.MSELoss()
    if task == 'multi' and (weight_class + weight_reg == 0):
        raise ValueError('Weights must sum to a positive value.')

    # Normalize weights for multi-task loss
    total_weight = weight_class + weight_reg
    if total_weight == 0:
        raise ValueError('Sum of weight_class and weight_reg must be > 0.')
    weight_class /= total_weight
    weight_reg /= total_weight

    # TODO: torch.cuda.amp.autocast() for model(xb)?

    # --- Training loop ---
    model.train()
    running_train_loss = 0.0

    for xb, yb in train_dataloader:
        xb = xb.to(device)
        if isinstance(yb, (list, tuple)):
            yb = [y.to(device) for y in yb]
        else:
            yb = yb.to(device)

        if task == 'class':
            pred = model(xb)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn_class(pred, yb)
        elif task == 'reg':
            pred = model(xb)
            loss = loss_fn_reg(pred, yb.float())
        elif task == 'multi':
            if not isinstance(yb, (list, tuple)) or len(yb) != 2:
                raise ValueError("Expected yb to be a tuple (class, reg) for 'multi' task.")
            y_class, y_reg = yb
            pred_class, pred_reg = model(xb)
            loss_class = loss_fn_class(pred_class, y_class)
            loss_reg = loss_fn_reg(pred_reg, y_reg.float())
            loss = weight_class * loss_class + weight_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        # TODO: Optional: clip gradients for LSTM
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_train_loss += loss.item() * xb.size(0)

    # --- Evaluation loop ---
    model.eval()
    running_valid_loss = 0.0
    correct, sq_error, total = 0, 0.0, 0

    with torch.no_grad():
        for xb, yb in valid_dataloader:
            xb = xb.to(device)
            if isinstance(yb, (list, tuple)):
                yb = [y.to(device) for y in yb]
            else:
                yb = yb.to(device)

            if task == 'class':
                pred = model(xb)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = loss_fn_class(pred, yb)
                predicted = pred.argmax(dim=1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)
            elif task == 'reg':
                pred = model(xb)
                loss = loss_fn_reg(pred, yb.float())
                sq_error += torch.sum((pred - yb.float()) ** 2).item()
                total += yb.size(0)
            elif task == 'multi':
                y_class, y_reg = yb
                pred_class, pred_reg = model(xb)
                loss_class = loss_fn_class(pred_class, y_class)
                loss_reg = loss_fn_reg(pred_reg, y_reg.float())
                loss = weight_class * loss_class + weight_reg * loss_reg
                predicted = pred_class.argmax(dim=1)
                correct += (predicted == y_class).sum().item()
                sq_error += torch.sum((pred_reg - y_reg.float()) ** 2).item()
                total += y_class.size(0)

            running_valid_loss += loss.item() * xb.size(0)

    # --- Epoch metrics ---
    n_train = len(train_dataloader.dataset)
    n_valid = len(valid_dataloader.dataset)

    train_loss = running_train_loss / n_train if n_train else float("nan")
    valid_loss = running_valid_loss / n_valid if n_valid else float("nan")

    if total == 0:
        acc, mse = float("nan"), float("nan")
    elif task == 'class':
        acc, mse = correct / total, float("nan")
    elif task == 'reg':
        acc, mse = float("nan"), sq_error / total
    else:  # multi
        acc = correct / total
        mse = sq_error / total

    df_metrics_epoch = pd.DataFrame({
        'TrainLoss': [train_loss],
        'ValidLoss': [valid_loss],
        'Accuracy': [acc],
        'MSE': [mse]
    })

    return model, df_metrics_epoch


def train_one_trial(
    model_class: Callable[..., nn.Module],
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    device: torch.device,
    task: str,
    n_epoch: int = 30,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    verbose: bool = False,
    weight_class: float = 1.0,
    weight_reg: float = 1.0,
    loss_fn_class: Optional[Callable] = None,
    loss_fn_reg: Optional[Callable] = None,
    model_kwargs: Optional[dict] = None
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train a PyTorch model for a single trial and return performance metrics.

    Args:
        model_class (Callable[..., nn.Module]): Model constructor
            (e.g., `TelemetryModelCNN`).
        train_dataloader (DataLoader): Dataloader for training data.
        valid_dataloader (DataLoader): Dataloader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        device (torch.device): Device to run the model on (e.g., cuda or cpu).
        task (str): Task type. Options: `'class'`, `'reg'`, `'multi'`.
        n_epoch (int): Number of training epochs. Defaults to 30.
        lr (float): Learning rate. Defaults to 1e-3.
        early_stopping_patience (int): Number of epochs to wait without
            improvement. Defaults to 10.
        verbose (bool): Whether to print training progress.
        weight_class: Classification loss weight (for `'multi'` task).
        weight_reg: Regression loss weight (for `'multi'` task).
        loss_fn_class (Callable, optional): Loss function for classification.
            Required for `'class'`/`'multi'`. Defaults to `nn.CrossEntropyLoss()`.
        loss_fn_reg (Callable, optional): Loss function for regression.
            Required for `'reg'`/`'multi'`. Defaults to `nn.MSELoss`.
        model_kwargs: Optional kwargs to pass to `model_class`.

    Raises:
        ValueError: For invalid task, missing loss functions, or weight errors.

    Returns:
        model: The trained model.
        df_metrics_trial: DataFrame with loss/accuracy/MSE for each epoch.
    """
    if task not in {'class', 'reg', 'multi'}:
        raise ValueError(f"Invalid task: {task}")
    if task in {'class', 'multi'} and loss_fn_class is None:
        loss_fn_class = nn.CrossEntropyLoss()
    if task in {'reg', 'multi'} and loss_fn_reg is None:
        loss_fn_reg = nn.MSELoss()

    # nn model
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler_mode = 'min' if task == 'reg' else 'max'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.5, patience=3
    )

    # Train the model
    best_acc, best_mse = 0.0, float('inf')
    epochs_since_improvement = 0
    metrics_per_epoch = []

    for epoch in range(n_epoch):

        model, df_epoch = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer,
            device=device,
            task=task,
            weight_class=weight_class,
            weight_reg=weight_reg,
            loss_fn_class=loss_fn_class,
            loss_fn_reg=loss_fn_reg
        )
        df_epoch['Epoch'] = epoch
        metrics_per_epoch.append(df_epoch)

        acc = df_epoch['Accuracy'].values[0]
        mse = df_epoch['MSE'].values[0]

        if verbose:
            print(f'Epoch {epoch+1}/{n_epoch} complete:')
            if task in {'class', 'multi'}:
                print(f'  Val Accuracy: {acc:.2%}')
            if task in {'reg', 'multi'}:
                print(f'  Val MSE: {mse:.4f}')

        # Adapt learning rate
        scheduler.step(mse if task == 'reg' else acc)

        # Exit early if accuracy/mse has not improved
        improved = False
        if task in {'class', 'multi'} and acc > best_acc:
            best_acc = acc
            improved = True
        if task in {'reg', 'multi'} and mse < best_mse:
            best_mse = mse
            improved = True

        if improved:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break

    df_metrics_trial = pd.concat(metrics_per_epoch, ignore_index=True)

    # Add metadata
    df_metrics_trial['ModelName'] = model_class.__name__
    df_metrics_trial['Task'] = task

    return model, df_metrics_trial


def train_model(
    dataset: Dataset,
    model_class: Callable[..., nn.Module],
    task: str,  # options: 'class', 'reg', 'multi',
    num_drivers: int,
    num_steps: int,
    num_params: int,
    f: float = 0.8,
    batch_size: int = 64,
    n_trials: int = 10,
    n_epoch: int = 30,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    verbose: bool = False,
    enable_tqdm: bool = True,
    weight_class: float = 1.0,
    weight_reg: float = 1.0,
    loss_fn_class: Optional[Callable] = None,
    loss_fn_reg: Optional[Callable] = None,
    model_kwargs: Optional[dict] = None
) -> Tuple[nn.Module, dict, pd.DataFrame]:
    """Train and validate pytorch model to predict the driver and/or regress
    lap/segment time.

    Dataset: `TelemetryDataset`

    Args:
        dataset: A `TelemetryDataset` instance containing telemetry data.
        model_class (Callable[..., nn.Module]): Model constructor
            (e.g., `TelemetryModelCNN`).
        task (str): Task type. Options: `'class'`, `'reg'`, `'multi'`.
        num_drivers (int): Number of drivers (for classification head).
        num_steps (int): Telemetry time steps per input.
        num_params (int): Number of parameters per step.
        f (float): Fraction of data to use for training. Default is 0.8.
        batch_size (int): Number of samples per batch. Default is 64.
        n_trials (int): Number of independent training runs. Default is 10.
        n_epoch: Number of epochs per trial. Default is 30.
        lr (float, optional): learning rate used during model training.
            Defaults to 1e-3.
        early_stopping_patience (int, optional): how many epochs to wait
            before early stopping if validation accuracy does not improve.
        verbose (bool, optional): if training progress (accuracy) should
            be printed during training. Defaults to `False`.
        enable_tqdm (bool): If True, use tqdm to track progress of training
            independent trials. Defaults to `True`.
        weight_class: Classification loss weight (for `'multi'` task).
        weight_reg: Regression loss weight (for `'multi'` task).
        loss_fn_class (Callable, optional): Loss function for classification.
            Required for `'class'`/`'multi'`. Defaults to`nn.CrossEntropyLoss()`.
        loss_fn_reg (Callable, optional): Loss function for regression.
            Required for `'reg'`/`'multi'`. Defaults to `nn.MSELoss`.
        model_kwargs: Optional kwargs to pass to `model_class`.

    Raises:
        ValueError: For invalid task or missing loss functions.

    Returns:
        Tuple[nn.Module, dict, pd.DataFrame]:
        - best_model: The best-performing trained model (moved to CPU).
        - model_kwargs: Model constructor arguments (for records).
        - df_metrics: DataFrame with metrics across trials and epochs.
    """
    def score_model(acc, mse, weight_class=0.5, weight_reg=0.5):
        """Score models for multi task."""
        accs = np.array(final_acc)
        mses = np.array(final_mse)
        acc_norm = (accs - accs.min()) / (accs.max() - accs.min() + 1e-8)
        mse_norm = 1 - (mses - mses.min()) / (mses.max() - mses.min() + 1e-8)
        return weight_class * acc_norm + weight_reg * mse_norm

    if task not in {'class', 'reg', 'multi'}:
        raise ValueError(f'Invalid task: {task}')

    if task in {'class', 'multi'} and loss_fn_class is None:
        loss_fn_class = nn.CrossEntropyLoss()
    if task in {'reg', 'multi'} and loss_fn_reg is None:
        loss_fn_reg = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Split dataset into training and validation sets
    train_size = int(f * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_ds, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size)

    # Handle model arguments (overwrite if neede)
    model_kwargs = model_kwargs or {}
    model_kwargs['task'] = task
    model_kwargs['num_classes'] = num_drivers
    model_kwargs['num_steps'] = num_steps
    model_kwargs['num_params'] = num_params

    # Train the model
    models, metrics = [], []
    final_acc, final_mse = [], []

    if enable_tqdm:
        trial_loop = tqdm(range(n_trials), desc='Running Trials', leave=True)
    else:
        trial_loop = range(n_trials)

    for trial in trial_loop:

        model, df_metrics_trial = train_one_trial(
            model_class=model_class,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            device=device,
            task=task,
            n_epoch=n_epoch,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
            weight_class=weight_class,
            weight_reg=weight_reg,
            loss_fn_class=loss_fn_class,
            loss_fn_reg=loss_fn_reg,
            model_kwargs=model_kwargs
        )

        df_metrics_trial['Trial'] = trial
        metrics.append(df_metrics_trial)
        models.append(model)

        final_acc.append(df_metrics_trial['Accuracy'].iloc[-1])
        final_mse.append(df_metrics_trial['MSE'].iloc[-1])

        if verbose:
            print(f'Trial {trial+1}/{n_trials} complete')
            if task in {'class', 'multi'}:
                print(f'  Final Val Acc: {final_acc[-1]:.2%}')
            if task in {'reg', 'multi'}:
                print(f'  Final Val MSE: {final_mse[-1]:.4f}')

    df_metrics = pd.concat(metrics).sort_values(['Trial', 'Epoch'])

    # Identify the best trial
    if task == 'class':
        best_trial = np.argmax(final_acc)
    elif task == 'reg':
        best_trial = np.argmin(final_mse)
    elif task == 'multi':
        scores = score_model(final_acc, final_mse, weight_class, weight_reg)
        best_trial = int(np.argmax(scores))
    df_metrics['BestTrial'] = (df_metrics['Trial'] == best_trial).astype(int)

    best_model = copy.deepcopy(models[best_trial]).cpu()

    return best_model, model_kwargs, df_metrics