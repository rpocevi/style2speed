# style2speed/models/architetures.py

from typing import Optional, Union

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TelemetryDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, list[np.ndarray]],
        task: str
    ):
        """Custom dataset for F1 telemetry data.

        Args:
            X (np.ndarray): Input telemetry features of shape
                [num_sets, num_steps, num_params].
            y (Union[np.ndarray, List[np.ndarray]]): Targets.
                - For `'class`' or `'reg'`: single array of shape [num_sets].
                - For `'multi'`: list of two arrays [class_labels, reg_targets],
                  each of shape [num_sets].
            task (str): Task type. One of: `'class'`, `'reg'`, `'multi'`.

        Raises:
            ValueError: If task is invalid or if y format doesn't match task.
        """
        self.task = task.lower()
        self.X = torch.tensor(X, dtype=torch.float32)

        if self.task not in {'class', 'reg', 'multi'}:
            raise ValueError(f"Invalid task: {task}. "
                             f"Choose from 'class', 'reg', 'multi'.")

        if self.task == 'multi':
            if not (isinstance(y, list) and len(y) == 2):
                raise ValueError("For 'multi' task, y must be a list of two arrays.")
            self.y = [
                torch.tensor(y[0], dtype=torch.long),
                torch.tensor(y[1], dtype=torch.float32)
            ]
        else:
            # Only allow np.ndarray for class and reg tasks
            if isinstance(y, list):
                raise TypeError(f"For task '{self.task}', y must be a single np.ndarray, not a list.")
            if not isinstance(y, np.ndarray):
                raise TypeError(f"Expected y to be np.ndarray, got {type(y)}")
            dtype = torch.long if self.task == 'class' else torch.float32
            self.y = torch.tensor(y, dtype=dtype)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.task == 'multi':
            return self.X[idx], [y_i[idx] for y_i in self.y]
        else:
            return self.X[idx], self.y[idx]
        

class TelemetryModelFC(nn.Module):
    def __init__(
        self,
        task: str,  # Options: 'class', 'reg', 'multi'
        num_steps: int,
        num_params: int = 15,
        num_classes: Optional[int] = None,
        fc_sizes: list[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Fully connected neural network for classifying drivers or regressing
        lap/segment times from flattened telemetry data.

        Input shape: [`batch_size`, `num_steps`, `num_params`].
        Internally flattened to: [`batch_size`, `num_steps * num_params`].

        Args:
            task (str): One of `'class'`, `'reg'`, or `'multi'`.
            num_steps (int): Number of time steps in each sample.
            num_params (int): Number of telemetry parameters.
            num_classes (int, optional): Number of classes
                (required for `'class'` or `'multi'`).
            fc_sizes (List[int]): Sizes of hidden fully connected layers.
            dropout_rate (float): Dropout rate between layers.

        Raises:
            ValueError: If task is invalid or num_classes is missing when required.
        """
        super().__init__()

        valid_tasks = {'class', 'reg', 'multi'}
        if task not in valid_tasks:
            raise ValueError(f'Invalid task: {task}. Choose from {valid_tasks}.')
        if task in {'class', 'multi'} and num_classes is None:
            raise ValueError('num_classes must be provided for classification tasks.')

        self.task = task

        # --- Shared FC layers ---
        fc_layers = []
        input_size = num_params * num_steps
        for output_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = output_size
        self.fc_shared = nn.Sequential(*fc_layers)

        # --- Task-specific heads ---
        if task in ['class', 'multi']:
            self.head_class = nn.Linear(input_size, num_classes)
        if task in ['reg', 'multi']:
            self.head_reg = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # → [batch_size, num_steps * num_params]
        x = self.fc_shared(x)

        if self.task == 'class':
            return self.head_class(x)
        elif self.task == 'reg':
            return self.head_reg(x).squeeze(1)  # → [batch_size]
        elif self.task == 'multi':
            class_out = self.head_class(x)          # [batch_size, num_classes]
            reg_out = self.head_reg(x).squeeze(1)   # [batch_size]
            return class_out, reg_out
        

class TelemetryModelCNN(nn.Module):
    def __init__(
        self,
        task: str,  # Options: 'class', 'reg', 'multi'
        num_steps: int,
        num_params: int = 15,
        num_classes: Optional[int] = None,
        cnn_sizes: list[int] = [32, 64],
        kernel_size: int = 5,
        use_batch_norm: bool = True,
        fc_sizes: list[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        1D convolutional neural network for classifying drivers or regressing
        lap/segment times from telemetry data.

        Expected input shape: [`batch_size`, `num_steps`, `num_params`].
        Internally, input is permuted to [`batch_size`, `num_params`, `num_steps`].

        Args:
            task (str): One of `'class'`, `'reg'`, or `'multi'`.
            num_steps (int): Number of time steps in each sample.
            num_params (int): Number of telemetry parameters.
            num_classes (int, optional): Number of classes
                (required for `'class'` or `'multi'`).
            cnn_sizes (list[int]): Sizes of hidden cnn layers.
            kernel_size (int): Kernel size for CNN layers.
            use_batch_norm (bool): If True, use batch normalization after
                each CNN layer. Defaults to True.
            fc_sizes (list[int]): Sizes of hidden fully connected layers.
            dropout_rate (float): Dropout rate between layers.

        Raises:
            ValueError: If task is invalid or num_classes is missing when required.
        """
        super().__init__()

        valid_tasks = {'class', 'reg', 'multi'}
        if task not in valid_tasks:
            raise ValueError(f'Invalid task: {task}. Choose from {valid_tasks}.')
        if task in {'class', 'multi'} and num_classes is None:
            raise ValueError('num_classes must be provided for classification tasks.')

        self.task = task

        # --- Shared CNN layers ---
        conv_layers = []
        input_size = num_params
        for output_size in cnn_sizes:
            conv_layers.append(nn.Conv1d(input_size, output_size, kernel_size,
                                         padding=kernel_size // 2))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(output_size))
            conv_layers.extend([
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout_rate)
            ])
            input_size = output_size

        self.conv = nn.Sequential(*conv_layers)

        # Adjust FC input after convolutions & pooling
        pool_factor = 2 ** len(cnn_sizes)
        reduced_num_steps = num_steps // pool_factor
        fc_input_size = reduced_num_steps * cnn_sizes[-1]

        # --- Shared FC layers ---
        fc_layers = []
        for fc_output_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            fc_input_size = fc_output_size
        self.fc_shared = nn.Sequential(*fc_layers)

        # --- Task-specific heads ---
        if task in ['class', 'multi']:
            self.head_class = nn.Linear(fc_input_size, num_classes)
        if task in ['reg', 'multi']:
            self.head_reg = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # → [batch_size, num_params, num_steps]
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_shared(x)

        if self.task == 'class':
            return self.head_class(x)
        elif self.task == 'reg':
            return self.head_reg(x).squeeze(1)  # → [batch_size]
        elif self.task == 'multi':
            class_out = self.head_class(x)          # [batch_size, num_classes]
            reg_out = self.head_reg(x).squeeze(1)   # [batch_size]
            return class_out, reg_out
        

class TelemetryModelLSTM(nn.Module):
    def __init__(
        self,
        task: str,  # options: 'class', 'reg', 'multi'
        num_steps: int,
        num_params: int = 15,
        num_classes: Optional[int] = None,
        lstm_size: int = 64,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        lstm_use_hidden: bool = False,
        lstm_use_all_hidden_layers: bool = False,
        lstm_step_pooling: str = 'mean',  # Options: 'last', 'mean', 'max'
        fc_sizes: list[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        LSTM neural network for classifying drivers or regressing lap/segment
        times from telemetry data.

        Expected input shape: [`batch_size`, `num_steps`, `num_params`].

        Args:
            task (str): One of `'class'`, `'reg'`, or `'multi'`.
            num_steps (int): Number of time steps in each sample.
                (unused here, included only for consistency).
            num_params (int): Number of telemetry parameters.
            num_classes (int, optional): Number of classes
                (required for 'class' or 'multi').
            lstm_size (int): Hidden size of LSTM layers.
            birectional (bool, optional): If True, use bidirectional LSTM.
                Defaults to True.
            lstm_num_layers (int): Number of LSTM layers.
            lstm_use_hidden (bool): True if the hidden state at the last step
                should be passed from LSTM to FC. False if the outer most output
                at each time step should be passed to FC. Defaults to False.
            lstm_use_all_hidden_layers (bool): If True, use all LSTM layers as
                input to the fully connected linear layer. If False, use only
                the last layer. Defaults to True. Only applicable if
                lstm_use_hidden is True.
            lstm_step_pooling (str or None, optional): If 'mean' or 'max', pool
                information from all LSTM steps using the corresponding agg
                function. If 'last', use the last step only.
                Defaults to 'mean'.
            fc_sizes (list[int]): Sizes of hidden fully connected layers.
            dropout_rate (float): Dropout rate between layers.

        Raises:
            ValueError: If task is invalid or num_classes is missing when required.
            ValueError: If invalid or missing lstm_step_pooling value.
        """
        super().__init__()

        valid_tasks = {'class', 'reg', 'multi'}
        if task not in valid_tasks:
            raise ValueError(f"Invalid task: {task}. Choose from {valid_tasks}.")
        if task in {'class', 'multi'} and num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        if not lstm_use_hidden:
            if lstm_step_pooling not in ['last', 'mean', 'max']:
                raise ValueError('Invalid or missing lstm_step_pooling value.')

        self.lstm_size = lstm_size
        self.num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_hidden = lstm_use_hidden
        self.use_all_hidden_layers = lstm_use_all_hidden_layers
        self.pooling_mode = lstm_step_pooling
        self.task = task

        # --- Shared LSTM Layer ---
        self.lstm = nn.LSTM(
            input_size=num_params,
            hidden_size=lstm_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # --- Determine FC input size ---
        if lstm_use_hidden:  # using h_n
            if lstm_use_all_hidden_layers:
                fc_input_size = lstm_size * lstm_num_layers * self.num_directions
            else:
                fc_input_size = lstm_size * self.num_directions
        else:  # using output
            fc_input_size = lstm_size * self.num_directions # same for all modes

        # --- Shared FC layers ---
        fc_layers = []
        for fc_output_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            fc_input_size = fc_output_size
        self.fc_shared = nn.Sequential(*fc_layers)

        # --- Task-specific heads ---
        if task in ['class', 'multi']:
            self.head_class = nn.Linear(fc_input_size, num_classes)
        if task in ['reg', 'multi']:
            self.head_reg = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # output shape: [batch_size, num_steps, hidden_size * num_directions]
        # h_n shape: [num_layers * num_directions, batch_size, hidden_size]

        if self.use_hidden:
            if self.use_all_hidden_layers:
                # Combine all hidden layers
                features = h_n.transpose(0, 1).reshape(x.size(0), -1)
            else:
                # Default: use last layer's final hidden state
                h_n_last = h_n[-self.num_directions:]
                features = h_n_last.transpose(0, 1).reshape(x.size(0), -1)
        else:
            if self.pooling_mode == 'mean':
                features = torch.mean(output, dim=1)
            elif self.pooling_mode == 'max':
                features, _ = torch.max(output, dim=1)
            elif self.pooling_mode == 'last':
                features = output[:, -1, :]
            else:
                raise ValueError(f'Invalid pooling mode: {self.pooling_mode}')

        features = self.fc_shared(features)

        if self.task == 'class':
            return self.head_class(features)
        elif self.task == 'reg':
            return self.head_reg(features).squeeze(1)
        elif self.task == 'multi':
            class_out = self.head_class(features)          # [batch_size, num_classes]
            reg_out = self.head_reg(features).squeeze(1)   # [batch_size]
            return class_out, reg_out
        

class TelemetryModelCNNLSTM(nn.Module):
    def __init__(
        self,
        task: str,  # options: 'class', 'reg', 'multi'
        num_steps: int,
        num_params: int = 15,
        num_classes: Optional[int] = None,
        cnn_sizes: list[int] = [32, 64],
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        lstm_size: int = 64,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        lstm_use_hidden: bool = False,
        lstm_use_all_hidden_layers: bool = False,
        lstm_step_pooling: str = 'mean',  # Options: 'last', 'mean', 'max'
        fc_sizes: list[int] = [256, 128],
        dropout_rate: float = 0.1
    ):
        """
        CNN + LSTM nn for classifying drivers from flattened telemetry data.

        Expected input shape: [`batch_size`, `num_steps`, `num_params`].
        Internally, input is permuted to [`batch_size`, `num_params`, `num_steps`]
        for CNN, and then back to [`batch_size`, `num_steps`, `num_params`] for LSTM.

        Args:
            task (str): One of `'class'`, `'reg'`, or `'multi'`.
            num_steps (int): Number of time steps in each sample.
                (unused here, included only for consistency).
            num_params (int): Number of telemetry parameters.
            num_classes (int, optional): Number of classes
                (required for `'class'` or `'multi'`).
            cnn_sizes (list[int]): Sizes of hidden cnn layers.
            kernel_size (int): Kernel size for CNN layers.
            use_batch_norm (bool): If True, use batch normalization after
                each CNN layer. Defaults to `True`.
            lstm_size (int): Hidden size of LSTM layers.
            lstm_num_layers (int): Number of LSTM layers.
            birectional (bool, optional): If `True`, use bidirectional LSTM.
                Defaults to `False`.
            lstm_use_hidden (bool): `True` if the hidden state at the last step
                should be passed from LSTM to FC. `False` if the outer most output
                at each time step should be passed to FC. Defaults to `False`.
            lstm_use_all_hidden_layers (bool): If `True`, use all LSTM layers as
                input to the fully connected linear layer. If `False`, use only
                the last layer. Defaults to `True`. Only applicable if
                `lstm_use_hidden` is `True`.
            lstm_step_pooling (str): If `'mean'` or `'max'`, pool information
                from all LSTM steps using the corresponding agg function.
                If `'last'`, use the last step only. Defaults to  `'mean'`.
                Only used if `lstm_use_hidden` is `False`.
            fc_sizes (list[int]): Sizes of hidden fully connected layers.
            dropout_rate (float): Dropout rate between layers.

        Raises:
            ValueError: If task is invalid or num_classes is missing when required.
            ValueError: If invalid or missing lstm_step_pooling value.
        """
        super().__init__()

        valid_tasks = {'class', 'reg', 'multi'}
        if task not in valid_tasks:
            raise ValueError(f"Invalid task: {task}. Choose from {valid_tasks}.")
        if task in {'class', 'multi'} and num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        if not lstm_use_hidden:
            if lstm_step_pooling not in ['last', 'mean', 'max']:
                raise ValueError('Invalid or missing lstm_step_pooling value.')

        self.lstm_size = lstm_size
        self.num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_hidden = lstm_use_hidden
        self.use_all_hidden_layers = lstm_use_all_hidden_layers
        self.pooling_mode = lstm_step_pooling
        self.task = task

        # --- Shared 1D CNN layer ---
        conv_layers = []
        input_size = num_params
        for output_size in cnn_sizes:
            conv_layers.append(nn.Conv1d(input_size, output_size, kernel_size,
                                         padding=kernel_size // 2))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(output_size))
            conv_layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = output_size
        self.cnn = nn.Sequential(*conv_layers)

        # --- Shared LSTM layer ---
        self.lstm = nn.LSTM(
            input_size=cnn_sizes[-1],
            hidden_size=lstm_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # --- Determine FC input ---
        if lstm_use_hidden:  # using h_n
            if lstm_use_all_hidden_layers:
                fc_input_size = lstm_size * lstm_num_layers * self.num_directions
            else:
                fc_input_size = lstm_size * self.num_directions
        else:  # using output
            fc_input_size = lstm_size * self.num_directions # same for all modes

        # --- Shared FC layers ---
        fc_layers = []
        for fc_output_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            fc_input_size = fc_output_size
        self.fc_shared = nn.Sequential(*fc_layers)

        # --- Task-specific heads ---
        if task in ['class', 'multi']:
            self.head_class = nn.Linear(fc_input_size, num_classes)
        if task in ['reg', 'multi']:
            self.head_reg = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # → [batches, num_params, num_steps]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # → [batches, num_steps, num_params]
        output, (h_n, _) = self.lstm(x)

        if self.use_hidden:
            if self.use_all_hidden_layers:
                # Combine all hidden layers
                features = h_n.transpose(0, 1).reshape(x.size(0), -1)
            else:
                # Default: use last layer's final hidden state
                h_n_last = h_n[-self.num_directions:]
                features = h_n_last.transpose(0, 1).reshape(x.size(0), -1)
        else:
            if self.pooling_mode == 'mean':
                features = torch.mean(output, dim=1)
            elif self.pooling_mode == 'max':
                features, _ = torch.max(output, dim=1)
            elif self.pooling_mode == 'last':
                features = output[:, -1, :]
            else:
                raise ValueError(f'Invalid pooling mode: {self.pooling_mode}')

        features = self.fc_shared(features)

        if self.task == 'class':
            return self.head_class(features)
        elif self.task == 'reg':
            return self.head_reg(features).squeeze(1)
        elif self.task == 'multi':
            class_out = self.head_class(features)          # [batch_size, num_classes]
            reg_out = self.head_reg(features).squeeze(1)   # [batch_size]
            return class_out, reg_out