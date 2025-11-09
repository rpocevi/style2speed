# style2speed/__init__.py

# Preprocessing
from .data import (
    compile_telemetry,
    interpolate_telemetry,
    derive_params,
)
from .preprocessing import (
    calc_segment_times,
    normalize_telemetry,
    normalize_times,
    inverse_normalize,
    get_baseline,
    build_torch_dataset,
)

# Stats
from .stats import (
    get_circuit_dominance,
    calc_param_stats,
    calc_significance,
)

# Model Architectures
from .models.architectures import (
    TelemetryDataset,
    TelemetryModelFC,
    TelemetryModelCNN,
    TelemetryModelLSTM,
    TelemetryModelCNNLSTM,
)

# Model Training
from .models.training import (
    train_one_epoch,
    train_one_trial,
    train_model,
)

# Model Inference
from .models.inference import (
    run_inference,
    run_nn_pipeline,
    combine_outputs,
    compare_models,
)

# Visualization - Circuit & Traces
from .plotting.circuit import (
    visualize_circuit,
    visualize_along_circuit_ind_values,
    visualize_along_circuit_segment_vals,
)
from .plotting.traces import (
    plot_traces,
    overlay_telemetry_and_importance_traces,
    overlay_tel_class_reg_traces,
)

# Visualization - Performance
from .plotting.performance import (
    plot_training_curves,
    plot_acc_across_drivers,
    plot_confusion_matrix,
    plot_violins,
    plot_param_importance_vs_drivers,
)

# Utils
from .utils import download_file_if_colab