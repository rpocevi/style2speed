# style2speed/config.py

EPSILON = 1e-10  # small constant to avoid division by zero

# Telemetry columns that uniquely identify data for individual laps
GROUP_COLS = ['Location', 'Year', 'Event', 'DriverNumber', 'LapNumber']
# Telemetry columns that uniquely identify attribution groups
GROUP_COLS_ATTR = ['Location', 'Year', 'Event', 'DriverNumber', 'Param',
                   'Distance', 'X', 'Y', 'ModelName', 'Task', 'Head']

# Raw inputs
IMPORT_PARAMS = ['Distance', 'X', 'Y', 'Z', 'Speed', 'RPM', 'Throttle',
                 'nGear', 'Brake', 'DRS', 'DistanceToDriverAhead']

# Data type groupings for interpolation
PARAMS_CATEGORICAL = ['Gear', 'Brake', 'DRS']
PARAMS_CONTINUOUS = ['Speed', 'RPM', 'Throttle', 'Timer', 'Distance', 'FollowingDistance']
PARAMS_TRAJECTORY = ['X', 'Y', 'Z']

# Data type groupings for parameter derivation
PARAMS_TO_DERIVE_CONT = ['Speed', 'RPM', 'Throttle']
PARAMS_TO_DERIVE_CATEG = ['Gear', 'Brake']
PARAMS_TO_INVERSE = ['FollowingDistance']

# Data type groupings for parameter normalization
PARAMS_CON_POS = ['Speed', 'RPM', 'Throttle', 'Curvature', 'Radius', 'LatAcc']
PARAMS_CON_SIGNED = ['dSpeed', 'dRPM', 'dThrottle']
PARAMS_ORD_POS = ['Gear']
PARAMS_ORD_SIGNED = ['GearChange']
PARAMS_INV = ['FollowingDistanceInv']

# Methods for calculating baseline telemetry param values
PARAM_BASELINE_STRATEGY = {
    'Brake': 'mode',
    'Gear': 'mode',
    'DRS': 'mode',
    'BrakeChange': 'mode',
    'GearChange': 'mode',
    'Speed': 'median',
    'RPM': 'median',
    'Throttle': 'median',
    'dSpeed': 'median',
    'dRPM': 'median',
    'dThrottle': 'median',
    'LatAcc': 'median',
    'Curvature': 'median',
    'Radius': 'median',
    'FollowingDistanceInv': 'median',
}

# Use cases
NN_PARAMS = ['Speed', 'dSpeed', 'RPM', 'dRPM','Throttle', 'dThrottle',
             'Gear', 'GearChange', 'Brake', 'BrakeChange', 'DRS',
             'Curvature', 'Radius', 'LatAcc', 'FollowingDistanceInv']
STATS_PARAMS = ['X', 'Y', 'Z', 'Speed', 'dSpeed', 'RPM', 'dRPM', 'Throttle',
                'dThrottle', 'Gear', 'GearChange', 'Brake', 'BrakeChange',
                'DRS', 'Curvature', 'Radius', 'LatAcc', 'FollowingDistanceInv']
STATS = ['mean', 'std', 'min', 'median', 'max']

# Colot palettes for plotting
TAB10_CUSTOM = [
    "#ff7f0e",  # orange
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf"   # cyan
    "#d62728",  # red
]