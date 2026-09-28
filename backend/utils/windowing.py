import numpy as np


def create_sequences(data, window_size, horizon, target_col_index=1):
    """
    Build windowed sequences for direct multi-horizon forecasting.

    Each window of `window_size` steps predicts the target column value
    `horizon` steps after the window ends.

    Raises:
        ValueError: if `data` isn't 2D, `window_size`/`horizon` aren't
            positive, `target_col_index` is out of range, or there isn't
            enough data to build at least one sequence. Raising here
            (instead of silently returning empty arrays) surfaces bad
            inputs immediately rather than as a confusing failure later
            in model.fit().
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D (timesteps, features); got shape {data.shape}")
    if window_size <= 0 or horizon <= 0:
        raise ValueError("`window_size` and `horizon` must be positive integers.")
    if not (0 <= target_col_index < data.shape[1]):
        raise ValueError(
            f"`target_col_index`={target_col_index} is out of range for "
            f"{data.shape[1]} columns."
        )

    n_sequences = len(data) - window_size - horizon + 1
    if n_sequences <= 0:
        raise ValueError(
            f"Not enough rows ({len(data)}) to build a sequence with "
            f"window_size={window_size} and horizon={horizon}. "
            f"Need at least {window_size + horizon} rows."
        )

    n_features = data.shape[1]
    X = np.empty((n_sequences, window_size, n_features), dtype=data.dtype)
    y = np.empty(n_sequences, dtype=data.dtype)

    for i in range(n_sequences):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size + horizon - 1, target_col_index]

    return X, y