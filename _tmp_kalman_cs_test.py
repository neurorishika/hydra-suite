import io
import logging

import numpy as np

from hydra_suite.core.filters.kalman import KalmanFilterManager

params = {
    "KALMAN_NOISE_COVARIANCE": 0.03,
    "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
    "KALMAN_DAMPING": 0.95,
    "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
    "REFERENCE_BODY_SIZE": 20.0,
    "RESIZE_FACTOR": 1.0,
    "KALMAN_MATURITY_AGE": 5,
    "KALMAN_INITIAL_VELOCITY_RETENTION": 0.2,
}
kfm = KalmanFilterManager(num_targets=3, params=params)
kfm.initialize_filter(0, np.array([100.0, 100.0, 1.0, 0.0, 0.0], dtype=np.float32))

# Inject bad off-diagonal that violates Cauchy-Schwarz
bad_P = np.diag([5.0, 5.0, 5.0, 10.0, 10.0]).astype(np.float32)
bad_P[0, 1] = bad_P[1, 0] = 999.0
kfm.P[0] = bad_P

log_buf = io.StringIO()
logging.getLogger("hydra_suite.core.filters.kalman").addHandler(
    logging.StreamHandler(log_buf)
)

kfm.predict()
limit = np.sqrt(kfm.P[0, 0, 0] * kfm.P[0, 1, 1])
print(
    f"P[0,0,1]={kfm.P[0, 0, 1]:.4f}  limit={limit:.4f}  ok={abs(kfm.P[0, 0, 1]) <= limit + 1e-5}"
)

kfm.correct(0, np.array([101.0, 101.0, 1.05], dtype=np.float32))
logs = log_buf.getvalue()
if "reset due to numerical instability" in logs:
    print("FAIL:", logs)
else:
    print("PASS: no track reset after injecting bad P")
