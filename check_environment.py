import sys
import numpy as np
import tensorflow as tf
from backend.ml_model import BASELINE, build_model, hardware_details


def main():
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
        print("Recommended interpreter: 64-bit Python 3.11.")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Compute device: {hardware_details()['compute_device']}")
    model = build_model(BASELINE)
    parameter_count = int(model.count_params())
    if parameter_count != 61706:
        raise RuntimeError(f"Baseline parameter check failed: expected 61706, got {parameter_count}.")
    prediction = model.predict(np.zeros((1, 28, 28, 1), dtype="float32"), verbose=0)
    if prediction.shape != (1, 10):
        raise RuntimeError(f"Output shape check failed: {prediction.shape}.")
    if not np.isclose(float(prediction.sum()), 1.0, atol=1e-4):
        raise RuntimeError("Softmax output check failed: class scores do not sum to one.")
    print("Baseline LeNet parameter check: PASS (61,706)")
    print("28 × 28 input and 10-class softmax check: PASS")
    print("Environment check complete. You can run: python train.py")


if __name__ == "__main__":
    main()
