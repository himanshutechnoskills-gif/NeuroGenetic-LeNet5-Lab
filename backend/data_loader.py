import numpy as np
from tensorflow import keras


def load_mnist(seed=42, ga_train=6000, ga_val=1500):
    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    images = images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    images = images[..., None]
    test_images = test_images[..., None]
    order = np.random.default_rng(seed).permutation(len(images))
    val_count = int(len(images) * 0.1)
    val_ids = order[:val_count]
    train_ids = order[val_count:]
    x_train = images[train_ids]
    y_train = labels[train_ids]
    x_val = images[val_ids]
    y_val = labels[val_ids]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": test_images,
        "y_test": test_labels,
        "x_ga": x_train[:min(ga_train, len(x_train))],
        "y_ga": y_train[:min(ga_train, len(y_train))],
        "x_ga_val": x_val[:min(ga_val, len(x_val))],
        "y_ga_val": y_val[:min(ga_val, len(y_val))],
    }


def save_test_examples(data, path):
    chosen_images = []
    chosen_labels = []
    for digit in range(10):
        index = int(np.flatnonzero(data["y_test"] == digit)[0])
        chosen_images.append(data["x_test"][index])
        chosen_labels.append(digit)
    np.savez_compressed(path, images=np.asarray(chosen_images), labels=np.asarray(chosen_labels))
