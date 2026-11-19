import io
import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers


SEARCH_SPACE = {
    "c1": [4, 6, 8, 12, 16, 24, 32],
    "c2": [8, 16, 24, 32, 48, 64],
    "kernel": [3, 5],
    "fc1": [64, 120, 168, 256],
    "fc2": [32, 64, 84, 128],
    "activation": ["tanh", "relu"],
    "pooling": ["avg", "max"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [0.0001, 0.0003, 0.001, 0.003, 0.01],
}

BASELINE = {
    "c1": 6,
    "c2": 16,
    "kernel": 5,
    "fc1": 120,
    "fc2": 84,
    "activation": "tanh",
    "pooling": "avg",
    "dropout": 0.0,
    "lr": 0.001,
}


def build_model(genome):
    pool_layer = layers.AveragePooling2D if genome["pooling"] == "avg" else layers.MaxPooling2D
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(genome["c1"], genome["kernel"], padding="same", activation=genome["activation"])(inputs)
    x = pool_layer(2)(x)
    x = layers.Conv2D(genome["c2"], genome["kernel"], padding="valid", activation=genome["activation"])(x)
    x = pool_layer(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(genome["fc1"], activation=genome["activation"])(x)
    x = layers.Dropout(genome["dropout"])(x)
    x = layers.Dense(genome["fc2"], activation=genome["activation"])(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=genome["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def set_seed(seed):
    keras.utils.set_random_seed(seed)


def estimate_macs(genome):
    k = int(genome["kernel"])
    c1 = int(genome["c1"])
    c2 = int(genome["c2"])
    first_conv = 28 * 28 * c1 * k * k
    side_after_first_pool = 14
    side_after_second_conv = side_after_first_pool - k + 1
    second_conv = side_after_second_conv * side_after_second_conv * c2 * c1 * k * k
    side_after_second_pool = side_after_second_conv // 2
    flattened = side_after_second_pool * side_after_second_pool * c2
    dense_one = flattened * int(genome["fc1"])
    dense_two = int(genome["fc1"]) * int(genome["fc2"])
    dense_output = int(genome["fc2"]) * 10
    return int(first_conv + second_conv + dense_one + dense_two + dense_output)


def preprocess_image_bytes(contents):
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = ImageOps.autocontrast(image)
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.mean() > 127:
        pixels = 255 - pixels
    foreground = pixels > 24
    if foreground.any():
        ys, xs = np.where(foreground)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        cropped = Image.fromarray(pixels[top:bottom, left:right])
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        x = (28 - cropped.width) // 2
        y = (28 - cropped.height) // 2
        canvas.paste(cropped, (x, y))
    else:
        canvas = Image.new("L", (28, 28), 0)
    normalized = np.asarray(canvas, dtype="float32") / 255.0
    return normalized[..., None], canvas


def image_to_png_bytes(image, scale=10):
    enlarged = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    output = io.BytesIO()
    enlarged.save(output, format="PNG")
    return output.getvalue()


def hardware_details():
    devices = tf.config.list_physical_devices("GPU")
    return {
        "compute_device": "GPU" if devices else "CPU",
        "gpu_devices": [device.name for device in devices],
        "tensorflow_version": tf.__version__,
    }


def model_explanation(genome):
    return [
        {"name": "Input", "shape": "28 × 28 × 1", "purpose": "A single grayscale digit is scaled into the same format used during training."},
        {"name": "Convolution C1", "shape": f"{genome['c1']} feature maps", "purpose": "Small shared filters learn simple local patterns such as edges, curves and short strokes."},
        {"name": "Pooling S2", "shape": "Downsampled feature maps", "purpose": f"{genome['pooling'].title()} pooling reduces spatial size while keeping useful responses."},
        {"name": "Convolution C3", "shape": f"{genome['c2']} feature maps", "purpose": "The next filters combine earlier patterns into more useful stroke fragments."},
        {"name": "Pooling S4", "shape": "Compact feature maps", "purpose": "A second reduction keeps the representation compact."},
        {"name": "Dense layers", "shape": f"{genome['fc1']} → {genome['fc2']} units", "purpose": "Learned features are combined to form evidence for each digit class."},
        {"name": "Softmax output", "shape": "10 class scores", "purpose": "Scores are converted into probabilities for digits 0 through 9."},
    ]
