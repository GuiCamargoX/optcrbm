"""Dataset loaders used by the tutorial and refactored run scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

from . import utils


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"


def load_dataset(name, window_shape, block_shape, as_tensors=True):
    """Load a named dataset with the preprocessing expected by the CRBM."""
    loaders = {
        "semeion": load_semeion,
        "caltech": load_caltech,
        "mpeg": load_mpeg,
        "mnist": load_mnist,
    }
    key = name.lower()
    if key not in loaders:
        raise ValueError("Unknown dataset: {}".format(name))
    return loaders[key](window_shape, block_shape, as_tensors=as_tensors)


def load_semeion(window_shape, block_shape, as_tensors=True, test_size=0.20, random_state=42):
    images, labels = read_semeion()
    x_train, x_test, _, _ = train_test_split(
        images, labels, stratify=labels, test_size=test_size, random_state=random_state
    )
    return _maybe_to_tensors(x_train, x_test, window_shape, block_shape, as_tensors)


def load_caltech(window_shape, block_shape, as_tensors=True):
    dataset_path = DATASET_DIR / "Caltech" / "caltech101_silhouettes_28_split1.mat"
    caltech = sio.loadmat(dataset_path)
    x_train = caltech["train_data"].reshape(4100, 28, 28)
    x_test = caltech["test_data"].reshape(2307, 28, 28)
    return _maybe_to_tensors(x_train, x_test, window_shape, block_shape, as_tensors)


def load_mpeg(window_shape, block_shape, as_tensors=True, test_size=0.20, random_state=42):
    dataset_path = DATASET_DIR / "MPEG" / "MPEG.csv"
    dataframe = pd.read_csv(dataset_path)
    images = dataframe.values[:, 1:].reshape(1402, 28, 28)
    images = np.around(images.astype(float) / 255)

    x_temp, x_test = train_test_split(images, test_size=test_size, random_state=random_state)
    x_train, _ = train_test_split(x_temp, test_size=test_size, random_state=random_state)
    return _maybe_to_tensors(x_train, x_test, window_shape, block_shape, as_tensors)


def load_mnist(window_shape, block_shape, as_tensors=True):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("MNIST loading requires the optional tensorflow dependency") from exc

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255
    return _maybe_to_tensors(x_train, x_test, window_shape, block_shape, as_tensors)


def read_semeion(path=None):
    dataset_path = Path(path) if path is not None else DATASET_DIR / "Semeion" / "semeion.data"
    width = 16
    height = 16
    image_size = width * height
    classes = 10

    images = []
    labels = []
    with dataset_path.open("r") as handle:
        for line in handle:
            values = line.split(" ")
            image = [int(float(values[index])) for index in range(image_size)]
            one_hot = [int(float(values[index])) for index in range(image_size, image_size + classes)]
            images.append(np.array(image))
            labels.append(np.where(np.array(one_hot) == 1)[0][0])

    return np.array(images).reshape(len(images), width, height), np.array(labels)


def _maybe_to_tensors(x_train, x_test, window_shape, block_shape, as_tensors):
    if not as_tensors:
        return x_train, x_test
    train_tensor = utils.process_array_to_pytorch(x_train, window_shape, block_shape)
    test_tensor = utils.process_array_to_pytorch(x_test, window_shape, block_shape)
    return train_tensor, test_tensor
