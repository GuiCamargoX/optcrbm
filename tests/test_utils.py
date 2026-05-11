import numpy as np

from optcrbm.data import preprocessing as utils


def test_trim_array_maxpool_trims_to_pooling_compatible_shape():
    array = np.arange(28 * 28).reshape(28, 28)

    trimmed = utils.trim_array_maxpool(
        arr=array,
        conv_window_shape=(10, 10),
        pooling_shape=(2, 2),
    )

    assert trimmed.shape == (27, 27)


def test_process_array_to_pytorch_adds_channel_dimension():
    dataset = np.zeros((3, 16, 16))

    tensor = utils.process_array_to_pytorch(dataset, (5, 5), (2, 2))

    assert tuple(tensor.shape) == (3, 1, 16, 16)
