import torch

from optcrbm.data.loaders import load_semeion, read_semeion


def test_read_semeion_returns_images_and_labels():
    images, labels = read_semeion()

    assert images.shape == (1593, 16, 16)
    assert labels.shape == (1593,)
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_load_semeion_returns_crbm_tensors():
    train_tensor, test_tensor = load_semeion((5, 5), (2, 2))

    assert train_tensor.shape == (1274, 1, 16, 16)
    assert test_tensor.shape == (319, 1, 16, 16)
    assert train_tensor.dtype == torch.float32
    assert test_tensor.dtype == torch.float32
