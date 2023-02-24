import pytest
import torch
import numpy as np
import torchtuples as tt
from pycox.models import PCHazard

from utils_model_testing import make_dataset, fit_model, assert_survs


def _make_dataset(n, m):
    np.random.seed(0)
    x = np.random.normal(0, 1, (n, 4)).astype('float32')
    duration_index = np.arange(m+1).astype('int64')
    durations = np.repeat(duration_index, np.ceil(n / m))[:n]
    events = np.random.uniform(0, 1, n).round().astype('float32')
    fracs = np.random.uniform(0, 1, n).astype('float32')
    return x, (durations, events, fracs), duration_index

@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n_mul', [2, 3])
@pytest.mark.parametrize('mp', [1, 2, -1])
def test_wrong_net_output(m, n_mul, mp):
    n = m * n_mul
    inp, tar, dur_index = _make_dataset(n, m)
    net = torch.nn.Linear(inp.shape[1], m+1)
    with pytest.raises(ValueError):
        model = PCHazard(net, duration_index=dur_index)

    model = PCHazard(net)
    with pytest.raises(ValueError):
        model.fit(inp, tar)

    model.duration_index = dur_index
    with pytest.raises(ValueError):
        model.predict_surv_df(inp)

    model.duration_index = dur_index
    dl = model.make_dataloader((inp, tar), 5, True)
    with pytest.raises(ValueError):
        model.fit_dataloader(dl)

@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n_mul', [2, 3])
def test_right_net_output(m, n_mul):
    n = m * n_mul
    inp, tar, dur_index = _make_dataset(n, m)
    net = torch.nn.Linear(inp.shape[1], m)
    model = PCHazard(net)
    model = PCHazard(net, duration_index=dur_index)
    model.fit(inp, tar, verbose=False)
    model.predict_surv_df(inp)
    dl = model.make_dataloader((inp, tar), 5, True)
    model.fit_dataloader(dl)
    assert True

@pytest.mark.parametrize('numpy', [True, False])
@pytest.mark.parametrize('num_durations', [3, 8])
def test_pc_hazard_runs(numpy, num_durations):
    data = make_dataset(True)
    input, (durations, events) = data
    durations += 1
    target = (durations, events)
    labtrans = PCHazard.label_transform(num_durations)
    target = labtrans.fit_transform(*target)
    data = tt.tuplefy(input, target)
    if not numpy:
        data = data.to_tensor()
    net = tt.practical.MLPVanilla(input.shape[1], [4], num_durations)
    model = PCHazard(net)
    fit_model(data, model)
    assert_survs(input, model)
    model.duration_index = labtrans.cuts
    assert_survs(input, model)
