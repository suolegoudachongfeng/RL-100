import copy

import torch

from rl_100.model.diffusion.ema_model import EMAModel


def test_step_copies_batchnorm_buffers():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model)

    model.train()
    model(torch.full((8, 3), 5.0))
    ema.step(model)

    model_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    assert model_buffers.keys() == ema_buffers.keys()
    for name in model_buffers:
        torch.testing.assert_close(ema_buffers[name], model_buffers[name])
