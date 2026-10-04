"""The conditioned model (models/conditioned.py): one channel per asked element, and queries do not see each other."""
import numpy as np
import torch

from sportcal import sports
from sportcal.models.conditioned import ConditionedUNet, query_inputs, raster
from sportcal.models.kpline import n_channels


def test_queries_match_model_a_channels_and_raster_is_template_only():
    for name in ("hockey-nhl", "soccer-fifa", "tennis-itf", "basketball-fiba"):
        s = sports.get(name)
        samples, extra = query_inputs(s, s.keypoints())
        assert len(samples) == len(extra) == n_channels(s, s.keypoints())
        assert np.abs(samples).max() <= 1 and raster(s).sum() > 50           # inside the drawing, something drawn
    assert not torch.equal(raster(sports.get("hockey-nhl")), raster(sports.get("tennis-itf")))


def test_permuting_queries_permutes_outputs():
    s = sports.get("hockey-nhl")
    samples, extra = (torch.from_numpy(a)[None] for a in query_inputs(s, s.keypoints()))
    net = ConditionedUNet(pretrained=False).eval()
    x, r = torch.randn(1, 3, 64, 96), raster(s)[None]
    perm = torch.randperm(samples.shape[1])
    with torch.no_grad():
        a = net(x, samples, extra, r)
        b = net(x, samples[:, perm], extra[:, perm], r)
    assert torch.allclose(a[:, perm], b, atol=1e-4)
