"""The template-conditioned keypoint + line model (ADR 0005, "model C").

Model A (`kpline.py`) has one output channel per named element of ONE sport's template, so a new sport needs a new last
layer and labels. This model is told WHICH elements to find, by looking at a drawing of the field:

    frame ──► U-Net ──► feature map F (16 numbers per pixel)                      "what is at each pixel"
    template drawing ──► small CNN ──► feature map T                              "what the field looks like"
    one QUERY per element (a point, or one visible end of a line):
        look up T where the element sits on the field  ──► small MLP ──► 16 weights + 1 bias   (a 1x1 kernel)
    heatmap of the element = sigmoid( weights . F(x, y) + bias )

So the heatmap of "the left face-off dot" is made by a kernel that was computed from "how the left face-off dot looks on
the drawing and where it is". The output has the same shape as model A's (one heatmap per element, same channel order),
so `kpline.estimate_H` and the shared solver are used unchanged.

Queries never see each other: the heatmap of an element does not depend on which other elements are asked, nor in what
order. That is what lets the model be asked for the elements of a sport it was never trained on.

World frame, element names and the template come from `sports.Sport`; this file knows nothing about any one sport.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from sportcal.models.unet import HalfResUNet

RASTER = (256, 128)      # (width, height) in px of the template drawing; every sport is fitted into this same canvas
PAD = 8                  # empty border of the drawing, px, so lines on the field edge are not cut
SAMPLES = 16             # points sampled along a line to describe it (a point is described by 16 copies of itself)
FEAT = 16                # channels of the frame feature map F, i.e. the size of each dynamic kernel
TFEAT = 32               # channels of the template feature map T
QDIM = TFEAT + 6         # what the MLP reads per query: the template feature (32) + 6 numbers, see `query_inputs`


# ----------------------------------------------------------------------------------------------- template -> numbers

def _to_canvas(sport, pts):
    """World metres -> pixels of the template drawing. The field is centred, scaled to fit the canvas and keeps its
    aspect ratio, so a 105 m pitch and a 15 m wide court end up in the same 256x128 picture. X stays along the length."""
    x0, x1, y0, y1 = sport.box
    w, h = RASTER
    scale = min((w - 2 * PAD) / (x1 - x0), (h - 2 * PAD) / (y1 - y0))
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.stack([(pts[:, 0] - (x0 + x1) / 2) * scale + w / 2, (pts[:, 1] - (y0 + y1) / 2) * scale + h / 2], 1)


def raster(sport):
    """(1, 128, 256) float32 drawing of the sport's painted lines, white on black, seen from above.
    ponytail: one channel for every line class; a sport whose line colours matter would need one channel per class."""
    img = np.zeros(RASTER[::-1], np.uint8)
    for _, line in sport.polylines():
        px = np.round(_to_canvas(sport, line) * 8).astype(np.int32)       # *8: cv2 sub-pixel, shift=3
        cv2.polylines(img, [px], False, 255, 1, cv2.LINE_AA, shift=3)
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def query_inputs(sport, kp):
    """What each output channel needs to know about its element. Returns, in the same order as model A's channels
    (all points first, then two channels per straight line, the a-side end and the b-side end):

    samples (Q, 16, 2): where to look on the drawing, as grid_sample coordinates in [-1, 1].
                        A point looks at itself; a line looks at 16 points along a -> b.
    extra   (Q, 6)    : [ax, ay, bx, by, side, is_line]. a and b are the element's end points on the field,
                        in [-1, 1] over the field box (a point has a == b). `side` is 0 for the a end and 1 for the
                        b end of a line; `is_line` tells a line end from a point. The position is what separates
                        mirror-image elements (left vs right dot) that look identical on the drawing."""
    x0, x1, y0, y1 = sport.box
    half = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    norm = lambda p: (np.asarray(p, float) - mid) / half                  # world -> [-1, 1] over the field
    canvas = lambda p: _to_canvas(sport, p) / np.array(RASTER) * 2 - 1    # world -> [-1, 1] over the drawing
    samples, extra = [], []
    for p in np.asarray(kp, float).reshape(-1, 2):
        samples.append(np.repeat(canvas(p), SAMPLES, 0))
        extra.append([*norm(p), *norm(p), 0.0, 0.0])
    for a, b in sport.straight_lines():
        t = np.linspace(0.0, 1.0, SAMPLES)[:, None]
        along = canvas(np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t)
        for side in (0.0, 1.0):                                           # the two ends share their description
            samples.append(along)
            extra.append([*norm(a), *norm(b), side, 1.0])
    return np.asarray(samples, np.float32), np.asarray(extra, np.float32)


# ----------------------------------------------------------------------------------------------------------- network

class ConditionedUNet(HalfResUNet):
    """`HalfResUNet` whose last 1x1 layer is replaced by kernels computed from the template (see the module docstring).
    The frame part (stem, e1..e4, d1..d4, final.0) keeps the names of model A, so a model-A checkpoint loads into it
    with `load_backbone`."""

    def __init__(self, pretrained=True):
        super().__init__(ncls=1, pretrained=pretrained)
        self.final = nn.Sequential(self.final[0])      # keep Block(32, 16) -> F; drop the 1x1 conv to one class
        # reads the drawing: stride 1, 2, 2, 1, then a dilated conv for a wider view; output is 64x32 cells
        self.tenc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 2, dilation=2), nn.ReLU(True),
            nn.Conv2d(32, TFEAT, 3, 1, 1))
        # query -> kernel: FEAT weights and one bias
        self.qmlp = nn.Sequential(nn.Linear(QDIM, 128), nn.ReLU(True), nn.Linear(128, 128), nn.ReLU(True),
                                  nn.Linear(128, FEAT + 1))
        with torch.no_grad():
            self.qmlp[-1].weight.mul_(0.1)
            self.qmlp[-1].bias.zero_()
            self.qmlp[-1].bias[FEAT] = -4.6    # CenterNet's prior: every pixel starts at p = 0.01 (see train_kpline)

    def load_backbone(self, path):
        """Load the frame part from a model-A checkpoint (e.g. the hockey best_h.pt). The old 1x1 output layer is
        skipped; the template CNN and the query MLP stay as initialised."""
        state = {k: v for k, v in torch.load(path, map_location="cpu").items() if not k.startswith("final.1.")}
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith(("tenc.", "qmlp.")) for k in missing), (missing, unexpected)

    def forward(self, x, samples, extra, rast):
        """x (B, 3, H, W) frames; samples (B, Q, 16, 2) and extra (B, Q, 6) from `query_inputs`; rast (B, 1, 128, 256)
        from `raster`. Returns logits (B, Q, H/2, W/2): one heatmap per query."""
        F = super().forward(x)                                            # (B, 16, H/2, W/2)  what is at each pixel
        T = self.tenc(rast)                                               # (B, 32, 32, 64)    what the field looks like
        # read T at each query's sample points and average them: one 32-number description per query
        feat = Fn.grid_sample(T, samples, mode="bilinear", align_corners=False)    # (B, 32, Q, 16)
        q = torch.cat([feat.mean(-1).transpose(1, 2), extra], -1)         # (B, Q, 38)
        k = self.qmlp(q)                                                  # (B, Q, 17): the dynamic 1x1 kernels
        # the 1x1 convolution with a different kernel per query: a dot product at every pixel
        return torch.einsum("bqc,bchw->bqhw", k[..., :FEAT], F) + k[..., FEAT, None, None]


@torch.no_grad()
def heatmaps(model, x, sport, kp):
    """(Q, H/2, W/2) sigmoid heatmaps of one frame tensor x (3, H, W) for the elements `kp` + lines of `sport`: the
    inference call, for any sport (including one the model never saw)."""
    dev = next(model.parameters()).device
    samples, extra = (torch.from_numpy(a)[None].to(dev) for a in query_inputs(sport, kp))
    return torch.sigmoid(model(x[None].to(dev), samples, extra, raster(sport)[None].to(dev)))[0].float().cpu().numpy()
