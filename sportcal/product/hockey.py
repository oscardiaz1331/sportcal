"""Hockey estimators for the product pipeline, wrapping the methods measured in
sportcal/lab/hockey. They use the lab evaluation logic and thresholds, but the results are
not bit-reproducible against the lab numbers: the RANSAC homography is sensitive to 0.1 px
differences in the keypoints (docs/decisions/0003-product-composition.md, "Verified on real weights").

Heavy dependencies (torch, ultralytics) are imported inside the constructors so that
importing the product package stays cheap.
"""
from pathlib import Path

import numpy as np

from sportcal.core.camera import is_plausible_view
from sportcal.core.geometry import REF_WIDTH, fit_homography_ransac
from sportcal.product.pipeline import Estimate, HomographyPipeline
from sportcal.sports import get as get_sport
from sportcal.sports.hockey import rink


class YoloKeypointEstimator:
    """56-keypoint YOLO pose model -> RANSAC homography. ~100% coverage, ~100 px median error.

    The best box is always taken, WITHOUT filtering on box confidence: the detection head
    is unreliable (median conf 0.59) and filtering loses ~42% of frames for no gain in
    keypoint quality (docs/experiments/hockey.md section 2).
    """
    name = "yolo-keypoints"

    def __init__(self, weights, sport="hockey-nhl", conf=0.3, min_kpts=6, ransac_px=8.0,
                 imgsz=1024, device=0, box_conf=0.001):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.template = get_sport(sport).keypoints()
        self.conf, self.min_kpts, self.ransac_px = conf, min_kpts, ransac_px
        self.imgsz, self.device, self.box_conf = imgsz, device, box_conf

    def estimate(self, frame):
        r = self.model.predict(source=frame, imgsz=self.imgsz, device=self.device,
                               verbose=False, conf=self.box_conf)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return None
        xy = r.keypoints.xy[0].cpu().numpy()
        cf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        sel = cf >= self.conf
        if sel.sum() < self.min_kpts:
            return None
        scale = REF_WIDTH / frame.shape[1]  # the RANSAC threshold is defined at 1920 px
        H, inliers = fit_homography_ransac(self.template[sel], xy[sel], self.ransac_px / scale)
        return None if H is None else Estimate(H, float(inliers.mean()), self.name)


class SegDltEstimator:
    """Line-segmentation network -> DLT + refinement. Precise where it answers (15-56 px)
    but answers on a few percent of frames: it refuses unless >= 5 correspondences are in
    view AND the refinement cost is low. The cost gate is NOT a reliable safety net on its
    own (docs/decisions/0002-seg-cost-gate.md): keep a fallback behind it.
    """
    name = "seg-dlt"
    MAX_COST = 0.0009

    def __init__(self, weights, rink_params=rink.RINK_NHL, device="cuda", max_cost=MAX_COST):
        # ponytail: the network and solver still live in sportcal.lab.hockey, imported lazily.
        # Ceiling: product depends on lab code. Upgrade: promote UNetResNet34 and
        # solve_from_probs into sportcal/models + core once the lab port lands (docs/porting-status.md).
        from sportcal.lab.hockey import diagnose_lines_seg as ds
        from sportcal.lab.hockey.seg_to_homography import solve_from_probs

        ds.PESOS = Path(weights)
        self._ds, self._solve = ds, solve_from_probs
        self.model, self.imgsz = ds.load_model(device)
        self.device, self.params, self.max_cost = device, rink_params, max_cost

    def estimate(self, frame):
        h, w = frame.shape[:2]
        probs = self._ds.predict_probs(self.model, frame, self.imgsz, self.device)
        H, _, _, cost = self._solve(probs, self.params, w, h)
        if H is None or cost > self.max_cost:
            return None
        return Estimate(H, 1.0 - cost / self.max_cost, self.name)


class KplineEstimator:
    """Heatmaps of the template keypoints and of the ends of the straight lines -> points + lines DLT, refused when no
    real camera gives that H (`core.camera.is_plausible_view`). The best measured method (ADR 0003)."""
    name = "kpline"

    def __init__(self, weights, sport="hockey-nhl", device="cuda"):
        # ponytail: same debt as SegDltEstimator: HalfResUNet and estimate_H live in sportcal.lab.hockey.train_kpline,
        # imported lazily. Upgrade: promote them to sportcal/models + core with the lab port (docs/porting-status.md).
        import torch
        from sportcal.lab.hockey import train_kpline as K

        self._K, self._torch = K, torch
        self.kp, self.sport, self.device = K.keypoints("base"), sport, torch.device(device)
        self.model = K.HalfResUNet(ncls=K.n_channels(self.kp, sport), pretrained=False).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=self.device))

    def estimate(self, frame):
        h, w = frame.shape[:2]
        x, _ = self._K.to_input(frame)
        torch, dev = self._torch, self.device
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            heat = torch.sigmoid(self.model(x[None].to(dev)).float()).cpu().numpy()[0]
        H = self._K.estimate_H(heat, self.sport, w, self.kp)
        if H is None or not is_plausible_view(H, w, h, self._K.BOX[self.sport]):
            return None
        return Estimate(H, 1.0, self.name)  # the gate is yes/no: no graded confidence to report


def build_pipeline(yolo_weights, seg_weights=None, sport="hockey-nhl", hold_frames=0, kpline_weights=None,
                   device="cuda:0"):
    """Hockey composition (ADR 0003), first answer wins: the keypoint + line model, segmentation-DLT, YOLO keypoints as
    the full-coverage fallback. Each stage is skipped when its weights are None."""
    stages = []
    if kpline_weights is not None:
        stages.append(KplineEstimator(kpline_weights, sport=sport, device=device))
    if seg_weights is not None:
        stages.append(SegDltEstimator(seg_weights, rink_params=rink.RINK_NHL if sport == "hockey-nhl"
                                      else rink.RINK_IIHF, device=device))
    if yolo_weights is not None:
        stages.append(YoloKeypointEstimator(yolo_weights, sport=sport, device=device))
    return HomographyPipeline(stages, hold_frames=hold_frames)
