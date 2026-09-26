"""The keypoint + line model as a product estimator, for any sport it was trained on (`models/kpline.py`)."""
from sportcal import sports
from sportcal.core.camera import is_plausible_view
from sportcal.product.pipeline import Estimate


class KplineEstimator:
    """Heatmaps of the template keypoints and of the ends of the straight lines -> points + lines DLT, refused when no
    real camera gives that H (`core.camera.is_plausible_view`). The best measured method (ADR 0003). Torch is imported
    here, not with the product package."""
    name = "kpline"

    def __init__(self, weights, sport="hockey-nhl", device="cuda"):
        import torch
        from sportcal.models import kpline as K

        self._K, self._torch = K, torch
        self.sport, self.device = sports.get(sport), torch.device(device)
        self.model, self.kp = K.load(weights, self.sport, self.device)

    def estimate(self, frame):
        h, w = frame.shape[:2]
        x, _ = self._K.to_input(frame)
        torch, dev = self._torch, self.device
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            heat = torch.sigmoid(self.model(x[None].to(dev)).float()).cpu().numpy()[0]
        H = self._K.estimate_H(heat, self.sport, w, self.kp)
        if H is None or not is_plausible_view(H, w, h, self.sport.box):
            return None
        return Estimate(H, 1.0, self.name)  # the gate is yes/no: no graded confidence to report
