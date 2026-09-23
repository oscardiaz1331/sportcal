"""Run the public PnLCalib soccer calibration (github.com/mguti97/PnLCalib, GPL-2.0) on a folder of images and
dump the camera it finds for each one to JSON.

Runs in PnLCalib's OWN environment, never in the sportcal venv: PnLCalib's requirements pin `opencv-python`,
which overwrites opencv-contrib's cv2 and removes cv2.ximgproc (docs/experiments/soccer.md section 7). It imports
nothing from sportcal; `pnlcalib_eval.py` (sportcal venv) converts the output to our world convention and scores it.
The process boundary is also the licence boundary: this is the only file that touches GPL code.

    <pnlcalib-python> sportcal/lab/soccer/pnlcalib_run.py --pnl-dir ../PnLCalib --weights-dir ../PnLCalib/weights \
        --images datasets/soccer_labels/images --out runs/pnlcalib/soccer_labels.json

Output: {"<image stem>": {"P": 3x4 projection in the SoccerNet world (origin at the centre spot, x along the length,
y TOWARDS the main camera, z down) or null, "rep_err": PnLCalib's own reprojection error, "mode": which of its
candidate solvers won, "seconds": wall time}}.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pnl-dir", required=True, help="clone of github.com/mguti97/PnLCalib")
    ap.add_argument("--weights-dir", required=True, help="folder with the SV_kp and SV_lines release files")
    ap.add_argument("--images", required=True, help="folder of .jpg/.png frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-refine", action="store_true", help="skip the PnL (points+lines) refinement")
    ap.add_argument("--kp-threshold", type=float, default=0.3434)    # PnLCalib's inference.py defaults
    ap.add_argument("--line-threshold", type=float, default=0.7867)
    args = ap.parse_args()

    pnl = Path(args.pnl_dir).resolve()
    sys.path.insert(0, str(pnl))  # PnLCalib is a folder of scripts, not an installable package
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    import yaml
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (complete_keypoints, coords_to_dict, get_keypoints_from_heatmap_batch_maxpool,
                                     get_keypoints_from_heatmap_batch_maxpool_l)

    def load(get_net, cfg, weights):
        net = get_net(yaml.safe_load(open(pnl / "config" / cfg)))
        net.load_state_dict(torch.load(Path(args.weights_dir) / weights, map_location=args.device))
        return net.to(args.device).eval()

    model = load(get_cls_net, "hrnetv2_w48.yaml", "SV_kp")
    model_l = load(get_cls_net_l, "hrnetv2_w48_l.yaml", "SV_lines")

    frames = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = {}
    for p in frames:
        t0 = time.time()
        bgr = cv2.imread(str(p))
        h0, w0 = bgr.shape[:2]
        x = TF.to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        if x.shape[-1] != 960:
            x = TF.resize(x, [540, 960])  # the networks were trained at 960x540
        with torch.no_grad():
            x = x.to(args.device)
            hm, hm_l = model(x), model_l(x)
        kp = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1]), threshold=args.kp_threshold)
        ln = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1]), threshold=args.line_threshold)
        kp, ln = complete_keypoints(kp[0], ln[0], w=960, h=540, normalize=True)
        cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
        cam.update(kp, ln)
        res = cam.heuristic_voting(refine_lines=not args.no_refine)
        rec = {"P": None, "rep_err": None, "mode": None}
        if res is not None:
            c = res["cam_params"]
            K = np.array([[c["x_focal_length"], 0, c["principal_point"][0]],
                          [0, c["y_focal_length"], c["principal_point"][1]], [0, 0, 1]])
            Rt = np.c_[np.eye(3), -np.asarray(c["position_meters"], float)]
            rec = {"P": (K @ np.asarray(c["rotation_matrix"], float) @ Rt).tolist(),
                   "rep_err": float(res["rep_err"]), "mode": res["mode"]}
        rec["seconds"] = round(time.time() - t0, 2)
        out[p.stem] = rec
        print("{:<16} {:>8} {:>6}s".format(p.stem, "ok" if rec["P"] else "refused", rec["seconds"]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("{} of {} frames calibrated -> {}".format(sum(r["P"] is not None for r in out.values()), len(out), args.out))


if __name__ == "__main__":
    main()
