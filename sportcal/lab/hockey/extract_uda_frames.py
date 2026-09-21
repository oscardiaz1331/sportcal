"""Extrae frames SIN etiquetar de los videos propios para la adaptacion de dominio
(train_lines_seg.py --uda; MIC del estudio, etapa 5). Uno cada ~1.5 s, a la
resolucion de entrenamiento, en datasets/uda_target/images/train/.

Se excluyen: nhl6 (pista amateur espanola, otra geometria -- seccion 7bis) y
cualquier frame a menos de 300 frames de un frame de VAL del mismo video, para
que la adaptacion no vea vecinos casi identicos de lo que luego se mide.

    python -m sportcal.lab.hockey.extract_uda_frames
"""
import re
import sys
from pathlib import Path

import cv2

from sportcal.paths import ROOT
VIDEOS = ["nhl3", "nhl4", "nhl5", "nhl7", "nhl8", "nhl9", "nhl10"]
W, H = 1024, 576
GAP = 300


def val_indices():
    out = {}
    for ds in ("hockeyrink", "hockeyrink_nhl"):
        for p in (ROOT / "datasets" / ds / "images" / "val").glob("*.jpg"):
            m = re.match(r"(nhl\d+)_(\d+)$", p.stem)
            if m:
                out.setdefault(m.group(1), []).append(int(m.group(2)))
    return out


def main():
    vals = val_indices()
    out = ROOT / "datasets" / "uda_target" / "images" / "train"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for v in VIDEOS:
        cap = cv2.VideoCapture(str(ROOT / (v + ".mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = int(round(fps * 1.5))
        bad = vals.get(v, [])
        i, n = 0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0 and not any(abs(i - b) < GAP for b in bad):
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "{}_{:06d}.jpg".format(v, i)), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
            i += 1
        cap.release()
        print("{:6s} step={:3d}  val en video={}  extraidos {}".format(v, step, len(bad), n))
        total += n
    print("total", total, "->", out)


if __name__ == "__main__":
    main()
