"""Descarga tramos de partidos de YouTube y saca las semillas de HockeyRink de cada uno.

Un tramo por URL (por defecto 4 minutos a partir del minuto 4), a 1080p como mucho, y deja
nhl4.mp4, nhl5.mp4, ... junto con su rink_preds_nhlN.json. Reanudable: salta lo ya descargado.

    python fetch_clips.py                       # las URLS de abajo
    python fetch_clips.py --start 4 --dur 4     # otro tramo
    python fetch_clips.py --urls <url1> <url2>  # otras URLs
    python fetch_clips.py --first-index 8       # empieza a numerar en nhl8.mp4
"""
import argparse
import os
import subprocess
import sys

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if "NVIDIA\\CUDNN" not in p
)

from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

import rink_reference as rr

ROOT = Path(__file__).resolve().parent
URLS = [
    "<video-url>",
    "<video-url>",
    "<video-url>",
    "<video-url>",
]

ap = argparse.ArgumentParser()
ap.add_argument("--urls", nargs="*", default=URLS)
ap.add_argument("--start", type=float, default=4.0, help="minuto inicial del tramo")
ap.add_argument("--dur", type=float, default=4.0, help="duracion del tramo en minutos")
ap.add_argument("--first-index", type=int, default=4, help="nhl<N>.mp4 del primer video")
ap.add_argument("--every", type=int, default=15, help="frames entre semillas")
args = ap.parse_args()


def hhmmss(minutes):
    s = int(round(minutes * 60))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


section = f"*{hhmmss(args.start)}-{hhmmss(args.start + args.dur)}"
model = None
for k, url in enumerate(args.urls):
    out = ROOT / f"nhl{args.first_index + k}.mp4"
    if not out.exists():
        ytdlp = Path(sys.executable).parent / "yt-dlp.exe"   # el del venv que lanza este script
        cmd = [str(ytdlp) if ytdlp.exists() else "yt-dlp", "-f", "bv*[height<=1080][ext=mp4]", "--download-sections", section,
               "--force-keyframes-at-cuts", "-o", str(out), url]
        print("[dl]", " ".join(cmd), flush=True)
        if subprocess.call(cmd) != 0 or not out.exists():
            print(f"[!] fallo la descarga de {url}, sigo con el siguiente")
            continue
    if (ROOT / f"rink_preds_{out.stem}.json").exists():
        print(f"[seeds] {out.name} ya tiene semillas, salto")
        continue
    if model is None:
        model = YOLO(hf_hub_download("SimulaMet-HOST/HockeyRink", "HockeyRink.pt"))
    rr.video_seeds(model, out.name, args.every)
