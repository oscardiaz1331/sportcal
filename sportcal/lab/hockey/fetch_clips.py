"""Download match segments from YouTube and extract the HockeyRink seeds of each one.

One segment per URL (default: 4 minutes starting at minute 4), at most 1080p. Produces
nhl4.mp4, nhl5.mp4, ... plus their rink_preds_nhlN.json. Resumable: skips what exists.

    python -m sportcal.lab.hockey.fetch_clips                       # the URLS below
    python -m sportcal.lab.hockey.fetch_clips --start 4 --dur 4     # another segment
    python -m sportcal.lab.hockey.fetch_clips --urls <url1> <url2>  # other URLs
    python -m sportcal.lab.hockey.fetch_clips --first-index 8       # number from nhl8.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

from sportcal.paths import ROOT

URLS = [
    "<video-url>",
    "<video-url>",
    "<video-url>",
    "<video-url>",
]


def hhmmss(minutes):
    s = int(round(minutes * 60))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="*", default=URLS)
    ap.add_argument("--start", type=float, default=4.0, help="segment start, minutes")
    ap.add_argument("--dur", type=float, default=4.0, help="segment length, minutes")
    ap.add_argument("--first-index", type=int, default=4, help="nhl<N>.mp4 of the first video")
    ap.add_argument("--every", type=int, default=15, help="frames between seeds")
    args = ap.parse_args(argv)

    # heavy imports only when actually run
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    from sportcal.lab.hockey import rink_reference as rr

    section = f"*{hhmmss(args.start)}-{hhmmss(args.start + args.dur)}"
    model = None
    for k, url in enumerate(args.urls):
        out = ROOT / f"nhl{args.first_index + k}.mp4"
        if not out.exists():
            ytdlp = Path(sys.executable).parent / "yt-dlp.exe"  # the one from the venv running this
            cmd = [str(ytdlp) if ytdlp.exists() else "yt-dlp", "-f", "bv*[height<=1080][ext=mp4]",
                   "--download-sections", section, "--force-keyframes-at-cuts", "-o", str(out), url]
            print("[dl]", " ".join(cmd), flush=True)
            if subprocess.call(cmd) != 0 or not out.exists():
                print(f"[!] download failed for {url}, continuing with the next one")
                continue
        if (ROOT / f"rink_preds_{out.stem}.json").exists():
            print(f"[seeds] {out.name} already has seeds, skipping")
            continue
        if model is None:
            model = YOLO(hf_hub_download("SimulaMet-HOST/HockeyRink", "HockeyRink.pt"))
        rr.video_seeds(model, out.name, args.every)


if __name__ == "__main__":
    main()
