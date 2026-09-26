"""What the SoccerNet frames left out of the H index (`soccernet_h`) are annotated with: per frame, the straight markings
with at least 2 points (what the index fit can use) and the points on each circle. Counts only, soccer.md section 21.

    python -m sportcal.lab.soccer.unused_gt --root datasets/calibration-2023
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from sportcal.lab.soccer.soccernet_h import CIRCLES, LINES, OUT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    used = {json.loads(line)["id"] for line in open(OUT, encoding="utf-8")}
    by_lines, circles, examples = Counter(), Counter(), {}
    for split_dir in ("train", "valid", "test"):
        for js in sorted((Path(args.root) / split_dir).glob("*.json")):
            if "soccernet/{}/{}".format(split_dir, js.stem) in used:
                continue
            ann = json.load(open(js, encoding="utf-8"))
            n_lines = sum(len(ann.get(k, [])) >= 2 for k in LINES)
            circ = {k.split()[-1]: len(ann.get(k, [])) for k in CIRCLES if ann.get(k)}
            key = ("{} lines".format(min(n_lines, 4) if n_lines < 4 else "4+"),
                   "centre circle" if "central" in circ else "penalty arc" if circ else "no circle")
            by_lines[key] += 1
            examples.setdefault(key, "{}/{}".format(split_dir, js.stem))
            for k, n in circ.items():
                circles[(k, "5+" if n >= 5 else str(n))] += 1
    print("{} frames not in the index".format(sum(by_lines.values())))
    for (lines, circ), n in sorted(by_lines.items()):
        print("  {:>8}  {:<14} {:>6}   e.g. {}".format(lines, circ, n, examples[(lines, circ)]))
    print("points on each circle, when annotated:")
    for (k, n), c in sorted(circles.items()):
        print("  {:<8} {:>3} points {:>6}".format(k, n, c))


if __name__ == "__main__":
    main()
