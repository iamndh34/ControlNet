import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("/home/haind/Desktop/ControlNet")
INPUT_DIR = ROOT / "TranhDongHo"
CAPTIONS = ROOT / "captions.json"
OUT_BASE = ROOT / "ControlNet" / "training" / "dongho"
TARGET_DIR = OUT_BASE / "target"
PROMPT_JSON = OUT_BASE / "prompt.json"

STYLE_PREFIX = "Vietnamese Dong Ho folk woodblock painting"
BRIGHT_DELTA = 20

if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)
TARGET_DIR.mkdir(parents=True)

captions = json.loads(CAPTIONS.read_text(encoding="utf-8"))


def bright(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def variants(img):
    rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot180 = cv2.rotate(img, cv2.ROTATE_180)
    rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    hflip = cv2.flip(img, 1)
    hflip_rot90 = cv2.rotate(hflip, cv2.ROTATE_90_CLOCKWISE)
    hflip_rot180 = cv2.rotate(hflip, cv2.ROTATE_180)
    hflip_rot270 = cv2.rotate(hflip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return [
        img, rot90, rot180, rot270,
        hflip, hflip_rot90, hflip_rot180, hflip_rot270,
        bright(img, BRIGHT_DELTA), bright(img, -BRIGHT_DELTA),
    ]


records = []
counter = 1
files = sorted(INPUT_DIR.glob("*.png"))
for f in tqdm(files):
    img = cv2.imread(str(f))
    if img is None:
        print(f"skip (unreadable): {f.name}")
        continue
    raw = captions.get(f.name, "").strip()
    prompt = f"{STYLE_PREFIX}, {raw}" if raw else STYLE_PREFIX
    for v in variants(img):
        name = f"dongho_{counter:05d}.png"
        cv2.imwrite(str(TARGET_DIR / name), v)
        records.append({
            "source": f"source/{name}",
            "target": f"target/{name}",
            "prompt": prompt,
        })
        counter += 1

with PROMPT_JSON.open("w", encoding="utf-8") as fp:
    for r in records:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} entries -> {PROMPT_JSON}")
print(f"Targets: {TARGET_DIR}")
