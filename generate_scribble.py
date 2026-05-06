from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path("/home/haind/Desktop/ControlNet/ControlNet/training/dongho")
TARGET_DIR = ROOT / "target"
SOURCE_DIR = ROOT / "source"

THRESHOLD = 90

SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def to_scribble(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


files = sorted(TARGET_DIR.glob("*.png"))
print(f"Generating scribbles for {len(files)} targets (threshold={THRESHOLD})...")
for f in tqdm(files):
    img = cv2.imread(str(f))
    if img is None:
        print(f"skip: {f.name}")
        continue
    scribble = to_scribble(img)
    cv2.imwrite(str(SOURCE_DIR / f.name), scribble)

print(f"Wrote scribbles -> {SOURCE_DIR}")
