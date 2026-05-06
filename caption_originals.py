import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

INPUT_DIR = Path("/home/haind/Desktop/ControlNet/TranhDongHo")
OUTPUT = Path("/home/haind/Desktop/ControlNet/captions.json")
MODEL_ID = "Salesforce/blip-image-captioning-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
model.eval()

exts = {".png", ".jpg", ".jpeg", ".webp"}
files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in exts)
print(f"Captioning {len(files)} images...")

captions: dict[str, str] = {}
with torch.no_grad():
    for f in tqdm(files):
        img = Image.open(f).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        captions[f.name] = processor.decode(out[0], skip_special_tokens=True).strip()

OUTPUT.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(captions)} captions -> {OUTPUT}")
