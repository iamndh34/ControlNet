# Hướng dẫn Training ControlNet cho Tranh Dân Gian Việt Nam

## Tóm tắt

Script này giúp bạn train ControlNet để sinh tranh dân gian Việt Nam từ sketch/phác thảo.

## Các file đã tạo

| File | Mô tả |
|------|-------|
| `prepare_vietnamese_dataset.py` | Download dataset từ HF và tạo edge maps |
| `viet_folk_art_dataset.py` | Custom Dataset class |
| `train_viet_folk_art.py` | Training script |
| `test_dataset_preview.py` | Preview dataset trước khi train |

---

## Bước 1: Chuẩn bị Dataset

Dataset sẽ được tự động download từ HuggingFace và tạo edge maps.

```bash
# Tạo dataset với 5000 ảnh (để test trước)
python prepare_vietnamese_dataset.py \
    --output_dir ./training/vietnamese_folk_art \
    --num_samples 5000 \
    --edge_method canny \
    --image_size 512

# Hoặc dùng toàn bộ dataset (~28,000 ảnh)
python prepare_vietnamese_dataset.py \
    --output_dir ./training/vietnamese_folk_art \
    --edge_method canny
```

**Các tham số:**
- `--output_dir`: Thư mục output (mặc định: `./training/vietnamese_folk_art`)
- `--num_samples`: Số lượng ảnh (None = toàn bộ)
- `--edge_method`: Phương pháp tạo edge (`canny`, `sobel`, `hed`)
- `--image_size`: Kích thước ảnh (mặc định: 512)

**Cấu trúc thư mục sau khi chạy:**
```
training/vietnamese_folk_art/
├── prompt.json
├── source/
│   ├── 000000.png  # Edge maps
│   ├── 000001.png
│   └── ...
└── target/
    ├── 000000.png  # Ảnh gốc
    ├── 000001.png
    └── ...
```

---

## Bước 2: Preview Dataset (Optional)

Kiểm tra dataset trước khi train:

```bash
python test_dataset_preview.py \
    --data_root ./training/vietnamese_folk_art \
    --num_samples 4
```

File `dataset_preview.png` sẽ được tạo để bạn xem kết quả.

---

## Bước 3: Download Stable Diffusion 1.5

1. Truy cập: https://huggingface.co/runwayml/stable-diffusion-v1-5
2. Download file `v1-5-pruned.ckpt`
3. Lưu vào thư mục `./models/v1-5-pruned.ckpt`

Hoặc dùng git-lfs:
```bash
pip install huggingface_hub
huggingface-cli download runwayml/stable-diffusion-v1-5 v1-5-pruned.ckpt --local-dir ./models
```

---

## Bước 4: Attach ControlNet vào SD

```bash
cd ControlNet
python tool_add_control.py \
    ../models/v1-5-pruned.ckpt \
    ../models/control_sd15_ini.ckpt
cd ..
```

File `control_sd15_ini.ckpt` sẽ được tạo - đây là model khởi đầu cho training.

---

## Bước 5: Training

### Training cơ bản (GPU 8-12GB VRAM):

```bash
python train_viet_folk_art.py \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 10
```

### Training với VRAM thấp (4-6GB):

```bash
python train_viet_folk_art.py \
    --batch_size 1 \
    --accumulate_grad_batches 4 \
    --precision 16 \
    --num_workers 2
```

### Training đa GPU:

```bash
python train_viet_folk_art.py \
    --batch_size 8 \
    --gpus 2 \
    --accumulate_grad_batches 2
```

**Các tham số quan trọng:**
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--batch_size` | 4 | Batch size per GPU |
| `--learning_rate` | 1e-5 | Learning rate |
| `--sd_locked` | True | Freeze SD weights (khuyến nghị) |
| `--only_mid_control` | False | Chỉ control middle layers (nhanh hơn) |
| `--accumulate_grad_batches` | 1 | Gradient accumulation |
| `--precision` | 32 | 16 hoặc 32 (bit) |
| `--max_epochs` | 10 | Số epochs |

---

## Bước 6: Testing với Gradio

Sau khi train xong, model sẽ được lưu ở `./logs/vietnamese_folk_art/`.

1. Sửa file `ControlNet/gradio_scribble2image.py`:
   ```python
   # Dòng 18: Thay đường dẫn đến model đã train
   model.load_state_dict(load_state_dict('../logs/vietnamese_folk_art/last.ckpt', location='cuda'))
   ```

2. Chạy Gradio:
   ```bash
   cd ControlNet
   python gradio_scribble2image.py
   ```

---

## Tips & Troubleshooting

### VRAM không đủ
- Giảm `--batch_size` xuống 1 hoặc 2
- Tăng `--accumulate_grad_batches` lên 4 hoặc 8
- Dùng `--precision 16`

### Training quá chậm
- Tăng `--num_workers` (số CPU cores)
- Dùng `--only_mid_control` để train nhanh hơn
- Tăng batch size nếu VRAM cho phép

### Model không tốt
- Train lâu hơn (tăng `--max_epochs`)
- Dùng nhiều data hơn (tăng `--num_samples`)
- Thử giảm learning rate: `--learning_rate 5e-6`
- Unlock SD weights: `--no_sd_locked` (rủi ro nhưng có thể tốt hơn)

### Sudden Convergence
Theo docs, ControlNet sẽ "bất ngờ" hội tụ ở khoảng 3000-7000 steps.
- Sau bước này, model cơ bản đã dùng được
- Tiếp tục training sẽ cải thiện chất lượng

---

## Monitoring Training

Training logs và checkpoints được lưu tại:
```
logs/vietnamese_folk_art/
├── image_log/           # Preview images mỗi 300 steps
├── checkpoints/         # Model checkpoints
└── last.ckpt           # Checkpoint mới nhất
```

---

## Ví dụ Prompts cho Tranh Dân Gian Việt Nam

```
Vietnamese traditional Dong Ho folk painting
Vietnamese water puppet show scene
Traditional Vietnamese village festival
Ancient Vietnamese pagoda architecture
Vietnamese Tet holiday celebration
Traditional Vietnamese market scene
Vietnamese rice farming landscape
Traditional Vietnamese wedding ceremony
Vietnamese Hue royal palace architecture
Vietnamese bamboo craft scene
```

---

## License

Dataset Vietnamese Cultural VQA: Apache 2.0
ControlNet: Apache 2.0
