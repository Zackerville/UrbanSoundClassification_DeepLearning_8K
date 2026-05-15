---
marp: true
theme: default
paginate: true
html: true
math: katex
header: 'HK251 — Học Sâu — BTL'
footer: 'Ngô Thái Minh Tiến — 2252809'
style: |
  section { font-size: 22px; }
  h1 { color: #1a3d7c; }
  h2 { color: #1a3d7c; }
  h3 { color: #2c5aa0; }
  table { font-size: 18px; }

  /* === COVER / TITLE SLIDE === */
  section.cover {
    background: linear-gradient(135deg, #0a1e3f 0%, #1a3d7c 45%, #2c5aa0 100%);
    color: #ffffff;
    text-align: center;
    justify-content: center;
    padding: 60px 80px;
  }
  section.cover::before {
    content: "🎵";
    font-size: 80px;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));
  }
  section.cover h1 {
    color: #ffffff;
    font-size: 46px;
    line-height: 1.15;
    margin: 0 0 12px 0;
    font-weight: 700;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }
  section.cover h2 {
    color: #ffd166;
    font-size: 24px;
    font-weight: 500;
    margin: 0 0 28px 0;
    letter-spacing: 0.5px;
  }
  section.cover .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 8px 28px;
    font-size: 18px;
    color: #ffffff;
    margin-bottom: 32px;
    backdrop-filter: blur(8px);
  }
  section.cover .info-card {
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #ffd166;
    border-radius: 10px;
    padding: 18px 28px;
    margin: 0 auto;
    max-width: 640px;
    text-align: left;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  }
  section.cover .info-card p {
    margin: 6px 0;
    font-size: 18px;
    color: #ffffff;
  }
  section.cover .info-card strong { color: #87ceeb; }
  section.cover .meta {
    margin-top: 22px;
    font-size: 16px;
    color: rgba(255,255,255,0.7);
    letter-spacing: 1px;
  }

  .pipeline-step {
    background: #e8f0fe;
    border-radius: 8px;
    padding: 10px 20px;
    margin: 8px auto;
    text-align: center;
    max-width: 700px;
    font-size: 20px;
  }
  .pipeline-step.feature { background: #fff4d6; }
  .pipeline-step.output { background: #d4edda; }
  .pipeline-step.model { background: #f8d7da; }
  .pipeline-arrow {
    text-align: center;
    font-size: 24px;
    color: #888;
  }
---

<!-- _class: cover -->
<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Phân loại âm thanh đô thị trên UrbanSound8K

## Log-Mel Spectrogram • CNN • ResNet18 • Augmentation

<div class="badge">Báo cáo Bài tập lớn — Học phần Học Sâu</div>

<div class="info-card">

**👨‍🏫 GVHD:** Thầy Huỳnh Văn Thống

**🎓 Sinh viên:** Ngô Thái Minh Tiến — MSSV 2252809

**🏛️ Khoa:** KH & KT Máy tính, ĐH Bách Khoa TP.HCM

</div>

<p class="meta">HK251 • THÁNG 5, 2026</p>

---

# Nội dung trình bày

1. **Bài toán & Bộ dữ liệu**
2. **EDA & Quyết định tiền xử lý**
3. **Tổng quan tài liệu**
4. **Pipeline & Mô hình**
   - CNN baseline
   - ResNet18
   - ResNet18 + Augmentation
5. **Thiết kế thực nghiệm**
6. **Kết quả & Phân tích**
7. **Kết luận**

---

# 1. Bài toán

## Urban Sound Classification

Cho một đoạn âm thanh đô thị ngắn → **gán nhãn** vào 1 trong 10 lớp.

$$
f(x) \to y, \quad y \in \{1, 2, \dots, 10\}
$$

**Ý nghĩa thực tiễn:**
- Giám sát môi trường, thành phố thông minh
- Phát hiện âm thanh bất thường (súng, còi báo động)
- Phân tích giao thông, quan trắc tiếng ồn

**Phân loại:** Supervised multi-class classification

---

# 2. Bộ dữ liệu — UrbanSound8K

| Thuộc tính | Giá trị |
|---|---|
| Tổng số mẫu | **8732** audio clips |
| Số lớp | **10** |
| Số fold | **10** (sẵn) |
| Duration trung bình | ~3.61 giây |
| Sample rate | 8000 – 192000 Hz (không đồng nhất) |
| Channels | Mono + Stereo |

**10 lớp âm thanh:** air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music

---

# Lý do chọn UrbanSound8K

- ✅ **Nhãn rõ ràng** — phù hợp supervised classification
- ✅ **Đã chia sẵn 10 fold** — dễ tổ chức train/val/test
- ✅ **Kích thước vừa phải** — đủ cho học sâu nhưng vẫn fit Colab/cá nhân
- ✅ **Tiêu chuẩn học thuật** — nhiều paper benchmark trên dataset này

---

# 3. EDA — Đặc điểm dữ liệu gốc

### Vấn đề phát hiện được:
- **Sample rate không đồng nhất:** 8 kHz → 192 kHz
- **Mix mono + stereo:** số kênh khác nhau giữa file
- **Duration không cố định:** trung bình 3.61s nhưng có biến thiên

→ Nếu để nguyên, mô hình sẽ học **format khác biệt** thay vì đặc trưng âm học.

### Lớp dễ vs lớp khó

| Dễ phân biệt | Khó phân biệt (machinery cluster) |
|---|---|
| car_horn, gun_shot, siren | air_conditioner, engine_idling |
| dog_bark | drilling, jackhammer |

→ Phổ năng lượng các âm máy móc rất giống nhau

---

# Quyết định tiền xử lý

| Bước | Lựa chọn | Lý do |
|---|---|---|
| Channels | **Mono** | Đồng nhất, giảm chiều |
| Sample rate | **22050 Hz** | Cân bằng thông tin / compute |
| Duration | **4.0 giây** (88200 mẫu) | Pad/truncate cố định |
| Feature | **log-Mel spectrogram** | Chuẩn cho audio + tương thích CNN |

### Tham số log-Mel
- `n_fft = 1024`
- `hop_length = 512`
- `n_mels = 128`
- `top_db = 80`

---

# 4. Tổng quan tài liệu

### Hướng tiếp cận truyền thống
- **Đặc trưng thủ công:** MFCC, ZCR, spectral centroid…
- **Bộ phân loại:** SVM, k-NN, Random Forest

→ Phụ thuộc mạnh vào đặc trưng được chọn, khó scale

### Hướng Deep Learning
- Biểu diễn: **waveform → spectrogram → Mel → log-Mel**
- Kiến trúc:
  - **CNN** — chuẩn cho spectrogram (2D pattern thời gian–tần số)
  - **ResNet** — residual learning, học mạng sâu hiệu quả
  - **Augmentation:** SpecAugment, Mixup

---

# Pipeline tổng quan

### 🎵 Audio → ⚙️ Preprocess → 📊 Feature → 🧠 Model → ✅ Output

| Bước | Nội dung |
|---|---|
| 🎵 **Input** | WAV file từ UrbanSound8K |
| ⚙️ **Preprocess** | Mono • Resample **22050 Hz** • Pad/Truncate **4.0 s** (88200 mẫu) |
| 📊 **Feature** | MelSpectrogram → AmplitudeToDB → Normalize |
| 🎯 **Tensor** | log-Mel spectrogram `[1 × 128 × 173]` |
| 🧠 **Model** | CNN baseline / ResNet18 / ResNet18 + augmentation |
| ✅ **Output** | Softmax → 1 trong **10 lớp** âm thanh đô thị |

**Tham số log-Mel:** `n_fft=1024` • `hop_length=512` • `n_mels=128` • `top_db=80`

---

# 5. Mô hình — CNN baseline

### Kiến trúc

4 conv blocks (16 → 32 → 64 → 128 channels), mỗi block:
- Conv 3×3 + BatchNorm + ReLU
- MaxPool 2×2 (3 block đầu) / AdaptiveAvgPool (block 4)

Classifier: Flatten → FC(128→64) → Dropout(0.3) → FC(64→10)

### Vai trò
- **Baseline cơ sở** — chuẩn so sánh cho các mô hình phức tạp hơn
- Đơn giản, ít tham số → dễ huấn luyện trên dataset vừa

---

# Mô hình — ResNet18

### Kiến trúc tùy chỉnh cho spectrogram
- **Input 1 channel** (thay vì 3 như RGB)
- **Stem điều chỉnh** — giữ chi tiết cục bộ (Conv 3×3 stride 1, không pool sớm)
- 4 stage × 2 BasicBlock với residual connection
- AdaptiveAvgPool + FC(→10)

### Ý nghĩa
- **Sâu hơn CNN baseline** → học pattern phức tạp hơn
- **Residual** → gradient lan truyền tốt khi mạng sâu
- **From scratch** — không dùng pretrained để đánh giá thuần kiến trúc

---

# Mô hình — ResNet18 + Augmentation

### SpecAugment trên log-Mel
- **Gaussian noise** nhẹ
- **Time shift** (dịch theo trục thời gian)
- **Frequency masking** (che dải tần số)
- **Time masking** (che dải thời gian)

→ Chỉ áp dụng trên **tập train**, validation/test giữ nguyên.

### Mục tiêu
- Tăng tính đa dạng dữ liệu → giảm overfit
- Đặc biệt giúp các **lớp khó** (air_conditioner, drilling…)
- Mô phỏng mất mát thông tin cục bộ → buộc model học đặc trưng bền vững

---

# 6. Thiết kế thực nghiệm

### Cấu hình chung — giữ cố định
- Dataset: UrbanSound8K
- Input: log-Mel spectrogram [1×128×173]
- 1 fold test (fold 1), 1 fold val (fold 2), 8 fold train

### 3 cấu hình so sánh

| ID | Mô hình | Augmentation |
|---|---|---|
| **C1** | CNN baseline | ❌ |
| **C2** | ResNet18 (1-channel, stem điều chỉnh) | ❌ |
| **C3** | ResNet18 + SpecAugment + Mixup | ✅ |

### Chỉ số đánh giá
- **Accuracy** — chính xác tổng thể
- **Macro-F1** — cân bằng giữa các lớp
- **Confusion Matrix** — phân tích lỗi định tính

---

# Quy trình train / eval

### Training
- Optimizer: Adam / AdamW
- Loss: CrossEntropy + label smoothing
- Scheduler: ReduceLROnPlateau / CosineAnnealing
- **Early stopping** trên val_acc
- W&B logging: loss, acc, F1 mỗi epoch

### Evaluation
- Load best checkpoint (theo val_acc)
- Đánh giá trên test set
- Xuất: metrics JSON, classification report CSV, confusion matrix PNG, predictions CSV

### Tính tái lập
- Seed cố định, code modular, log đầy đủ qua W&B

---

# 7. Kết quả tổng quan

| Mô hình | Test Accuracy | Test Macro-F1 |
|---|---|---|
| **CNN baseline** | **0.7446** | 0.7525 |
| **ResNet18** | 0.7297 | 0.7201 |
| **ResNet18 + aug** | 0.7297 | **0.7617** |

### Nhận xét chính

- ⚡ **CNN baseline** đạt Accuracy cao nhất → kiến trúc đơn giản phù hợp dataset
- 🟰 **ResNet18** chưa vượt CNN → kiến trúc sâu hơn ≠ luôn tốt hơn
- 📈 **ResNet18 + aug**: Accuracy bằng ResNet18 **nhưng Macro-F1 tăng rõ rệt** (+4.2 điểm)
  → Augmentation cải thiện **cân bằng giữa các lớp**

---

# CNN baseline — F1 theo lớp

| Lớp | F1 | Lớp | F1 |
|---|---|---|---|
| **car_horn** | 0.900 | drilling | 0.653 |
| **gun_shot** | 0.889 | jackhammer | 0.661 |
| **siren** | 0.878 | **air_conditioner** | **0.313** |
| street_music | 0.827 | engine_idling | 0.795 |
| dog_bark | 0.826 | children_playing | 0.782 |

### Phát hiện
- 🟢 **Class dễ:** car_horn, gun_shot, siren (đặc trưng âm học rõ)
- 🔴 **Class yếu:** **air_conditioner** F1 chỉ 0.31 → âm nền liên tục, dễ nhầm

---

# ResNet18 — F1 theo lớp

| Lớp | F1 | Lớp | F1 |
|---|---|---|---|
| **siren** | 0.928 | dog_bark | 0.721 |
| **engine_idling** | 0.866 | jackhammer | 0.658 |
| gun_shot | 0.850 | drilling | 0.602 |
| street_music | 0.837 | **air_conditioner** | **0.089** |
| car_horn | 0.836 | children_playing | 0.814 |

### Phát hiện đáng chú ý
- **air_conditioner SỤP ĐỔ** F1 = 0.089 (so với CNN 0.313)
- Mô hình thiên về đặc trưng mạnh → bỏ qua âm nền yếu
- → Lý do Macro-F1 thấp hơn CNN baseline

---

# ResNet18 + Augmentation — F1 theo lớp

| Lớp | F1 | Thay đổi vs ResNet18 |
|---|---|---|
| **car_horn** | **0.986** | +0.150 ⬆️ |
| gun_shot | 0.958 | +0.108 ⬆️ |
| siren | 0.863 | −0.065 ⬇️ |
| street_music | 0.857 | +0.020 ⬆️ |
| children_playing | 0.853 | +0.039 ⬆️ |
| dog_bark | 0.821 | +0.100 ⬆️ |
| engine_idling | 0.714 | −0.152 ⬇️ |
| jackhammer | 0.604 | −0.054 ⬇️ |
| drilling | 0.589 | −0.013 ⬇️ |
| **air_conditioner** | **0.372** | **+0.283** ⬆️⬆️ |

→ AC tăng **mạnh nhất** nhờ augmentation

---

# Phân tích confusion matrix

### Cặp nhầm lẫn chính (kỳ vọng và thực tế)

- **air_conditioner ↔ engine_idling**
  - Cả hai đều là âm nền liên tục, phổ năng lượng tương tự
- **drilling ↔ jackhammer**
  - Đều là âm cơ khí, có dải tần số mạnh tương tự
- Âm thanh môi trường liên tục **dễ bị nhầm sang lớp machinery**

### Augmentation giải quyết được gì?
✅ Tăng tính bền vững → giảm nhầm lẫn ở cặp dễ overfit
❌ Vẫn chưa giải quyết hoàn toàn confusion giữa machinery class

---

# So sánh trực tiếp

### Theo Accuracy
🥇 CNN baseline (0.7446) > ResNet18 = ResNet18+aug (0.7297)

### Theo Macro-F1
🥇 ResNet18 + aug (**0.7617**) > CNN baseline (0.7525) > ResNet18 (0.7201)

### Bài học
- **Không nên chỉ dùng 1 chỉ số** để kết luận mô hình tốt nhất
- Accuracy ưu tiên → CNN
- Cân bằng giữa các lớp → ResNet18 + aug
- Augmentation cải thiện **chất lượng thực tế** dù không tăng Accuracy

---

# Hạn chế của đề tài

- ⚠️ **Chỉ 1 fold split** — chưa áp dụng 10-fold CV chuẩn của UrbanSound8K
- ⚠️ Chưa khảo sát sâu **tuning** cho ResNet18
- ⚠️ Chưa thử nhiều phiên bản **augmentation** khác nhau
- ⚠️ Chưa thử các kỹ thuật: **Mixup, OneCycleLR, ensemble**
- ⚠️ Phần literature review **chưa đi sâu** từng paper cụ thể

→ Trong phạm vi BTL cá nhân và compute hạn chế, kết quả vẫn đủ rút ra insight có giá trị.

---

# Kết luận

### Đã làm
✅ Pipeline hoàn chỉnh: EDA → preprocess → train → evaluate
✅ So sánh công bằng 3 mô hình trên cùng setup
✅ Phân tích định lượng + định tính (per-class F1 + confusion matrix)
✅ W&B logging đầy đủ — tái lập được

### Insight chính
🔑 **Architecture lớn hơn ≠ luôn tốt hơn** trên dataset vừa
🔑 **Augmentation** quan trọng cho cân bằng lớp, không nhất thiết tăng Accuracy
🔑 **Machinery cluster** (AC, engine_idling, drilling, jackhammer) là bottleneck

---

# Tài liệu tham khảo

1. Salamon, J., Jacoby, C., Bello, J. P. (2014). *A dataset and taxonomy for urban sound research*. ACM MM.
2. Salamon, J., Bello, J. P. (2017). *Deep convolutional neural networks and data augmentation for environmental sound classification*. IEEE SPL.
3. He, K. et al. (2016). *Deep residual learning for image recognition*. CVPR.
4. Park, D. S. et al. (2019). *SpecAugment: a simple data augmentation method for ASR*. Interspeech.
5. Piczak, K. J. (2015). *Environmental sound classification with CNN*. IEEE MLSP.
6. Kong, Q. et al. (2020). *PANNs: large-scale pretrained audio neural networks*. IEEE TASLP.

---

<!-- _class: lead -->

# Cảm ơn!

## Q & A

**Code:** https://github.com/Zackerville/UrbanSoundClassification_DeepLearning_8K
**W&B:** https://wandb.ai/tien-ngozack2004-ho-chi-minh-city-university-of-technology/urban-sound-classification

Ngô Thái Minh Tiến — 2252809
games.tien.minh@gmail.com
