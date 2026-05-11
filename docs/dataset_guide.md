# Hướng Dẫn Thu Thập Dữ Liệu — FedSSL-MERC (Mục tiêu Q1)

## Chiến Lược Dữ Liệu Cho Paper Q1

> [!IMPORTANT]
> **Để đạt Q1**, paper cần report kết quả trên **ít nhất 2 benchmark chính** mà tất cả SOTA papers đều dùng. Chuẩn vàng là: **IEMOCAP + MELD**. Nếu có thêm **CMU-MOSEI** hoặc **M3ED** sẽ tạo lợi thế lớn khi review.

### Ưu tiên Dataset

| Ưu tiên | Dataset | Lý do | Trạng thái |
|:---:|:---|:---|:---:|
| 🔴 **#1** | **MELD** | Public, multimodal (T+A+V), multi-party, 13K utterances | ✅ Đã tải |
| 🔴 **#2** | **IEMOCAP** | Gold standard, dyadic, ~7.4K utterances, mọi Q1 paper đều dùng | ⏳ Cần xin |
| 🟡 **#3** | **CMU-MOSEI** | Lớn nhất (23K sentences), cross-corpus evaluation | 📥 Public |
| 🟢 **#4** | **DailyDialog** | Text-only, 102K utterances, test text-only baseline | 📥 Public |

---

## 1. MELD — ✅ ĐÃ TẢI XONG

| Thuộc tính | Chi tiết |
|:---|:---|
| **Tên đầy đủ** | Multimodal EmotionLines Dataset |
| **Nguồn** | TV series *Friends* |
| **Modalities** | Text + Audio + Video |
| **Kích thước** | Train: 9,989 utt / Dev: 1,109 utt / Test: 2,610 utt (tổng ~13.7K) |
| **Dialogues** | 1,433 dialogues |
| **Speakers** | Multi-party (260+ speakers) |
| **Emotions** | 7 classes: anger, disgust, fear, joy, **neutral**, sadness, surprise |
| **Sentiment** | 3 classes: positive, negative, neutral |
| **Đặc điểm** | Class imbalance nặng — **neutral chiếm ~47%** |

### Download

```
✅ CSV annotations: Đã tải tại data/raw/MELD/
   - train_sent_emo.csv (9,989 utterances)
   - dev_sent_emo.csv (1,109 utterances)
   - test_sent_emo.csv (2,610 utterances)

📥 Raw videos (~4GB): Học cần tải lên Google Drive
   wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
```

### Cấu trúc video files
```
MELD.Raw/
├── train/
│   ├── dia0_utt0.mp4    # Dialogue 0, Utterance 0
│   ├── dia0_utt1.mp4
│   └── ...
├── dev/
└── test/
```

### Emotion Distribution (Train)
```
neutral     4,710 (47.2%) ████████████████████████  ← majority class
joy         1,743 (17.4%) █████████
surprise    1,205 (12.1%) ██████
anger       1,109 (11.1%) █████
sadness       683 ( 6.8%) ███
disgust       271 ( 2.7%) █                         ← minority class
fear          268 ( 2.7%) █                         ← minority class
```

---

## 2. IEMOCAP — ⏳ CẦN XIN QUYỀN TRUY CẬP

> [!CAUTION]
> **IEMOCAP bắt buộc phải có** cho Q1 paper. Mọi paper ERC top-tier đều report trên IEMOCAP. **Xin access NGAY hôm nay** vì có thể mất 1-2 tuần để được duyệt.

| Thuộc tính | Chi tiết |
|:---|:---|
| **Tên đầy đủ** | Interactive Emotional Dyadic Motion Capture |
| **Tổ chức** | University of Southern California (USC) SAIL Lab |
| **Modalities** | Text + Audio + Video + **Motion Capture** (khuôn mặt) |
| **Kích thước** | ~12 giờ data, ~7,433 utterances (hoặc 5,531 sau filter) |
| **Dialogues** | 151 conversations (dyadic — 2 người) |
| **Speakers** | 10 diễn viên (5 nam, 5 nữ) |
| **Emotions** | Anger, Happiness, Excitement, Sadness, Frustration, Fear, Surprise, Neutral |
| **Evaluation** | Thường dùng **6-way** hoặc **4-way** (Happy+Excited merged) |

### Cách xin access

```
📌 Bước 1: Vào https://sail.usc.edu/iemocap/
📌 Bước 2: Tìm mục "Release" → Điền form yêu cầu truy cập
📌 Bước 3: Dùng email trường (FPT University) để tăng uy tín
📌 Bước 4: Ghi rõ mục đích "academic research on emotion recognition in conversations"
📌 Liên hệ: anfengxu@usc.edu (nếu chưa nhận được phản hồi sau 1 tuần)
```

### Setup sau khi nhận được
```
IEMOCAP/
├── Session1/
│   ├── dialog/
│   │   ├── EmoEvaluation/     # Emotion labels
│   │   └── transcriptions/    # Text transcripts
│   ├── sentences/
│   │   ├── wav/               # Audio files (.wav)
│   │   └── ...
│   └── ...
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

### Evaluation Protocol cho Q1
```
Cách 1 (phổ biến nhất): 6-class → angry, happy, excited, sad, frustrated, neutral
Cách 2 (4-class):        4-class → angry, happy+excited, sad, neutral
Metric: Weighted Average F1 (WF1)
Cross-validation: 5-fold (leave-one-session-out)
```

---

## 3. CMU-MOSEI — 📥 PUBLIC (Nên dùng cho cross-corpus)

| Thuộc tính | Chi tiết |
|:---|:---|
| **Tên đầy đủ** | CMU Multimodal Opinion Sentiment and Emotion Intensity |
| **Tổ chức** | Carnegie Mellon University MultiComp Lab |
| **Modalities** | Text + Audio + Video |
| **Kích thước** | **23,453 sentences** từ 5,000 videos (lớn nhất) |
| **Speakers** | 1,000+ YouTube speakers |
| **Emotions** | 6 classes: happy, sad, angry, fearful, disgusted, surprised |
| **Sentiment** | -3 → +3 scale (7 levels) |
| **Đặc điểm** | In-the-wild, real-world, đa dạng nhất |

### Download

```bash
# Cách 1: CMU Multimodal SDK (pre-extracted features — KHUYÊN DÙNG)
pip install mmsdk
# Hoặc clone:
git clone https://github.com/A2Zadeh/CMU-MultimodalSDK.git

# Cách 2: Raw data
# http://immortal.multicomp.cs.cmu.edu/
```

### Pre-extracted Features (đã align)
| Modality | Feature | Dim |
|:---|:---|:---:|
| Text | GloVe / BERT | 300 / 768 |
| Audio | COVAREP | 74 |
| Visual | OpenFace 2.0 (FAUs) | 35 |

> [!TIP]
> CMU-MOSEI có **pre-extracted features** sẵn → không cần tự extract, tiết kiệm rất nhiều thời gian trên T4.

---

## 4. CMU-MOSI — 📥 PUBLIC (Nhỏ, dùng test nhanh)

| Thuộc tính | Chi tiết |
|:---|:---|
| **Tên đầy đủ** | CMU Multimodal Opinion-level Sentiment Intensity |
| **Modalities** | Text + Audio + Video |
| **Kích thước** | 2,199 opinion segments |
| **Speakers** | 93 YouTube speakers |
| **Sentiment** | -3 → +3 scale |
| **Đặc điểm** | Nhỏ gọn, perfect cho debug và prototype nhanh |

### Download — cùng SDK như MOSEI
```bash
pip install mmsdk
```

---

## 5. DailyDialog — 📥 PUBLIC (Text-only baseline)

| Thuộc tính | Chi tiết |
|:---|:---|
| **Modalities** | **Text only** |
| **Kích thước** | 13,118 dialogues / **102,979 utterances** (lớn nhất text) |
| **Emotions** | 7: Ekman 6 + neutral |
| **Đặc điểm** | Open-domain, high quality, human-written |
| **Download** | `huggingface.co/datasets/daily_dialog` |

```python
# Tải bằng HuggingFace
from datasets import load_dataset
dataset = load_dataset("daily_dialog")
```

> [!TIP]
> DailyDialog rất hữu ích cho **text-only baseline** và test FL partitioning vì nó lớn nhất (102K utterances).

---

## 6. EmoryNLP — 📥 PUBLIC (Text-only, supplementary)

| Thuộc tính | Chi tiết |
|:---|:---|
| **Modalities** | Text only (Friends transcripts) |
| **Kích thước** | 897 dialogues / 12,606 utterances |
| **Emotions** | 7: neutral, joyful, peaceful, powerful, scared, mad, sad |
| **Download** | `github.com/emorynlp/emotion-detection` |

---

## 7. M3ED — 📥 PUBLIC (Chinese, bonus)

| Thuộc tính | Chi tiết |
|:---|:---|
| **Modalities** | Text + Audio + Video (Chinese language) |
| **Kích thước** | 990 dialogues / 24,449 utterances |
| **Emotions** | 7: happy, surprise, sad, disgust, anger, fear, neutral |
| **Download** | `github.com/AIM3-RUC/RUCM3ED` |

> [!NOTE]
> M3ED là tiếng Trung. Dùng nếu muốn show **cross-lingual** capability — ấn tượng cho reviewer Q1.

---

## So Sánh Tổng Hợp

| Dataset | Modalities | Utterances | Dialogues | Emotions | Access | Q1 Priority |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **IEMOCAP** | T+A+V+MC | ~7,433 | 151 | 6-8 | Xin USC | 🔴 Bắt buộc |
| **MELD** | T+A+V | 13,708 | 1,433 | 7 | Public | 🔴 Bắt buộc |
| **CMU-MOSEI** | T+A+V | 23,453 | — | 6 | Public | 🟡 Nên có |
| **DailyDialog** | T | 102,979 | 13,118 | 7 | Public | 🟢 Tùy chọn |
| **EmoryNLP** | T | 12,606 | 897 | 7 | Public | 🟢 Tùy chọn |
| **M3ED** | T+A+V | 24,449 | 990 | 7 | Public | 🟢 Bonus |

---

## SOTA Performance Benchmarks (Tham khảo Q1 papers)

### IEMOCAP (6-way, Weighted F1)
| Method | WF1 | Year |
|:---|:---:|:---:|
| DialogueRNN | 62.75% | 2019 |
| COSMIC | 65.28% | 2020 |
| MMGCN | 66.22% | 2021 |
| Hybrid Transformer+GNN | ~68-70% | 2024 |
| LLM-based (GPT/LLaMA adapter) | ~70-72% | 2025 |

### MELD (7-way, Weighted F1)
| Method | WF1 | Year |
|:---|:---:|:---:|
| DialogueRNN | 57.03% | 2019 |
| COSMIC | 65.21% | 2020 |
| EmoBERTa | 65.61% | 2021 |
| SOTA 2024-2025 | ~66-68% | 2024-25 |

> [!IMPORTANT]
> **Mục tiêu Q1:** Phương pháp FL+SSL của nhóm không nhất thiết phải beat SOTA centralized. Điểm mạnh là **beat SOTA trong setting federated** và chứng minh **privacy-utility tradeoff** tốt. Nghĩa là:
> - Centralized baseline: ~65% WF1
> - Federated (FedAvg) baseline: ~55-58% WF1 (performance drop)
> - **FedSSL-MERC (nhóm)**: ~60-63% WF1 (thu hẹp gap)
> → Contribution = thu hẹp gap giữa centralized và federated, đồng thời chỉ cần 10% labeled data

---

## Kế Hoạch Hành Động — Ai Làm Gì?

### Lộc (Team Lead) — NGAY HÔM NAY
1. **XIN ACCESS IEMOCAP** → Vào [sail.usc.edu/iemocap](https://sail.usc.edu/iemocap/), điền form, dùng email FPT
2. Cài CMU-MOSEI SDK: `pip install mmsdk`
3. Tải DailyDialog: `pip install datasets` → `load_dataset("daily_dialog")`

### Học (GPU A100) — Tuần này
1. Tải MELD raw videos lên Google Drive (~4GB):
   ```bash
   wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
   tar -xzf MELD.Raw.tar.gz
   ```
2. Tải CMU-MOSEI pre-extracted features qua SDK
3. Test extract features từ MELD videos (audio MFCC + visual ResNet)

### Phú (Research) — Tuần này
1. Tổng hợp bảng performance từ 5 papers gần nhất trên IEMOCAP + MELD
2. Tìm papers dùng FL/SSL cho NLP/multimodal tasks
3. Ghi chú chi tiết evaluation protocol mỗi dataset (splits, metrics, folds)
