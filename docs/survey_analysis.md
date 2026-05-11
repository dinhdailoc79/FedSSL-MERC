# Survey Analysis: Multi-modal Conversational Emotion Recognition

> **Paper:** "A Comprehensive Survey on Multi-modal Conversational Emotion Recognition with Deep Learning"  
> **Authors:** Yuntao Shou, Tao Meng, Wei Ai, Fangze Fu, Nan Yin, Keqin Li  
> **Published:** ACM Transactions on Information Systems (TOIS), Vol. 44, No. 2, 2026  
> **DOI:** [10.1145/3786343](https://dl.acm.org/doi/10.1145/3786343) | **ArXiv:** [2312.05735](https://arxiv.org/abs/2312.05735)

---

## 1. Tóm Tắt Tư Tưởng Chính

### 1.1 Định nghĩa MCER
Multi-modal Conversational Emotion Recognition (MCER) nhằm nhận diện và theo dõi trạng thái cảm xúc của người nói trong cuộc hội thoại, sử dụng đồng thời 3 modalities: **Text**, **Audio/Speech**, và **Visual/Video**.

### 1.2 Tại sao MCER khó hơn MER truyền thống?
- MER truyền thống: Phân tích 1 utterance → 1 emotion label
- MCER: Phải xử lý **multi-turn dialogue** + **multi-speaker** + **multi-modal** → chuỗi emotion labels
- 3 thách thức cốt lõi:
  1. **Emotional Interaction Relationships** — Cảm xúc phụ thuộc vào ngữ cảnh hội thoại
  2. **Multi-modal Consistency & Complementarity** — Modalities có thể bổ sung hoặc mâu thuẫn
  3. **Speaker Dynamics** — Mỗi speaker có baseline cảm xúc khác nhau

---

## 2. Taxonomy — 4 Nhóm Phương Pháp

### 2.1 Context-Free Modeling
- **Mô tả:** Phân tích từng utterance độc lập, không xét ngữ cảnh
- **Ưu điểm:**
  - ✅ Computational cost thấp
  - ✅ Dễ triển khai, train nhanh
  - ✅ Phù hợp real-time đơn giản
- **Nhược điểm:**
  - ❌ Bỏ qua ngữ cảnh → mất nuance
  - ❌ Accuracy thấp cho hội thoại phức tạp
  - ❌ Không xử lý được sarcasm, emotional shift

### 2.2 Sequential Context Modeling
- **Mô tả:** Mô hình hóa chuỗi hội thoại theo thời gian (RNN, LSTM, Transformer)
- **Phương pháp tiêu biểu:** DialogueRNN, HiTrans, BERT-based
- **Ưu điểm:**
  - ✅ Nắm bắt temporal dynamics
  - ✅ Hiểu emotional transitions
  - ✅ Cải thiện đáng kể so với context-free
- **Nhược điểm:**
  - ❌ Tốn bộ nhớ với hội thoại dài
  - ❌ Vanishing gradient (RNN/LSTM)
  - ❌ Context window giới hạn (Transformer)

### 2.3 Speaker-Differentiated Modeling
- **Mô tả:** Tạo embedding/profile riêng cho từng speaker
- **Phương pháp tiêu biểu:** COSMIC, EmoBERTa
- **Ưu điểm:**
  - ✅ Personalized emotional profiles
  - ✅ Giảm ambiguity multi-party
  - ✅ Xử lý tốt intra-speaker consistency
- **Nhược điểm:**
  - ❌ Cần nhiều data cho mỗi speaker
  - ❌ Tăng model complexity
  - ❌ Khó generalize sang unseen speakers

### 2.4 Speaker-Relationship Modeling
- **Mô tả:** Dùng GNN mô hình hóa quan hệ giữa speakers
- **Phương pháp tiêu biểu:** DialogueGCN, MMGCN, DAG-ERC
- **Ưu điểm:**
  - ✅ Nắm bắt emotional contagion
  - ✅ Hiểu group dynamics
  - ✅ Holistic understanding
- **Nhược điểm:**
  - ❌ Model phức tạp, khó optimize
  - ❌ Nhạy cảm với speaker diarization errors
  - ❌ Đòi hỏi graph structure rõ ràng

---

## 3. Datasets

| Dataset | Modalities | Dialogues | Utterances | Labels | Source |
|---------|:----------:|:---------:|:----------:|--------|--------|
| **IEMOCAP** | T+A+V+MoCap | 151 | ~7,433 | 6 emotions | USC Acted |
| **MELD** | T+A+V | 1,433 | ~13,708 | 7 emotions + sentiment | Friends TV |
| **EmoryNLP** | Text | 897 | ~12,606 | 7 emotions | Friends |
| **DailyDialog** | Text | 13,118 | ~102,979 | 7 emotions | Human-written |

### Feature Extraction
| Modality | Pre-trained Models |
|----------|-------------------|
| Text | BERT, RoBERTa, GPT, LLaMA |
| Audio | OpenSMILE, wav2vec 2.0, HuBERT |
| Visual | OpenFace, ResNet, 3D-CNN |

---

## 4. Xu Hướng Mới (2024-2025)

1. **LLM/MLLM Integration** — Adapter modules cho emotion understanding
2. **Hybrid Transformer + GNN** — Kết hợp global + local reasoning
3. **Knowledge Distillation** — Teacher-student paradigm
4. **Self-Supervised Pre-training** — Leverage unlabeled data
5. **Cross-modal Attention** — Dynamic fusion

---

## 5. Research Gap → Đề Tài Nhóm

Survey **KHÔNG** đề cập:
- ❌ Federated Learning (privacy-preserving)
- ❌ Semi-Supervised Learning (label scarcity)
- ❌ Non-IID data handling

→ **Đề tài "Federated Semi-Supervised MERC"** lấp đầy gap này.

---

## 6. TODO cho Phú (Survey Phase)

- [ ] Tìm thêm 5-10 papers liên quan đến FL + SSL + MERC
- [ ] So sánh performance các methods trên IEMOCAP và MELD
- [ ] Tổng hợp bảng so sánh chi tiết (Table of Methods)
- [ ] Tìm existing implementations trên GitHub
- [ ] Đọc thêm về FL for NLP/multimodal tasks
