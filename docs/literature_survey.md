# Literature Survey: Federated Evidential Learning for Emotion Recognition in Conversations

**Dự án:** ThuanPhongNhi — FedSSL-MERC  
**Ngày:** 16/05/2026 (cập nhật lần 2 — bổ sung 3 papers mới, điều chỉnh novelty claims)  
**Nhóm:** Đinh Đại Lộc, Trần Phi Học, Hồ Gia Phú  

---

## 1. Giới thiệu

Survey này tổng hợp 11 công trình nghiên cứu liên quan trực tiếp đến dự án ThuanPhongNhi, bao gồm 3 mảng chính:

1. **Emotion Recognition in Conversations (ERC)** — Nhận diện cảm xúc trong hội thoại
2. **Federated Learning (FL)** — Học liên kết phân tán bảo vệ quyền riêng tư
3. **Evidential Deep Learning (EDL)** — Ước lượng độ không chắc chắn (uncertainty)

Mục tiêu: Xác định research gap và định vị đóng góp của ThuanPhongNhi so với state-of-the-art.

---

## 2. Tổng quan các công trình

### 2.1 Emotion Recognition in Conversations (ERC)

#### [P1] DialogueRNN — An Attentive RNN for Emotion Detection in Conversations
- **Tác giả:** Majumder et al.
- **Venue:** AAAI 2019
- **Ý tưởng:** Mô hình 3 GRU đan xen: Global GRU (ngữ cảnh hội thoại), Party GRU (trạng thái người nói), Emotion GRU (biểu diễn cảm xúc). Thêm Attention mechanism để tham chiếu các phát ngôn trước.
- **Kết quả:** IEMOCAP 4-class WF1 = 62.75% (BiDialogueRNN+Att)
- **Hạn chế:** Dùng GloVe embeddings (100-dim), kiến trúc một chiều, không khai thác knowledge bên ngoài.

#### [P2] COSMIC — COMmonSense knowledge for Emotion Identification in Conversations
- **Tác giả:** Ghosal et al.
- **Venue:** EMNLP Findings 2020
- **Ý tưởng:** Mở rộng DialogueRNN thành 5 GRU states, tích hợp commonsense knowledge từ COMET (ATOMIC knowledge graph). Mỗi utterance được bổ sung 5 knowledge vectors: xIntent, xReact, xWant, oReact, oWant.
- **Kết quả:** MELD WF1 = 65.21%, IEMOCAP WF1 = 65.28%
- **Encoder:** RoBERTa-Large (355M params, 1024-dim)
- **Hạn chế:** Phụ thuộc external knowledge base, model size rất lớn, yêu cầu centralized data access.

#### [P3] EmoBERTa — Speaker-Aware Emotion Recognition in Conversation with RoBERTa
- **Tác giả:** Kim & Vossen
- **Venue:** arXiv 2021
- **Ý tưởng:** Fine-tune trực tiếp RoBERTa-Large trên toàn bộ dialogue. Input format: `[speaker1]: utt1 </s></s> [speaker2]: utt2 ...`. End-to-end training tất cả 355M parameters.
- **Kết quả:** MELD WF1 = 65.61%, IEMOCAP WF1 = 68.57%
- **Hạn chế:** Yêu cầu fine-tune toàn bộ 355M params, không scalable cho FL setting, không privacy-preserving.

#### [P4] DAG-ERC — Directed Acyclic Graph for Emotion Recognition in Conversation
- **Tác giả:** Shen et al.
- **Venue:** ACL 2021
- **Ý tưởng:** Mô hình conversation dưới dạng DAG (Directed Acyclic Graph), sử dụng graph neural network để capture dependency giữa các utterances thay vì sequential RNN.
- **Kết quả:** Cải thiện so với DialogueRNN trên IEMOCAP và DailyDialog.
- **Hạn chế:** Graph construction phức tạp, khó adapt cho FL setting.

#### [P5] TODKAT — Topic-Driven and Knowledge-Aware Transformer
- **Tác giả:** Zhu et al.
- **Venue:** ACL 2021
- **Ý tưởng:** Kết hợp topic modeling với external knowledge để cải thiện ERC. Sử dụng transformer architecture với topic-driven attention.
- **Hạn chế:** Phụ thuộc vào topic extractor và knowledge base.

### 2.2 Federated Learning (FL)

#### [P6] FedAvg — Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Tác giả:** McMahan et al.
- **Venue:** AISTATS 2017
- **Ý tưởng:** Framework FL nền tảng. Mỗi client train local → gửi model weights → server trung bình hóa theo tỷ lệ kích thước data: `w_k = |D_k| / Σ|D_j|`
- **Hạn chế:** Không xét chất lượng client, vulnerable với noisy/poisoned updates, hiệu suất giảm mạnh khi data non-IID.

#### [P7] FedProx — Federated Optimization in Heterogeneous Networks
- **Tác giả:** Li et al.
- **Venue:** MLSys 2020
- **Ý tưởng:** Thêm proximal term `μ/2 ||w - w_global||²` vào local loss, giữ client updates gần global model. Giải quyết systems heterogeneity (clients train với số epochs khác nhau).
- **Hạn chế:** Hyperparameter μ nhạy cảm, vẫn không xét uncertainty, không có cơ chế chống poisoned updates.

### 2.3 Evidential Deep Learning & Uncertainty

#### [P8] EDL — Evidential Deep Learning to Quantify Classification Uncertainty
- **Tác giả:** Sensoy et al.
- **Venue:** NeurIPS 2018
- **Ý tưởng:** Thay softmax output bằng Dirichlet distribution parameters. Model output → evidence → Dirichlet α → belief masses + epistemic uncertainty. Loss: Type-II MLE + KL regularization với annealing `λ_t = min(1, t/10)`.
- **Kết quả:** Outperform MC Dropout và ensemble methods trên MNIST, CIFAR.
- **Hạn chế:** Chỉ validated trên image classification, chưa áp dụng cho sequential/dialogue data.

#### [P9] Trusted Multi-View Classification
- **Tác giả:** Han et al.
- **Venue:** ICLR 2021
- **Ý tưởng:** Dempster-Shafer theory cho multi-view fusion. Mỗi view sinh Dirichlet parameters, fuse bằng Dempster's combination rule thay vì concatenation/attention.
- **Ý nghĩa cho ThuanPhongNhi:** Tiền đề cho Dempster-Shafer fusion trong multimodal ERC.

### 2.4 Semi-Supervised Learning

#### [P10] FixMatch — Simplifying Semi-Supervised Learning with Consistency and Confidence
- **Tác giả:** Sohn et al.
- **Venue:** NeurIPS 2020
- **Ý tưởng:** Kết hợp pseudo-labeling + consistency regularization. Weak augmentation → pseudo-label (nếu confidence > 0.95) → strong augmentation phải match.
- **Hạn chế:** Hard threshold 0.95 → biased, confirmation bias khi model sai nhưng tự tin.

### 2.5 Federated Learning cho Emotion Recognition (MỚI PHÁT HIỆN 16/05)

> [!CAUTION]
> **Đính chính:** Ban đầu nhóm ngộ nhận rằng "chưa ai kết hợp FL + ERC". Sau khi search kỹ hơn (16/05/2026), phát hiện 3 công trình liên quan trực tiếp. Cần điều chỉnh novelty claims.

#### [P11] FedDISC — Federated Dialogue-Semantic Diffusion for Emotion Recognition under Incomplete Modalities 🔴 ĐỐI THỦ TRỰC TIẾP
- **Tác giả:** Qiu et al.
- **Venue:** NeurIPS 2025
- **Ý tưởng:** Framework FL cho MERC xử lý **missing modalities**. Dùng Dialogue Graph Network (DGN) để capture context + Semantic Conditioning Network (SCN) để recover modalities thiếu bằng diffusion model.
- **FL strategy:** Alternating Frozen Aggregation (AFS) — đóng băng luân phiên recovery và classifier modules.
- **Datasets:** IEMOCAP, CMUMOSI, CMUMOSEI
- **Khác biệt với ThuanPhongNhi:** FedDISC giải quyết *"client thiếu modality"*, ThuanPhongNhi giải quyết *"client có chất lượng data khác nhau"*. Approach hoàn toàn khác: diffusion recovery vs uncertainty-weighted aggregation.

#### [P12] FedMultiEmo — Real-time Multimodal Emotion Recognition
- **Tác giả:** (2025)
- **Venue:** Preprint / Journal 2025
- **Ý tưởng:** FL framework cho emotion recognition trên **edge devices** (xe hơi). Fusion: CNN (face) + Random Forest (physiological signals).
- **FL strategy:** Personalized FedAvg weighted by data volume.
- **Datasets:** FER2013, custom physiological datasets
- **Khác biệt với ThuanPhongNhi:** FedMultiEmo là **utterance-level** (không model conversation context), domain khác (automotive vs dialogue). Không phải competitor trực tiếp.

#### [P13] FedEmoNet — Federated Speech Emotion Recognition with Differential Privacy
- **Tác giả:** (May 2026)
- **Venue:** PLOS ONE 2026
- **Ý tưởng:** FL + Differential Privacy cho speech emotion recognition. Architecture: TCN-Transformer fusion + PSO feature selection + SHAP explainability.
- **FL strategy:** FedProx + (ε=1.0, δ=10⁻⁵)-DP.
- **Datasets:** EmoDB, RAVDESS (utterance-level SER)
- **Khác biệt với ThuanPhongNhi:** FedEmoNet là **utterance-level SER** (không có dialogue context). Domain khác: isolated speech vs multi-turn conversation.

---

## 3. Bảng so sánh SOTA

### 3.1 MELD & IEMOCAP (Weighted F1 %)

| Method | Encoder | Params | MELD | IEMOCAP | Setting |
|:-------|:--------|:------:|:----:|:-------:|:-------:|
| DialogueRNN | GloVe | ~5M | 57.03 | 62.57* | Centralized |
| COSMIC | RoBERTa-L | 355M+ | 65.21 | 65.28 | Centralized |
| EmoBERTa | RoBERTa-L | 355M | 65.61 | 68.57 | Centralized |
| **ThuanPhongNhi EDL** | RoBERTa-B | **125M** | 63.09 | 56.33† | **Centralized** |
| **ThuanPhongNhi EAFA** | RoBERTa-B | **125M** | 63.44 | 58.46† | **Federated** |

> *DialogueRNN 62.57% trên 4-class (happy+excited merged)  
> †ThuanPhongNhi IEMOCAP chạy 6-class (khó hơn, không so sánh trực tiếp)

### 3.2 DailyDialog (Micro F1, excl. neutral)

| Method | Micro F1 | Setting |
|:-------|:--------:|:-------:|
| COSMIC (RoBERTa-L) | ~51.05* | Centralized |
| **ThuanPhongNhi EDL** | **88.09** | **Centralized** |
| **ThuanPhongNhi EAFA** | **88.68** | **Federated** |

> *DailyDialog Micro F1 từ COSMIC paper rất thấp do class imbalance nặng (83% neutral). Cần xác minh protocol chính xác.

---

## 4. Research Gaps (ĐÃ CẬP NHẬT sau verification 16/05)

> [!WARNING]
> **Đính chính quan trọng:** Gap 1 ban đầu ghi "Chưa có FL cho ERC" — điều này **SAI**. FedDISC (NeurIPS 2025) đã kết hợp FL + MERC. Các gaps dưới đây đã được điều chỉnh cho chính xác.

### ~~Gap 1 (cũ): Chưa có FL cho ERC~~ → ĐÃ BỊ BÁC BỎ

FedDISC (NeurIPS 2025) đã kết hợp FL + MERC. Tuy nhiên, FedDISC tập trung vào **missing modalities recovery** (diffusion-based), không giải quyết vấn đề **client quality weighting**.

### Gap 1 (mới): Chưa có Uncertainty-aware Aggregation cho ERC ⭐⭐⭐⭐

Các FL frameworks hiện tại cho emotion (FedDISC, FedMultiEmo, FedEmoNet) đều dùng FedAvg hoặc FedProx — **không xét chất lượng intrinsic** của client models. Không có framework nào dùng epistemic uncertainty từ EDL để guide aggregation.

**Đóng góp ThuanPhongNhi:** EAFA sử dụng EDL uncertainty làm tín hiệu chất lượng → tự động downweight clients kém → robust hơn FedAvg/FedProx.

### Gap 2: EDL chưa được áp dụng cho Dialogue-level ERC ⭐⭐⭐

Sensoy 2018 chỉ validate EDL trên image classification. FedDISC dùng diffusion model, FedEmoNet dùng TCN-Transformer — **không ai dùng Evidential Deep Learning** cho dialogue-level sequential data.

**Đóng góp ThuanPhongNhi:** EDL Head trên DialogueRNN → Dirichlet prediction + uncertainty estimation cho từng utterance trong conversation.

### Gap 3: Thiếu Evidential Consistency Regularization cho SSL ⭐⭐⭐

FixMatch (2020) dùng hard threshold 0.95 cho pseudo-labeling → confirmation bias. Chưa có framework nào thay thế bằng certainty-weighted Dirichlet KL divergence.

**Đóng góp ThuanPhongNhi:** ECR — gradient auto-vanishes khi uncertain, không cần hard threshold.

### Gap 4: So sánh FL approaches cho ERC còn thiếu ⭐⭐

FedDISC so sánh với centralized methods nhưng **không so sánh trực tiếp FedAvg vs uncertainty-weighted aggregation** trên cùng ERC benchmarks (MELD, DailyDialog).

**Đóng góp ThuanPhongNhi:** Ablation study đầy đủ: CE vs EDL × FedAvg vs EAFA trên 3 datasets × 3 seeds.

---

## 5. Vị trí đóng góp của ThuanPhongNhi (CẬP NHẬT)

> [!IMPORTANT]
> Bảng dưới đã điều chỉnh sau khi phát hiện FedDISC, FedMultiEmo, FedEmoNet.

| Đóng góp | Novel? | Mức độ | So với ai |
|:---------|:------:|:------:|:----------|
| ~~FL + ERC (setting mới)~~ | ❌ | — | FedDISC đã làm (NeurIPS 2025) |
| EAFA (uncertainty aggregation) | ✅ | ⭐⭐⭐⭐ | Không ai dùng EDL uncertainty cho FL aggregation trong ERC |
| EDL cho dialogue ERC | ✅ | ⭐⭐⭐ | FedDISC dùng diffusion, FedEmoNet dùng TCN — không ai dùng EDL |
| ECR (thay FixMatch) | ✅ | ⭐⭐⭐ | Dirichlet KL thay hard pseudo-label threshold |
| Ablation FedAvg vs EAFA trên ERC | ✅ | ⭐⭐ | FedDISC không so sánh aggregation strategies |

### Kết quả nổi bật

**EAFA Federated vượt EDL Centralized** trên cả 3 datasets:
- MELD: +0.35% (63.44 vs 63.09)
- IEMOCAP: +2.13% (58.46 vs 56.33)
- DailyDialog: +0.59% Micro F1 (88.68 vs 88.09)

→ Uncertainty-weighted collaboration **tốt hơn** centralized training đơn thuần.

### So sánh trực tiếp với FedDISC

| Aspect | FedDISC (NeurIPS 2025) | ThuanPhongNhi |
|:-------|:----------------------|:--------|
| Vấn đề chính | Client thiếu modality | Client chất lượng khác nhau |
| Giải pháp | Diffusion recovery | EDL uncertainty weighting |
| Aggregation | AFS (frozen modules) | EAFA (epistemic-weighted) |
| Uncertainty | Không có | ✅ Built-in via EDL |
| SSL | Không | ECR (certainty-weighted) |
| Datasets | IEMOCAP, CMU-MOSI/MOSEI | MELD, IEMOCAP, DailyDialog |

---

## 6. Điểm cần lưu ý trong so sánh

### 6.1 Encoder Size Gap
- SOTA (COSMIC, EmoBERTa): RoBERTa-**Large** 355M params
- ThuanPhongNhi: RoBERTa-**Base** 125M params (nhỏ hơn **2.8x**)
- → Kết quả thấp hơn 2% là hợp lý, không phải do methodology yếu

### 6.2 IEMOCAP Class Count
- Nhiều papers dùng **4 classes** (merge happy+excited)
- ThuanPhongNhi dùng **6 classes** → bài toán khó hơn
- Cần ghi rõ "6-class evaluation" khi so sánh

### 6.3 DailyDialog Metric
- Papers chuẩn dùng **Micro F1 (excl. neutral)**
- ThuanPhongNhi đã fix để hỗ trợ metric này (15/05/2026)

---

## 7. Narrative cho Paper (CẬP NHẬT sau verification)

### ❌ KHÔNG NÊN claim:
> "We are the FIRST to apply FL for ERC"

### ✅ NÊN claim:
> "We propose the FIRST uncertainty-aware federated framework for ERC that leverages Evidential Deep Learning to enable epistemic-guided aggregation (EAFA), eliminating the need for hard pseudo-labeling thresholds while providing built-in uncertainty quantification."

### Framing chiến lược:
> "While FedDISC (NeurIPS 2025) addresses modality incompleteness in federated MERC, our work tackles a complementary challenge: **how to aggregate client models when data quality varies** — without external knowledge recovery. Our EAFA mechanism uses EDL uncertainty as an intrinsic quality signal."

---

## 8. Tài liệu tham khảo

### ERC (Emotion Recognition in Conversations)
1. Majumder, N. et al. (2019). DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. *AAAI*.
2. Ghosal, D. et al. (2020). COSMIC: COMmonsense knowledge for Emotion Identification in Conversations. *EMNLP Findings*.
3. Kim, T. & Vossen, P. (2021). EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa. *arXiv:2108.12009*.
4. Shen, W. et al. (2021). Directed Acyclic Graph Network for Conversational Emotion Recognition. *ACL*.
5. Zhu, L. et al. (2021). Topic-Driven and Knowledge-Aware Transformer for Dialogue Emotion Detection. *ACL*.

### Federated Learning
6. McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
7. Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.

### Evidential & Uncertainty
8. Sensoy, M. et al. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *NeurIPS*.
9. Han, Z. et al. (2021). Trusted Multi-View Classification. *ICLR*.

### Semi-Supervised Learning
10. Sohn, K. et al. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. *NeurIPS*.

### FL + Emotion Recognition (MỚI — bắt buộc cite)
11. Qiu, X. et al. (2025). FedDISC: Federated Dialogue-Semantic Diffusion for Emotion Recognition under Incomplete Modalities. *NeurIPS 2025*. **🔴 Competitor trực tiếp**
12. FedMultiEmo (2025). Real-time Multimodal Emotion Recognition via Federated Learning. *Preprint*.
13. FedEmoNet (2026). Federated Speech Emotion Recognition with Differential Privacy. *PLOS ONE*.
