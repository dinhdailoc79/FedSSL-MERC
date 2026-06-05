# Kế hoạch Triển khai: Weak Accept (6) → Accept (7-8)

## Phân tích tình hình

Người phản biện hiện tại đánh giá bài báo ở mức **Weak Accept** và đã liệt kê rõ ràng các điểm yếu còn lại. Để đạt điểm **Accept (7-8)**, chúng ta phải giải quyết triệt độ từng điểm yếu này. Dưới đây là danh sách các hành động được sắp xếp theo thứ tự ưu tiên.

---

## TIER 1 — TÁC ĐỘNG CAO, CODE + THỰC NGHIỆM (Hạn chót: ~10 tháng 7)

### T1.1 Kiểm định Ý nghĩa Thống kê ⭐⭐⭐
**Nhận xét của Reviewer:** *"statistical significance is not consistently established"* (ý nghĩa thống kê chưa được thiết lập một cách nhất quán)

> [!IMPORTANT]
> Đây được coi là rào cản lớn nhất để bài báo được nhận (Accept). Nếu không có p-value, người phản biện không thể phân biệt được hiệu năng cải tiến thực sự hay chỉ là do nhiễu ngẫu nhiên giữa các seed.

**Kế hoạch:** Chạy kiểm định Wilcoxon signed-rank test cặp trên tất cả các so sánh chính (EAFA so với FedAvg, ECR so với Supervised, v.v.) qua các seed khác nhau. Báo cáo giá trị p-value trong bảng hoặc dưới dạng chú thích.

**Các file liên quan:** Các script có sẵn như `scripts/run_significance.py` và `scripts/run_statistical_analysis.py` có thể đã chứa logic này.

---

### T1.2 So sánh DS Fusion với Logit Averaging và Learnable Gating ⭐⭐⭐
**Nhận xét của Reviewer:** *"DS fusion improvements appear small; a direct comparison to logit averaging and/or a learnable gating baseline would clarify DS fusion's added value"* (cải tiến của DS fusion khá nhỏ; việc so sánh trực tiếp với logit averaging hoặc baseline gating có thể học được sẽ làm rõ giá trị gia tăng của DS fusion)

**Kế hoạch:** Triển khai thêm hai phương pháp fusion baseline:
1. **Logit Averaging** — Tính trung bình cộng của softmax logits từ hai nhánh văn bản và âm thanh.
2. **Learnable Gating** — Một mạng MLP 2 lớp siêu nhẹ để học trọng số động của từng modality.

Chạy thực nghiệm trên MELD multimodal với 3 seeds. So sánh trực tiếp với phương pháp DS evidence summation hiện tại.

**Các file liên quan:**
- [NEW] `models/evidential/fusion_baselines.py`
- [MODIFY] `scripts/run_multimodal_experiments.py` hoặc tạo script mới.

---

### T1.3 Kháng Tấn công Byzantine/Adversarial ⭐⭐
**Nhận xét của Reviewer:** *"robustness to malicious or strategic clients is not evaluated"* + *"Could adversarial clients game the aggregation by under-reporting uncertainty?"* (khả năng chống chịu đối với client độc hại hoặc client chiến lược chưa được đánh giá; liệu các client độc hại có thể thao túng việc tổng hợp bằng cách báo cáo thiếu độ không đảm bảo?)

**Kế hoạch:** Giả lập 1-2 client Byzantine (trong tổng số 5 client):
- (a) **Spoofing độ không đảm bảo (Uncertainty spoofing):** Client liên tục gửi ū_k = 0.0 (giả vờ cực kỳ tự tin) trong khi gửi các model weights ngẫu nhiên/nhiễu.
- (b) **Độc hại hóa mô hình (Model poisoning):** Gửi các gradient ngẫu nhiên hoặc bị đảo ngược.

So sánh mức độ sụt giảm hiệu năng của EAFA so với FedAvg dưới các cuộc tấn công này. EAFA được kỳ vọng sẽ tự động giảm thiểu ảnh hưởng vì các client bị độc hại hóa khi chạy trên dữ liệu kiểm thử của server (hoặc thông qua cơ chế giám sát cục bộ) sẽ tự sinh ra mức độ không đảm bảo (epistemic uncertainty) rất cao.

**Các file liên quan:**
- [NEW] `scripts/run_byzantine_robustness.py`

---

### T1.4 Báo cáo Per-Class F1 và Confusion Matrix ⭐⭐
**Nhận xét của Reviewer:** *"Can you report per-class F1 and confusion matrices consistently?"* (Bạn có thể báo cáo điểm F1 theo từng lớp và confusion matrix một cách nhất quán không?)

**Kế hoạch:** Trích xuất kết quả F1 chi tiết cho từng class từ các checkpoint tốt nhất hiện tại trên MELD và IEMOCAP-6. Vẽ heatmap confusion matrix. Chứng minh rằng EAFA cải thiện đáng kể trên các class thiểu số (minority classes).

**Các file liên quan:** Script `scripts/extract_fairness_metrics.py` có thể đã có sẵn một phần logic này.

---

### T1.5 Tăng cường Dữ liệu Văn bản (Stronger Text Augmentation) ⭐⭐
**Nhận xét của Reviewer:** *"text augmentation relies on Gaussian noise and dropout, which are relatively weak for NLP; stronger text augmentations not explored"* (phương pháp tăng cường văn bản dựa trên Gaussian noise và dropout là khá yếu đối với NLP; các phương pháp tăng cường mạnh hơn chưa được khám phá)

**Kế hoạch:** Triển khai và kiểm thử ít nhất một phương pháp tăng cường mức feature dành riêng cho NLP:
- **Token-level dropout** (triệt tiêu ngẫu nhiên 15% các token embeddings).
- **Feature cutout** (triệt tiêu một vùng liên tục 20% các chiều của feature).

Chạy thực nghiệm ECR với kỹ thuật tăng cường mới này trên MELD và IEMOCAP. Nếu hiệu năng cải thiện, chúng ta sẽ cập nhật vào bài báo. Nếu không cải thiện, đây cũng sẽ là một kết quả âm tính (negative result) có giá trị để báo cáo.

> [!NOTE]
> Do chúng ta sử dụng frozen RoBERTa embeddings đã được trích xuất trước (không phải văn bản thô), các phương pháp như back-translation hoặc synonym replacement là bất khả thi. Can thiệp mức feature là lựa chọn thực tế nhất.

**Các file liên quan:**
- [MODIFY] `semi_supervised/augmentation.py`
- [NEW] `scripts/run_augmentation_ablation.py`

---

### T1.6 FixMatch Tương thích với Dirichlet Expected Probabilities ⭐⭐
**Nhận xét của Reviewer:** *"FixMatch/FlexMatch comparisons may be disadvantaged by not adapting them to evidential outputs"* (việc so sánh với FixMatch/FlexMatch có thể không công bằng do chưa điều chỉnh chúng tương thích với đầu ra evidential)

**Kế hoạch:** Xây dựng phiên bản `DirichletFixMatch`:
1. Sử dụng `EvidentialDialogueRNN` (thay vì DialogueRNN thông thường).
2. Tính toán pseudo-label dựa trên xác suất kỳ vọng Dirichlet: $\hat{p}_c = \alpha_c / S$.
3. Sử dụng ngưỡng lọc dựa trên độ không đảm bảo (uncertainty-aware threshold): chỉ chấp nhận pseudo-label nếu $u < \tau$.

Điều này đảm bảo phép so sánh giữa ECR và FixMatch là hoàn toàn sòng phẳng và tối ưu hóa cho cả hai.

**Các file liên quan:**
- [NEW] `semi_supervised/dirichlet_fixmatch.py`
- [NEW] `scripts/run_dirichlet_fixmatch.py`

---

## TIER 2 — TÁC ĐỘNG CAO, CHỈ SỬA NỘI DUNG (Có thể làm song song với Tier 1)

### T2.1 Làm rõ cách tính ū_k ⭐⭐⭐
**Nhận xét của Reviewer:** *"How exactly is ūk computed... Have you analyzed whether clients with higher unlabeled ratios are systematically down-weighted?"* (ū_k được tính chính xác như thế nào... Bạn đã phân tích liệu các client có tỷ lệ dữ liệu không nhãn cao hơn có hệ thống bị giảm trọng số không?)

**Kế hoạch:** Thêm câu làm rõ rõ ràng trong phần §3.2 (EAFA) rằng $\bar{u}_k$ chỉ được tính toán trên **dữ liệu có nhãn (labeled data)** của mỗi client (để tránh thiên vị hoặc nhiễu từ dữ liệu không nhãn). Thêm 1-2 câu thảo luận về ảnh hưởng của tỷ lệ dữ liệu nhãn.

---

### T2.2 Tiết chế phát biểu về Centralized SOTA ⭐⭐
**Nhận xét của Reviewer:** *"Centralized 'SOTA' comparisons are not apples-to-apples... claims of competitiveness should be tempered"* (So sánh với Centralized SOTA không phải là so sánh tương đồng... các tuyên bố về tính cạnh tranh cần được giảm nhẹ)

**Kế hoạch:** Bổ sung câu hạn chế trong đoạn kết quả chính, thừa nhận sự khác biệt về feature pipeline và định vị lại kết quả của chúng ta là "cạnh tranh trong cùng một thiết lập feature" thay vì so sánh tuyệt đối với các SOTA phức tạp khác.

---

### T2.3 Lý giải tại sao EAFA > Centralized: Diagnostics ⭐⭐
**Nhận xét của Reviewer:** *"Why might EAFA surpass centralized training? Can you provide diagnostics?"* (Tại sao EAFA lại vượt trội hơn huấn luyện tập trung? Bạn có thể cung cấp phân tích chẩn đoán không?)

**Kế hoạch:** Thêm một đoạn thảo luận giải thích hiệu ứng điều hòa ngầm (implicit regularization) của huấn luyện federated: (i) việc trung bình hóa giữa nhiều client hoạt động như một cơ chế ensemble mô hình, (ii) trọng số uncertainty của EAFA hoạt động như bộ lọc ensemble chất lượng cao, (iii) báo cáo chỉ số đa dạng mô hình (cosine distance giữa các client state dicts).

---

### T2.4 Thảo luận về tích hợp LLM-based ERC ⭐
**Nhận xét của Reviewer:** *"ERC baselines are mostly pre-LLM architectures... how the method might integrate with recent instruction-tuned or LLM-based ERC is limited"* (Các baseline ERC chủ yếu là kiến trúc trước thời LLM... việc tích hợp phương pháp này với LLM-based ERC còn hạn chế)

**Kế hoạch:** Thêm 2-3 câu trong phần Related Work hoặc Discussion chỉ ra rằng EAFA/ECR là các cơ chế độc lập với backbone và có thể dễ dàng áp dụng lên các bộ mã hóa LLM-based ERC (ví dụ: thông qua huấn luyện tham số hiệu quả LoRA trong môi trường federated).

---

### T2.5 Cam kết Công bố Mã nguồn (Code Release) ⭐
**Nhận xét của Reviewer:** *"Could you release code?"* (Bạn có thể công bố mã nguồn không?)

**Kế hoạch:** Thêm câu cam kết trong bài báo: "Mã nguồn và các file cấu hình sẽ được công bố công khai sau khi bài báo được chấp nhận."

---

## TIER 3 — TIỂU TIẾT / NẾU CÒN THỜI GIAN

### T3.1 Temperature Scaling cho FixMatch
Reviewer đề cập đến việc căn chỉnh nhiệt độ / hiệu chuẩn xác suất cho FixMatch. Vì chúng ta đã chứng minh persistent FlexMatch không cải thiện nhiều, mục này có ưu tiên thấp hơn.

### T3.2 Phân tích lỗi mất modality trong DS Fusion
Đã được bao quát một phần trong bài báo; có thể bổ sung 1-2 câu nếu còn trống dòng trống trang.

---

## Các Câu hỏi Mở cần Anh Xác nhận

> [!IMPORTANT]
> 1. **Byzantine Robustness (T1.3):** Anh muốn chạy giả lập code đầy đủ hay chỉ cần viết 1 đoạn lập luận thảo luận lý thuyết trong bài báo? Chạy giả lập thực tế sẽ mất khoảng vài ngày huấn luyện.
> 2. **NLP Augmentation (T1.5):** Vì chúng ta dùng embedding RoBERTa dạng tĩnh (frozen), chúng ta chỉ có thể can thiệp bằng feature-level perturbation (Token-level dropout/cutout). Anh có đồng ý với hướng tiếp cận này không?
> 3. **DS Fusion Baselines (T1.2):** Dữ liệu đa phương thức (text+audio) đã được trích xuất và lưu cache đầy đủ trong máy của anh chưa?
> 4. **Kiểm soát Phạm vi:** Với hạn chót là 17/7 (khoảng 6 tuần), chúng ta nên ưu tiên giải quyết thật kỹ các mục quan trọng nhất (T1.1 kiểm định ý nghĩa thống kê, T1.2 baselines fusion, nhóm sửa LaTeX T2.1-T2.5) trước, rồi mới làm các giả lập phụ (Byzantine, Augmentation, Dirichlet FixMatch) nếu thời gian cho phép chứ?

## Ước lượng Tiến độ

| Tuần | Công việc |
|------|-----------|
| 4/6 - 10/6 | T1.1 (thống kê ý nghĩa), T1.4 (per-class F1), T2.1-T2.5 (sửa LaTeX) |
| 11/6 - 17/6 | T1.2 (baseline DS fusion), T1.6 (Dirichlet FixMatch) |
| 18/6 - 24/6 | T1.3 (Byzantine), T1.5 (NLP augmentation) |
| 25/6 - 1/7 | Tích hợp toàn bộ kết quả, kiểm tra biên dịch LaTeX |
| 2/7 - 10/7 | Tối ưu hóa văn bản, xem xét thủ công, định dạng PDF |
| 11/7 - 17/7 | Hạn chót duyệt nội dung nội bộ nhóm |
