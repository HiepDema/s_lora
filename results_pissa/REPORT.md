# S-LoRA vs LoRA trên Mistral-7B, cấu hình PiSSA

**Kết luận: ở ngân sách 167M trên bài toán suy luận toán, S-LoRA thua LoRA rõ rệt
— 10,08 điểm GSM8K và 8,82 điểm MATH. Đây là kết quả thật về phương pháp, không
phải lỗi pipeline: baseline LoRA chạy trên cùng đường ống này vượt cả hai con số
LoRA công bố trong PiSSA và PMSS.**

---

## 1. Thiết lập

Mistral-7B-v0.1, MetaMathQA → GSM8K + MATH, theo đúng PiSSA (Meng et al.,
NeurIPS 2024) — nguồn mà PMSS phụ lục A.3 dẫn làm chuẩn:

AdamW · batch 128 (8 × accum 16) · lr 2e-5 · cosine · warmup 0,03 · weight decay 0
· alpha = r · dropout 0 · cả 7 lớp tuyến tính · float32 · **1 epoch** · seed 0 ·
max_len 512 · max_new_tokens 512 · loss chỉ trên phần response · greedy decoding.

Hai cấu hình ở **cùng ngân sách tham số**:

| | phương pháp | rank | tham số |
|---|---|---|---|
| Run 1 | S-LoRA hybrid | 116 | 167.247.872 |
| Run 2 | LoRA | 64 | 167.772.160 |

r=64 là đúng baseline LoRA của PiSSA/PMSS (2.621.440 × 64 = 167,77M). Ở ngân
sách đó S-LoRA mua được rank 116 — gấp **1,81×** — vì 5 trong 7 ma trận của
Mistral-7B không vuông nên S-LoRA trả `2kr` thay vì `r(n+m)`.

Chấm bằng `is_equiv` bản gốc của MetaMath (`metamath_eval.py`), đúng thứ
PiSSA/PMSS dùng.

## 2. Kết quả

| | S-LoRA r=116 | LoRA r=64 | chênh |
|---|---|---|---|
| Tham số | 167,25M | 167,77M | −0,3% |
| Train loss | 0,1848 | **0,1543** | |
| Val loss | 0,1792 | **0,1523** | |
| **GSM8K** | **61,94%** (817/1319) | **72,02%** (950/1319) | **+10,08** |
| **MATH** | **13,30%** (665/5000) | **22,12%** (1106/5000) | **+8,82** |

LoRA thắng ở mọi chỉ số: train loss, val loss, và cả hai benchmark.

### MATH theo chủ đề — LoRA thắng cả bảy

| chủ đề | n | S-LoRA | LoRA | chênh |
|---|---|---|---|---|
| prealgebra | 871 | 25,26% | 39,61% | +14,35 |
| algebra | 1187 | 19,38% | 31,17% | +11,79 |
| geometry | 479 | 7,31% | 18,58% | +11,27 |
| counting_and_probability | 474 | 11,39% | 18,35% | +6,96 |
| intermediate_algebra | 903 | 5,87% | 10,74% | +4,87 |
| number_theory | 540 | 7,96% | 12,41% | +4,44 |
| precalculus | 546 | 5,49% | 9,34% | +3,85 |

Nhất quán trên toàn bộ, không có chủ đề nào S-LoRA thắng. Chênh lệch lớn nhất ở
các chủ đề *dễ* (prealgebra, algebra) — nơi cả hai còn nhiều dư địa.

## 3. Vì sao tin được con số này: pipeline đã được kiểm chứng

Đây là lý do run 2 được đổi từ S-LoRA r=64 sang LoRA. Baseline trích dẫn không
đủ tin, vì PiSSA và PMSS báo cáo lệch nhau 1,80 điểm GSM8K trên cùng method cùng
dữ liệu.

| LoRA 168M | GSM8K | MATH |
|---|---|---|
| **run này** | **72,02** | **22,12** |
| PiSSA Bảng 2 (gaussian) | 69,50 ±0,42 | 20,08 ±0,20 |
| PMSS Bảng 4 | 67,70 | 19,68 |

Pipeline này **vượt** cả hai baseline công bố, hơn 2,52 và 4,32 điểm GSM8K. Nên:

- Nghi vấn **thiếu token BOS** không gây tổn thất điểm nào đáng kể. Nó vẫn là một
  khác biệt thật so với MetaMath/PiSSA, nhưng không giải thích được gì.
- 61,94% của S-LoRA **không phải** hệ quả của đường ống. Nó là tính chất của
  phương pháp ở cấu hình này.

Kiểm thêm, đều âm tính:

- **Hàm chấm:** `is_equiv` của MetaMath và hàm tự viết khớp *tuyệt đối* trên
  GSM8K (61,94 vs 61,94; 72,02 vs 72,02), lệch 0,08–0,12 trên MATH.
- **Cắt ngắn:** `max_new_tokens = 512`, đúng MetaMath. Câu sinh dài trung bình
  378 (S-LoRA) và 379 (LoRA) ký tự — như nhau.
- **Định dạng:** soi câu sai, cả hai đều sinh lập luận đúng khuôn, sai vì **lập
  luận sai** chứ không vì hỏng output.
- **Thiếu nhãn `The answer is:`** — GSM8K: 8 vs 5 câu. MATH: 737 (14,7%) vs 647
  (12,9%). Chênh 1,8 điểm phần trăm, không đủ để giải thích khoảng cách 8,82.

## 4. S-LoRA còn chậm hơn

| | S-LoRA | LoRA |
|---|---|---|
| Train | 8620,7 s (0,090 it/s) | **7165,4 s (0,108 it/s)** |
| Chấm GSM8K | 808,3 s | **743,5 s** |
| Chấm MATH | 6132,5 s | **5529,6 s** |
| VRAM đỉnh | 34,27 GB | 34,12 GB |

Chậm hơn **20% mỗi bước train** và ~9% khi sinh. Lợi thế tỷ lệ cạnh của S-LoRA
nằm ở **số tham số**, không ở chi phí tính toán — ở đây nó còn đắt hơn.

## 5. Đọc kết quả này thế nào

Kết quả trước đây trên Qwen2.5-1.5B / E2E NLG có lợi cho S-LoRA: ở 0,115M tham
số nó đạt 65,53 BLEU so với 65,61 của LoRA ở 0,803M — ngang nhau với 7× ít tham
số hơn; bản hybrid ở 0,287M thắng LoRA cả ba chỉ số.

Ở đây thì ngược hẳn. Khác biệt giữa hai bối cảnh:

| | E2E (thắng) | Toán 7B (thua) |
|---|---|---|
| Ngân sách | 0,06–0,8M | 167M |
| Nhiệm vụ | sinh văn bản bề mặt | suy luận nhiều bước |
| Model | 1,5B | 7B |
| Lớp | k/v hoặc q,k,v | cả 7 |

**Giả thuyết** (chưa kiểm): ràng buộc `row(ΔW) ⊆ row(W₀)` rẻ khi bản cập nhật
cần thiết nhỏ và mang tính bề mặt, nhưng đắt khi nhiệm vụ đòi hỏi thay đổi lớn
theo những hướng *mới*. Thêm rank không bù được — S-LoRA có rank 116 so với 64
của LoRA mà vẫn thua, tức là vấn đề nằm ở **hướng bị giới hạn**, không ở số
chiều.

Đây là giả thuyết, không phải kết luận. Muốn xác nhận cần quét ngân sách: nếu
khoảng cách thu hẹp khi giảm về 20M, 50M thì câu chuyện "ràng buộc đắt khi cập
nhật lớn" đứng vững; nếu không đổi thì nguyên nhân nằm chỗ khác.

## 6. Giới hạn của kết quả này

- **1 seed mỗi cấu hình.** PiSSA báo trung bình 3 run. Khoảng cách 10,08 điểm lớn
  hơn nhiều so với sd của PiSSA (±0,42 trên LoRA), nên khó là nhiễu — nhưng chưa
  đo được sd của chính mình.
- **lr chỉ quét cho S-LoRA**, hai điểm (2e-5, 2e-4; 2e-4 phân kỳ ngay ở loss
  5,15). LoRA dùng thẳng 2e-5 vì đó là giá trị PiSSA dùng cho LoRA. Không loại
  trừ được khả năng S-LoRA cần lr khác hẳn — nhưng 2e-4 đã phân kỳ, nên nếu có
  thì phải nằm giữa, và quét thưa như vậy là một điểm yếu thật.
- **Train 98.859 mẫu** thay vì 100.000 (tách 1% validation + 141 mẫu bị
  `max_len 512` cắt hết lời giải). Ít hơn PiSSA 1,14%, áp dụng như nhau cho cả
  hai run.
- **Không có BOS** ở cả train lẫn eval, khác MetaMath/PiSSA. Đã chứng minh không
  gây hại (LoRA vẫn vượt baseline công bố) nhưng vẫn là một sai khác chưa sửa.
- **PMSS và CURLoRA chưa chạy thật**, chỉ đối chiếu số công bố.

## 7. Cảnh báo về baseline công bố

PMSS chú thích Bảng 4 rằng các baseline có dấu † là *"taken from Meng et al.
(2024)"*, tức PiSSA. Nhưng **không con nào** trong năm số Mistral-7B đó — 67,02 /
67,70 / 72,86 / 19,68 / 21,54 — xuất hiện trong paper PiSSA (bản camera-ready
31/10/2024, đã tìm toàn văn). Nhiều khả năng PMSS trích arXiv v1 hồi 4/2024.
Hệ quả: baseline lệch tới **2,89 điểm GSM8K** tuỳ bảng nào được đọc.

Và đặt cạnh số PiSSA tự công bố thì PMSS **hoà** GSM8K (73,31 vs 73,31) và **thua**
MATH (21,34 vs 23,12) — ở nửa số tham số, vẫn là kết quả thật, nhưng cách kể
"PMSS vượt PiSSA" chỉ đứng vững khi so với bản số cũ mà chính họ chép lại.

## 8. Số đo hiệu năng

| | |
|---|---|
| Phân rã, lần đầu | 19,1 phút (`geqp3` trên CPU, 26 core) |
| Phân rã, có cache pivot | **4 giây** |
| Sai số kiểm tra phân rã | 6,12e-06 |
| Train mỗi run | 2,0–2,4 giờ |
| Chấm 6.319 bài | 1,7–1,9 giờ |

Cache pivot lưu **chỉ chỉ số hoán vị** (25 MB); lưu thẳng `X` sẽ tốn ~32 GB.
Dựng lại `W₂` và `X` bằng một phép giải hệ `k×k` trên GPU.
