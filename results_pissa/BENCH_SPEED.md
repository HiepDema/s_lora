# Đo tốc độ code mới — H100 PCIe, Mistral-7B, MetaMathQA

**Cả hai cấu hình chạy trên cùng một card, cùng dữ liệu, cùng 60 bước optimizer
ở batch hiệu dụng 128.** Không so với số cũ trên box SXM5 — đó là phần cứng
khác, sẽ lẫn giữa code và card.

Ngày đo: 2026-09-26. Box: H100 PCIe 80GB, 26 core, torch 2.13.0+cu130,
transformers 5.17.0, vllm 0.30.0.

## Train

| | cũ: fp32 + grad-ckpt, b8×16 | mới: bf16 + sdpa, b16×8 | |
|---|---|---|---|
| it/s | 0,063 | **0,149** | **2,37×** |
| 60 bước | 950 s | **402 s** | 2,36× |
| VRAM đỉnh | 34,25 GB | 61,50 GB | tăng 1,80× |

Bỏ gradient checkpointing đổi VRAM lấy tốc độ đúng như thiết kế. 61,5/80 GB
vẫn vừa nhưng biên mỏng — smoke test lúc đầu từng báo OOM với 2,4 MB trống
trước khi allocator tự gỡ được.

## Chấm điểm — chỗ ăn lớn nhất

Cùng 500 bài GSM8K, cùng trọng số, cùng greedy, cùng `metamath_eval.is_equiv`.

| | thời gian | mỗi bài | accuracy |
|---|---|---|---|
| `model.generate()` HF, batch 16 | 411 s | 0,82 s | 18,20% |
| **vLLM** | **9 s** | 0,02 s | 18,60% |
| | **44,7×** | | lệch +0,40 |

Accuracy khớp nhau trong 0,4 điểm → đường vLLM **đúng**, không phải nhanh nhờ
làm sai. Engine khởi động thêm 13,2 s; tính cả vào vẫn là ~18,7×.

## Chiếu lên run thật (100K mẫu, 1 epoch, 6.319 bài chấm) trên box PCIe này

| | cũ | mới |
|---|---|---|
| Train 773 bước | 3,58 h | 1,43 h |
| Chấm 6.319 bài | 1,44 h | ~0,04 h |
| **Tổng** | **5,0 h** | **1,5 h** |

**3,4×** — khớp sát ước tính 3,3× đưa ra trước khi đo.

## Ba cảnh báo

**1. Box PCIe chậm hơn SXM5 1,5×.** Cấu hình cũ: 0,063 it/s ở đây, 0,090 trên
SXM5. Đem 0,149 so thẳng với 0,090 sẽ ra "1,66×" — sai. Con số 2,37× là
code-only vì đo cùng card.

**2. Cấu hình fp32 phân kỳ trên stack này.** Loss của A đi lên đều
(1,33 → 2,59 → 3,40 → 3,86 → 4,21 → 4,47), val loss cuối **5,749**. Cấu hình B
(bf16) đi xuống bình thường, val loss **0,446**. Ngược hẳn với run thật trước
đây trên SXM5 (fp32, val 0,179).

Hai cấu hình khác nhau ở **hai biến** cùng lúc (dtype và grad-ckpt) nên
**chưa kết luận được** nguyên nhân. Ô thứ ba cần chạy: fp32 + tắt grad-ckpt.

Nghi can: `model.gradient_checkpointing_enable()` trước đây gọi không chỉ định
`use_reentrant`, mà transformers 4.x và 5.x chọn mặc định khác nhau — cùng một
dòng lệnh cho hai hành vi tuỳ box. Đã thêm `--ckpt-reentrant` để nói rõ, mặc
định là bản không reentrant. **Đây là phỏng đoán, chưa phải bằng chứng.**

Nếu grad-ckpt thật sự làm hỏng training âm thầm thì mọi run dùng cờ đó đều đáng
ngờ — kể cả hai run Mistral trong REPORT.md, vốn đều bật `--grad-ckpt` (nhưng
trên transformers 4.x, và loss ở đó giảm bình thường).

**3. bf16 làm sai số phân rã tăng 4.300 lần**: 2,66e-02 so với 4,74e-06 ở fp32.
Chỉ đụng S-LoRA (phải dựng lại W qua phân rã) chứ không đụng LoRA (dùng thẳng
W). Số **tốc độ** hợp lệ; số **accuracy** ở bf16 thì chưa biết có so được không.

## Sáu trở ngại đã phải gỡ khi dựng box

Ghi lại để lần sau dựng nhanh hơn:

1. `--model-dtype` / `--attn` mới chỉ thêm vào parser của `finetune_e2e.py`,
   trong khi `finetune_math.py` có parser riêng — cờ không tồn tại ở entry point
   thật sự dùng.
2. `--eval-before` bị `--skip-gsm8k` triệt tiêu (tập test rỗng thì không có gì
   để chấm).
3. Zero-shot chạm sàn 0% ở mọi cấu hình — Mistral chưa fine-tune không sinh
   `The answer is: `, mà bộ chấm MetaMath tính sai hết khi thiếu nhãn. Vô dụng
   làm phép thử phân biệt.
4. `pip install -q --user pandas` im lặng không cài (pip coi bản hệ thống là đã
   thoả) → xung đột ABI numpy 1.x/2.x. Phải dùng `--upgrade`.
5. `ml_dtypes` bản hệ thống kéo vào qua `xgrammar → tvm_ffi`, cũng build cho
   numpy 1.x → vLLM EngineCore chết. Phải `pip install --user --upgrade ml_dtypes`.
6. Thiếu `ninja` → flashinfer không JIT-compile được module sampling.

Lệnh dựng box gọn:

```bash
pip install --user vllm
pip install --user --upgrade pandas scikit-learn scipy ml_dtypes ninja datasets
export PATH=$HOME/.local/bin:$PATH
```
