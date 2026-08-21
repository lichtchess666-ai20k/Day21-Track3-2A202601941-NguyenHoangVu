# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hoàng Vũ  **MSSV**: 2A202601941  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Colab Free T4 (14,6 GB khả dụng), fp16

> ⚠ **LƯỢT ĐO NÀY CHẠY Ở CHẾ ĐỘ SMOKE.** `baselines_frozen.json` ghi `eval_limit: 8`,
> `smoke_mode: true` — mọi điểm eval dưới đây tính trên **8/50** ticket target và **8/15**
> câu regression. Phần **huấn luyện là đầy đủ** (`max_steps=30`, tức `EPOCHS=2`, cả bốn
> run). Các số ở §3, §5, §6 và cột `target` ở §4 **phải đo lại** bằng một lượt NB2+NB5
> không đặt `EVAL_LIMIT` trước khi nộp; §1, §2 và các cột huấn luyện ở §4 không đổi.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định, không đổi) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** (mặc định tier T4) — p95 đo được là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** (cả bốn run, `results/runs.csv`) |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*. Chuỗi
render ra giữ nguyên `<think>...</think>` cùng phần thân, `verdict: "reasoning preserved
— safe to train on traces"`. Không phải xử lý gì thêm.

**Vì sao `max_length=1024` chứ không phải 256 như p95 gợi ý.** NB1 cảnh báo đúng: phân bố
độ dài đo được là mean 93,1 · p50 93 · p95 98 · p99 100 · **max 101** token, nên
`_round_pow2(p95)` đề xuất 256. Tôi giữ 1024 của tier và đây **không phải** một con số
đoán: mẫu dài nhất trong corpus là 101 token, tức `max_length=1024` **không cắt cụt một
mẫu nào** — nó là trần không ràng buộc. Với `per_device_batch=1` và padding động, trần
này cũng không tốn thêm VRAM (đo được 12,01 GB, khớp mức ~10 GB nhà cung cấp công bố cho
bf16 LoRA 4B). Nếu corpus đổi sang loại có đuôi dài, con số phải đặt lại theo p95 mới.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0,4149** (39/94 token) |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (`masked_preview`) — toàn bộ system + user + phần mở đầu lượt assistant:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

Đọc được gì: 41% token nằm trong loss, phần còn lại là prompt. Con số này xa ngưỡng hỏng
`≥ 0,95` — nếu nó xấp xỉ 1,0 thì nghĩa là model đang bị dạy chép lại chính câu hỏi. Ranh
giới rơi đúng sau `</think>`, tức phần được giám sát bắt đầu ngay tại JSON.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0,0000 | 0,750 | 0,000 | 3498,3 |
| (b) base + optimized prompt | 0,6875 | 0,750 | 1,000 | 1065,9 |
| (c) LoRA fine-tune | **0,9375** | **0,625** | 1,000 | 1718,6 |

**(b) có thật sự mạnh hơn (a) không?** **Có, và cách biệt rất lớn.** `format` đi từ
**0,000 → 1,000**: với prompt ngây thơ, model **không sinh ra nổi JSON hợp lệ một lần
nào**, nên `target = 0` không phải vì phân loại sai mà vì không có gì để chấm. Prompt tối
ưu vá đúng chỗ đó và kéo `target` lên 0,6875. Nó còn nhanh hơn gấp 3,3 lần (3498 → 1066
ms) vì model ngừng viết văn xuôi lan man.

Tôi **không sửa** `OPTIMIZED_PROMPT`: `optimized_prompt_sha = 719e74d3b6232053`, và
`verify.py` xác nhận `baseline (b) prompt unmodified`. Mốc phải vượt vì thế là 0,6875 —
mốc thật, không phải mốc dựng lên cho dễ thắng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.464.896 | 1e-4 | 0,6265 | **0,9375** | 1,000 | 989,6 | 12,01 |
| `attn_only` | q,v | **283** *(matched)* | 32.456.704 | 1e-4 | **0,5381** | **0,9375** | 1,000 | 834,0 | 12,02 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | **1e-5** | 1,5704 | **0,0000** | 0,000 | 972,6 | 12,01 |
| `qlora` | text-linear | 16 | 32.464.896 | 1e-4 | 0,7058 | 0,8438 | 1,000 | 1039,7 | **7,09** |

Cả bốn run cùng `max_steps = 30`; `attn_only` lệch ngân sách tham số **0,025%** so với
`correct` — `verify.py` báo `attn_only is a FAIR contrast`. Mỗi run đổi **đúng một biến**:
`attn_only` đổi **vị trí gắn adapter** (và nâng r để giữ nguyên số tham số), `wrong_lr` đổi
**learning rate**, `qlora` đổi **độ chính xác của base** (4-bit).

**Hai cột xếp hạng KHÔNG giống nhau:**

* theo `final_loss`: `attn_only` (0,5381) **thắng** `correct` (0,6265)
* theo `target`: hai run **hoà** đúng 0,9375

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`.** Trên tập target nó
**hoà**, không thắng cũng không thua. Nhưng theo cột train loss nó **thắng rõ rệt** — và
đó chính là cái bẫy của lab này. Nếu chấm bằng loss, tôi đã kết luận "gắn adapter chỉ vào
q,v rồi nâng rank là cấu hình *tốt hơn*", trong khi thang đo tác vụ nói nó chỉ *ngang
bằng*. Loss thấp hơn ở đây gần như chắc chắn là ghi nhớ: r=283 trên 225 mẫu và 30 step là
thừa dung lượng để thuộc lòng tập train mà không khá hơn trên dữ liệu chưa thấy. Về *rank
vs vị trí*: khi ngân sách tham số đã bị ghim, **đổi vị trí không đổi kết quả trên tác vụ
này**. Điều đó không bác bỏ Lỗi #1 của deck — nó khoanh vùng: triage JSON 4 trường là tác
vụ hẹp, `format` đã bão hoà ở 1,000, và cả hai cấu hình đều thừa sức. Muốn phân biệt vị trí
với rank thì phải có tác vụ khó hơn hoặc tập eval lớn hơn 8 mẫu. Một điểm phụ đáng chú ý:
`attn_only` sinh nhanh hơn hẳn (927 vs 1719 ms) vì chỉ phải chạy nhánh adapter trên 2 loại
module thay vì 12.

**4.2 — `wrong_lr` chỉ khác đúng một con số** (1e-5 thay vì 1e-4) **và nó sập hoàn toàn.**
Loss cuối 1,5704 so với 0,6265, gấp 2,5 lần — đường loss không đi xuống tới vùng hữu ích
trong 30 step. Trên tác vụ: `target = 0,000` **và** `format = 0,000`, nghĩa là nó còn không
học nổi việc "trả lời bằng JSON". Đây là run duy nhất mà cột loss xếp hạng **đúng**: sai
10× LR là sai đủ lớn để lộ ra ngay cả trên một chỉ số tồi. Nếu chỉ nhìn loss mà không biết
LR, kết luận sai sẽ là *"dữ liệu quá khó / model không học được tác vụ này"* — trong khi ba
run còn lại dùng đúng dữ liệu đó và đạt 0,84–0,94. Với LoRA, LR phải ở thang **~10×
full-fine-tune**; lấy thang full-FT áp vào là bỏ phí toàn bộ ngân sách huấn luyện. Latency
5405 ms cũng là triệu chứng: model không biết dừng nên viết lan man tới trần token.

**4.3 — `qlora` tiết kiệm 41,0% VRAM** (7,09 vs 12,01 GB) và **trả giá 9,4 điểm target**
(0,8438 vs 0,9375), `format` vẫn giữ 1,000, thời gian train nhỉnh hơn 5% (1039,7 vs 989,6 s
— lượng tử hoá không miễn phí về tốc độ). Số đo **ủng hộ** khuyến nghị "không dùng QLoRA
cho dòng Qwen3.5" ở đúng bối cảnh của lab: tại tier T4, fp16 LoRA **vẫn vừa card** (12,01
trong 14,6 GB), nên khoản tiết kiệm VRAM không mua được gì mà lại mất chất lượng — đánh đổi
thua thiệt. Nhưng nó cũng cho thấy khuyến nghị đó **không tuyệt đối**: 41% là mức tiết kiệm
rất lớn, và nếu buộc phải chạy một model 9B trên chính chiếc T4 này thì 0,84 có model còn
hơn 0,94 không chạy nổi. Kết luận đúng là "đừng dùng QLoRA khi chưa cần", không phải "QLoRA
vô dụng".

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0,250` · `regression Δ = -0,125` (ngưỡng 0,020) · `valid_trace_rate = 0,00`

Lý do gate đưa ra: *"general capability regressed by 0.125 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune **thắng đậm trên tác vụ đích**: 0,6875 → 0,9375, tức +0,25 so
với baseline (b) đã được prompt tử tế — và thắng trong khi **chỉ dùng prompt ngây thơ**,
đúng mục tiêu của fine-tuning là dời hành vi vào trọng số để prompt co lại. `format` giữ
nguyên 1,000. Nhưng nó **trượt cổng** vì `regression` tụt 0,750 → 0,625: model đã đánh đổi
năng lực phổ thông lấy năng lực chuyên biệt, tức **quên thảm hoạ** ở dạng kinh điển. Nguyên
nhân trực tiếp nằm ở dữ liệu: 225/225 mẫu huấn luyện đều là ticket → JSON, không có một mẫu
phổ thông nào làm đối trọng. Cách sửa theo deck §14.3 là trộn 1–5% dữ liệu phổ thông vào
tập train (replay) rồi train lại đúng 30 step đó — chứ **không** phải nới ngưỡng 0,020 cho
tới khi gate xanh.

**Một cảnh báo thành thật về độ tin cậy của chính con số này.** Tập regression ở lượt đo
này chỉ có **8 câu**, nên bước nhảy nhỏ nhất là 1/8 = 0,125 — tức là **cổng bị trượt bởi
đúng một câu hỏi**. Kết luận "có hồi quy" đúng về dấu, nhưng độ lớn thì chưa đo được: -0,125
ở đây có thể là -0,03 hoặc -0,20 trên tập đủ 15 câu. Tương tự, `target` chấm trên 8 ticket ×
4 trường = 32 quyết định, bước nhảy 0,031 — nên chữ "hoà" giữa `correct` và `attn_only` ở §4
thực chất có nghĩa là "không phân biệt được ở độ phân giải này". Đây chính là lý do lượt nộp
phải bỏ `EVAL_LIMIT`.

**`valid_trace_rate = 0,00` không phải lỗi.** Template *có* giữ `<think>` (§1), nhưng toàn
bộ 250 câu trả lời huấn luyện là JSON trần, và template đóng sẵn khối `<think></think>` rỗng
trong prompt sinh — nên trong vùng được giám sát không còn dấu vết suy luận nào để học. Model
không sinh trace vì nó chưa từng được dạy sinh trace. Muốn chạy phần tương phản §13.5 (thưởng
B3) thì phải có corpus mà câu trả lời chứa trace thật.

**Latency: một thứ fine-tune làm tệ đi.** 1065,9 → 1718,6 ms/mẫu, chậm hơn baseline (b) 653
ms dù prompt đã ngắn hơn nhiều. Thủ phạm là adapter chưa merge — mỗi lớp phải cộng thêm một
nhánh LoRA lúc suy luận. Đây đúng là thứ NB6 (`merge_and_serve`) xử lý, và là lý do kỹ thuật
để làm phần thưởng B1.

---

## 6. Định tính — bắt buộc có cả ca THUA

> ⚠ Cột **(b) prompt** chưa điền được: lượt chạy này dùng NB2 bản cũ, chưa sinh
> `results/baseline_preds.json`, nên không có dự đoán của baseline (b) trên **từng** ticket
> để đặt cạnh. Lượt chạy lại sẽ sinh file đó và bảng này điền được đầy đủ. Ngoài ra ở
> `EVAL_LIMIT=8`, model **không thua ca nào** (điểm thấp nhất là 0,75) — số liệu hiện có
> không đủ để chỉ ra 2 ca thua như rubric 3.4 đòi.

| # | Ticket (rút gọn) | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây... Cho tôi trả lại | *(chờ đo lại)* | `doi_tra / cao / chuột không dây / tich_cuc` — **1,00** | ✅ đúng 4/4 trường |
| 2 | Xin chào, mình đặt đèn bàn LED... Hoàn tiền. Quá hạn rồi | *(chờ đo lại)* | `hoan_tien / cao / đèn bàn LED / tich_cuc` — **1,00** | ✅ bắt đúng `urgency=cao` từ "quá hạn" |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt... **Chưa thấy tiền** | *(chờ đo lại)* | `hoan_tien / trung_binh / bình giữ nhiệt / ...` — **0,75** | ❌ sai 1/4 trường |
| 4 | Shop ơi, mình đặt nồi chiên không dầu... **Thiếu phụ kiện** | *(chờ đo lại)* | `san_pham_loi / trung_binh / nồi chiên không dầu / ...` — **0,75** | ❌ sai 1/4 trường |
| 5 | Alo shop, mình đặt máy xay sinh tố... Muốn đổi. Đã 3 ngày | *(chờ đo lại)* | `doi_tra / trung_binh / máy xay sinh tố / ...` — **1,00** | ✅ |

**Mẫu chung ở hai ca yếu nhất (#3, #4):** cả hai đều **không nói thẳng ý định**. "Chưa thấy
tiền" và "thiếu phụ kiện" là *mô tả triệu chứng*, phải suy ra ý định; còn những ca đạt 1,00
đều chứa động từ trực tiếp ("trả lại", "hoàn tiền", "muốn đổi"). Trường bị sai cũng luôn là
trường phải suy luận (`urgency` / `sentiment`), không bao giờ là `product` — trích tên sản
phẩm là việc sao chép, model làm đúng 8/8. Nói cách khác: fine-tune đã học **rất chắc phần
ánh xạ bề mặt** và vẫn yếu ở phần cần suy luận ngầm.

---

## 7. Kết luận & điều tôi học được

**Kết luận.**

> *(Đoạn dưới là bản nháp dựng từ số đo — đọc lại, sửa theo cách hiểu của bạn rồi giữ ≥150
> từ. Đừng nộp nguyên văn nếu bạn không đồng ý với lập luận.)*

Tôi **chưa deploy** bản fine-tune này, dù nó thắng baseline (b) tới +0,25 trên tác vụ đích
với prompt ngắn hơn. Lý do không nằm ở tác vụ đích mà ở cái giá kèm theo: `regression` tụt
0,125 trong khi ngưỡng là 0,020, tức model đã trả bằng năng lực phổ thông; và latency tăng
653 ms/mẫu vì adapter chưa merge. Cả hai đều có đường sửa đã biết — trộn 1–5% dữ liệu replay
theo deck §14.3, và merge adapter ở NB6 — nên kết luận đúng là *"chưa xong"*, không phải
*"không nên fine-tune"*. Về đòn bẩy: trong bốn cấu hình đo được, **learning rate là biến duy
nhất tạo ra khác biệt sinh tử**. Sai LR 10× kéo target từ 0,94 xuống 0,00. Đổi vị trí adapter
khi đã ghim ngân sách tham số: **không đổi gì** (hoà 0,9375). Lượng tử hoá 4-bit: mất 9,4
điểm target để đổi 41% VRAM. Còn chất lượng dữ liệu quyết định thứ mà không cấu hình nào cứu
được — chính sự vắng mặt của dữ liệu phổ thông trong tập train là nguyên nhân trực tiếp làm
trượt cổng hồi quy. Bài học lớn nhất lại đến từ chỗ khác: **cột `final_loss` xếp `attn_only`
thắng `correct`, còn thang đo tác vụ nói hai run hoà.** Nếu tôi dừng ở chỉ số thay thế, tôi
đã báo cáo một kết luận mà dữ liệu không ủng hộ.

**Ba điều tôi học được** (cụ thể, không generic):

1. *(của bạn — gợi ý: `format = 0,000` ở baseline (a) nói gì về khoảng cách giữa "model dở"
   và "prompt dở"?)*
2. *(của bạn — gợi ý: trước khi thấy `attn_only` loss 0,5381 mà chỉ hoà trên target, bạn tin
   gì về cột loss?)*
3. *(của bạn — gợi ý: -0,125 trên 8 câu là đúng một câu; điều đó đổi cách bạn đọc mọi con số
   trong report này thế nào?)*

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** *(của bạn — ứng viên tự nhiên: trộn 1–5% replay rồi
train lại đúng 30 step để xem cổng hồi quy có xanh mà không mất target; hoặc chạy NB6 để đo
lại latency sau merge.)*

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse — **không khả thi trên corpus mặc định**: `valid_trace_rate = 0,00`, câu trả lời huấn luyện là JSON trần nên `masked-think` / `response-only` / `assistant-only` sinh ra mask giống hệt nhau
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
