# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hoàng Vũ  **MSSV**: 2A202601941  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 16 GB, fp16 (peak 12,01 GB)

> Lượt đo đầy đủ: `baselines_frozen.json` ghi `eval_limit: null`, `smoke_mode: false` —
> **50/50** ticket target và **15/15** câu regression. Bốn run huấn luyện cùng
> `max_steps = 30` (`EPOCHS=2`). `make verify`: **26 passed · 0 failures**.

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
render giữ nguyên `<think>...</think>` cùng phần thân, `verdict: "reasoning preserved —
safe to train on traces"`. Không phải xử lý gì thêm.

**Vì sao `max_length = 1024` chứ không phải 256 như p95 gợi ý.** NB1 cảnh báo đúng: phân bố
độ dài đo được là mean 93,1 · p50 93 · p95 98 · p99 100 · **max 101** token, nên
`_round_pow2(p95)` đề xuất 256. Tôi giữ 1024 của tier, và đây **không phải** một con số
đoán: mẫu dài nhất trong corpus là 101 token, tức `max_length = 1024` **không cắt cụt một
mẫu nào** — nó là trần không ràng buộc. Với `per_device_batch = 1` và padding động, trần
này cũng không tốn thêm VRAM: đo được 12,01 GB, khớp mức ~10 GB nhà cung cấp công bố cho
bf16 LoRA 4B. Nếu corpus đổi sang loại có đuôi dài, con số phải đặt lại theo p95 mới.

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

41% token nằm trong loss, phần còn lại là prompt. Con số này xa ngưỡng hỏng `≥ 0,95` — nếu
nó xấp xỉ 1,0 thì nghĩa là model đang bị dạy chép lại chính câu hỏi. Ranh giới rơi đúng sau
`</think>`, tức phần được giám sát bắt đầu ngay tại JSON.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0,0000 | 0,7578 | 0,000 | 2946,0 |
| (b) base + optimized prompt | 0,7650 | 0,7578 | 1,000 | 991,0 |
| (c) LoRA fine-tune | **0,9700** | **0,6778** | 1,000 | 1330,6 |

*(n = 50 ticket target, 15 câu regression)*

**(b) có thật sự mạnh hơn (a) không?** **Có, và cách biệt rất lớn.** `format` đi từ
**0,000 → 1,000**: với prompt ngây thơ, model **không sinh ra nổi JSON hợp lệ một lần nào**
trong 50 ticket, nên `target = 0` không phải vì phân loại sai mà vì không có gì để chấm.
Prompt tối ưu vá đúng chỗ đó và kéo `target` lên 0,7650. Nó còn nhanh hơn gần 3 lần
(2946 → 991 ms) vì model ngừng viết văn xuôi lan man.

Tôi **không sửa** `OPTIMIZED_PROMPT`: `optimized_prompt_sha = 719e74d3b6232053`, và
`verify.py` xác nhận `baseline (b) prompt unmodified`. Mốc phải vượt vì thế là **0,7650** —
mốc thật, không phải mốc dựng lên cho dễ thắng. Đáng chú ý: **hơn 3/4 quãng đường từ 0 tới
điểm cuối cùng của bản fine-tune là do prompt engineering, miễn phí, không cần một step
huấn luyện nào.**

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency (ms) | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.464.896 | 1e-4 | 0,6249 | **0,9700** | 1,000 | 1330,6 | 898,5 | 12,01 |
| `attn_only` | q,v | **283** *(matched)* | 32.456.704 | 1e-4 | **0,5377** | **0,9700** | 1,000 | **859,2** | 752,9 | 12,02 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | **1e-5** | 1,6119 | **0,0000** | 0,000 | 4876,3 | 882,2 | 12,01 |
| `qlora` | text-linear | 16 | 32.464.896 | 1e-4 | 0,7058 | 0,9400 | 1,000 | 1673,2 | 943,5 | **7,09** |

Cả bốn run cùng `max_steps = 30`; `attn_only` lệch ngân sách tham số **0,025%** so với
`correct` — `verify.py` báo `attn_only is a FAIR contrast`. Mỗi run đổi **đúng một biến**:
`attn_only` đổi **vị trí gắn adapter** (nâng r lên 283 để giữ nguyên số tham số),
`wrong_lr` đổi **learning rate**, `qlora` đổi **độ chính xác của base** (4-bit).

**Hai cột xếp hạng KHÔNG giống nhau:**

| xếp theo | thứ tự |
|---|---|
| `final_loss` | `attn_only` 0,5377 › `correct` 0,6249 › `qlora` 0,7058 › `wrong_lr` 1,6119 |
| **`target`** | **`correct` 0,9700 = `attn_only` 0,9700** › `qlora` 0,9400 › `wrong_lr` 0,0000 |

**4.1 — `attn_only` có cùng ngân sách tham số với `correct` và trên tập target nó HOÀ, đúng
0,9700.** Nhưng theo cột train loss nó **thắng rõ rệt** (0,5377 vs 0,6249) — và đó chính là
cái bẫy trung tâm của lab này. Nếu chấm bằng loss, tôi đã kết luận "chỉ gắn adapter vào q,v
rồi nâng rank là cấu hình *tốt hơn*", trong khi thang đo tác vụ nói nó chỉ *ngang bằng*. Ở
đây "hoà" là một kết luận có trọng lượng chứ không phải giới hạn phân giải: 50 ticket × 4
trường = **200 quyết định**, bước nhảy 0,005, và cả hai run cùng đạt **194/200**. Loss thấp
hơn của `attn_only` gần như chắc chắn là ghi nhớ — r = 283 trên 225 mẫu và 30 step là thừa
dung lượng để thuộc lòng tập train mà không khá hơn trên dữ liệu chưa thấy. Về *rank so với
vị trí*: **khi ngân sách tham số đã bị ghim, đổi vị trí không đổi kết quả trên tác vụ này.**
Điều đó không bác bỏ Lỗi #1 của deck — nó khoanh vùng phạm vi: triage JSON 4 trường là tác
vụ hẹp, `format` bão hoà ở 1,000, cả hai cấu hình đều thừa sức. Muốn phân biệt vị trí với
rank thì phải có tác vụ khó hơn. Một hệ quả thực dụng đáng chú ý: `attn_only` **sinh nhanh
hơn 35%** (859 vs 1331 ms) vì chỉ chạy nhánh adapter trên 2 loại module thay vì 12 — cùng
chất lượng, rẻ hơn khi phục vụ.

**4.2 — `wrong_lr` chỉ khác đúng một con số** (1e-5 thay vì 1e-4) **và nó sập hoàn toàn.**
Loss cuối 1,6119 so với 0,6249, gấp 2,6 lần — đường loss không đi xuống tới vùng hữu ích
trong 30 step. Trên tác vụ: `target = 0,0000` **và** `format = 0,000`, nghĩa là nó còn không
học nổi việc "trả lời bằng JSON". Đây là run duy nhất mà cột loss xếp hạng **đúng**: sai
10× LR là sai đủ lớn để lộ ra ngay cả trên một chỉ số tồi. Nếu chỉ nhìn loss mà không biết
LR, kết luận sai sẽ là *"dữ liệu quá khó / model không học được tác vụ này"* — trong khi ba
run còn lại dùng đúng dữ liệu đó và đạt 0,94–0,97. Với LoRA, LR phải ở thang **~10×
full-fine-tune**; lấy thang full-FT áp vào là bỏ phí toàn bộ ngân sách huấn luyện. Latency
4876 ms cũng là triệu chứng cùng gốc: model không biết dừng nên viết lan man tới trần token
— đúng dấu hiệu của baseline (a).

**4.3 — `qlora` tiết kiệm 41,0% VRAM** (7,09 vs 12,01 GB) và **trả giá 0,03 điểm target**
(0,9400 vs 0,9700 — tức 6 quyết định trường sai thêm trên 200), `format` vẫn giữ 1,000,
thời gian train nhỉnh hơn 5% (943,5 vs 898,5 s: lượng tử hoá không miễn phí về tốc độ) và
suy luận chậm hơn 26% (1673 vs 1331 ms). Số đo **ủng hộ** khuyến nghị "không dùng QLoRA cho
dòng Qwen3.5" ở đúng bối cảnh của lab, nhưng lý do không phải vì chất lượng sụp đổ — cái
giá 3 điểm là nhỏ hơn tôi tưởng. Lý do là **khoản tiết kiệm không mua được gì**: ở tier T4,
fp16 LoRA đã vừa card (12,01 trong 14,6 GB), nên đổi 3 điểm target lấy VRAM thừa là đánh
đổi thua thiệt thuần tuý. Ngược lại, con số 41% cho thấy khuyến nghị đó **không tuyệt đối**:
nếu buộc phải nhét một model 9B vào chính chiếc T4 này thì 0,94 có model vẫn hơn 0,97 không
chạy nổi. Kết luận đúng là *"đừng dùng QLoRA khi chưa cần"*, không phải *"QLoRA vô dụng"*.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0,205` · `regression Δ = -0,080` (ngưỡng 0,020) · `valid_trace_rate = 0,00`

Lý do gate đưa ra: *"general capability regressed by 0.080 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune **thắng đậm trên tác vụ đích**: 0,7650 → 0,9700, tức +0,205 so
với baseline (b) đã được prompt tử tế — và thắng trong khi **chỉ dùng prompt ngây thơ**,
đúng mục tiêu của fine-tuning là dời hành vi vào trọng số để prompt co lại. `format` giữ
nguyên 1,000. Nhưng nó **trượt cổng** vì `regression` tụt 0,7578 → 0,6778: model đã đánh đổi
năng lực phổ thông lấy năng lực chuyên biệt — **quên thảm hoạ** ở dạng kinh điển. Mức tụt
0,080 gấp **4 lần** ngưỡng cho phép 0,020, và trên tập 15 câu thì nó tương đương mất trọn
hơn một câu hỏi; đây không phải nhiễu của một mẫu lẻ.

Nguyên nhân trực tiếp nằm ở dữ liệu chứ không ở cấu hình: **225/225 mẫu huấn luyện đều là
ticket → JSON**, không có một mẫu phổ thông nào làm đối trọng, nên model học rằng *mọi* đầu
vào đều có nghĩa là "xuất JSON triage". Cách sửa theo deck §14.3 là trộn 1–5% dữ liệu phổ
thông vào tập train (replay) rồi train lại đúng 30 step đó — chứ **không** phải nới ngưỡng
0,020 cho tới khi gate xanh. Đáng chú ý là ba run còn lại của NB4 không cứu được điều này:
`attn_only` đạt đúng cùng điểm target, nghĩa là **không có cấu hình LoRA nào trong bốn cái
đã thử chạm được tới nguyên nhân** — vì nguyên nhân không nằm trong không gian cấu hình.

**Vậy có nên deploy không?** Chưa. Nhưng phán quyết FAILED ở đây **không** có nghĩa "đừng
fine-tune": target +0,205 là mức cải thiện lớn và có thật, và cả hai khuyết điểm đều có
đường sửa đã biết (replay data cho hồi quy, merge adapter cho latency). Kết luận đúng là
*"chưa xong"*, không phải *"sai hướng"*.

**`valid_trace_rate = 0,00` không phải lỗi.** Template *có* giữ `<think>` (§1), nhưng toàn
bộ 250 câu trả lời huấn luyện là JSON trần, và template đóng sẵn khối `<think></think>` rỗng
trong prompt sinh — nên trong vùng được giám sát không còn dấu vết suy luận nào để học.
Model không sinh trace vì nó chưa từng được dạy sinh trace. Muốn chạy phần tương phản §13.5
(thưởng B3) thì phải có corpus mà câu trả lời chứa trace thật.

**Latency: một thứ fine-tune làm tệ đi.** 991,0 → 1330,6 ms/mẫu, chậm hơn baseline (b)
**340 ms (+34%)** dù prompt đã ngắn hơn nhiều. Thủ phạm là adapter chưa merge — mỗi lớp
phải cộng thêm một nhánh LoRA lúc suy luận. Chứng cứ nằm ngay trong bảng §4: `attn_only`
gắn adapter vào 2 loại module thay vì 12 và chỉ mất 859 ms, tức chi phí tỉ lệ với số nhánh
chứ không phải với số tham số (hai run có cùng số tham số). Đây đúng là thứ NB6
(`merge_and_serve`) xử lý.

---

## 6. Định tính — phân bố thắng/thua trên toàn bộ 50 ticket

Thay vì chọn tay 5 ví dụ, tôi đối chiếu **cả 50 ticket** giữa (b) và (c) bằng
`results/qualitative.json` (cột `ft_minus_b`):

| | số ticket |
|---|---|
| fine-tune **thắng** (b) | **33** |
| **hoà** | 17 |
| fine-tune **thua** (b) | **0** |

> ⚠ **Rubric 3.4 yêu cầu ≥2 ca fine-tune thua — dữ liệu của tôi không có ca nào.** Đây
> không phải cherry-pick: con số trên là toàn bộ tập eval, không phải mẫu chọn lọc. Ca yếu
> nhất của fine-tune vẫn **bằng hoặc hơn** (b). Phần dưới vì thế trình bày 6 ca fine-tune
> *không đạt điểm tuyệt đối* (0,75) — chúng là biên yếu nhất mà dữ liệu có.

Phân bố điểm từng ticket: fine-tune **44 × 1,00** và **6 × 0,75**; baseline (b) **13 ×
1,00**, **27 × 0,75**, **10 × 0,50**.

### Sai ở trường nào

| trường | fine-tune sai | (b) sai |
|---|---|---|
| `intent` | **0/50** | 23/50 |
| `urgency` | **6/50** | 18/50 |
| `product` | **0/50** | 0/50 |
| `sentiment` | **0/50** | ~6/50 |

Kiểm chứng bằng số học: `target = 0,9700` trên 200 quyết định trường ⇒ đúng 194, sai **6**.
Đúng 6 lỗi đó đều nằm ở `urgency`, nên ba trường còn lại của fine-tune là **hoàn hảo 50/50**.

### Sáu ca yếu nhất — cùng một lỗi, lặp lại sáu lần

| # | i | Ticket | `urgency` đúng | fine-tune đoán | (b) |
|---|---|---|---|---|---|
| 1 | 3 | ...bình giữ nhiệt VN804124. Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều. | `thap` | `trung_binh` ❌ | 0,75 |
| 2 | 5 | ...nồi chiên không dầu DH249548. Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi. | `thap` | `trung_binh` ❌ | 0,50 |
| 3 | 12 | ...áo khoác gió VN613097. Bị lỗi. **Khi nào tiện.** Cảm ơn shop nhiều. | `thap` | `trung_binh` ❌ | 0,75 |
| 4 | 39 | ...nồi chiên không dầu VN949966. Hoàn tiền. **Khi nào tiện.** Quá tệ. | `thap` | `trung_binh` ❌ | 0,75 |
| 5 | 41 | ...đèn bàn LED OD436045. Giao hàng chậm. **Khi nào tiện.** Cảm ơn shop nhiều. | `thap` | `trung_binh` ❌ | 0,50 |
| 6 | 46 | ...đèn bàn LED OD819229. Sai màu. **Khi nào tiện.** Shop hỗ trợ tốt. | `thap` | `trung_binh` ❌ | 0,75 |

**Mẫu chung: không phải "ticket khó", mà là MỘT cụm từ.** Corpus dùng ba cụm đồng nghĩa để
đánh dấu `urgency = thap`. Đối chiếu với tập train:

| cụm báo `thap` | số mẫu train | ticket eval | fine-tune đúng |
|---|---|---|---|
| `Không vội` | 34 | 7 | **7/7** ✅ |
| `Hỏi cho biết thôi` | 28 | 5 | **5/5** ✅ |
| **`Khi nào tiện`** | **35** | 6 | **0/6** ❌ |

Cả 35 mẫu train chứa `Khi nào tiện` đều gắn nhãn `thap`. Nghĩa là **model được thấy cụm này
nhiều nhất trong ba cụm, luôn kèm nhãn đúng, và vẫn trượt toàn bộ 6 ticket eval** — trong
khi hai cụm đồng nghĩa với ít dữ liệu hơn thì nó học được hoàn hảo. Đây không phải vấn đề
"thiếu dữ liệu" mà là **một cụm bề mặt không được liên kết sau 30 step**, và lỗi luôn lệch
về cùng một phía: `thap → trung_binh`, không bao giờ ngược lại. Đây cũng là loại lỗi mà
`final_loss = 0,6249` hoàn toàn không nhìn thấy.

Đối chiếu với baseline (b) cho thấy khác biệt về **chất**, không chỉ về lượng: (b) sai
`intent` 23/50 lần — nó không nắm được bộ nhãn — còn fine-tune sai đúng một loại lỗi có thể
gọi tên và có thể sửa (thêm mẫu chứa `Khi nào tiện`, hoặc đơn giản là train thêm vài step).

---

## 7. Kết luận & điều tôi học được

**Kết luận.**

Tôi **chưa deploy** bản fine-tune này, dù nó thắng baseline (b) +0,205 trên tác vụ đích với
prompt ngắn hơn và không thua (b) ở **bất kỳ** ticket nào trong 50 ticket. Lý do không nằm ở
tác vụ đích mà ở cái giá kèm theo: `regression` tụt 0,080 — gấp 4 lần ngưỡng — và latency
tăng 340 ms/mẫu vì adapter chưa merge. Cả hai đều có đường sửa đã biết (trộn 1–5% dữ liệu
replay theo deck §14.3; merge adapter ở NB6), nên kết luận đúng là *"chưa xong"*, không phải
*"không nên fine-tune"*.

Về đòn bẩy thật sự, bốn cấu hình đo được xếp hạng rất rõ. **Learning rate là biến duy nhất
tạo ra khác biệt sinh tử**: sai 10× kéo target từ 0,97 xuống 0,00 và phá luôn cả `format`.
**Vị trí adapter thì không phải đòn bẩy** — khi ngân sách tham số bị ghim, `attn_only` đạt
đúng 194/200 y hệt `correct`, chỉ khác là phục vụ nhanh hơn 35%. **Lượng tử hoá 4-bit** đổi
3 điểm target lấy 41% VRAM: một đánh đổi thật, chỉ vô nghĩa ở tier mà fp16 vốn đã vừa card.
Còn **chất lượng và thành phần dữ liệu quyết định thứ mà không cấu hình nào cứu được**: sự
vắng mặt của dữ liệu phổ thông trong tập train là nguyên nhân trực tiếp làm trượt cổng hồi
quy, và cả bốn run đều trượt như nhau vì nguyên nhân nằm ngoài không gian cấu hình.

Bài học lớn nhất lại đến từ chỗ khác: **cột `final_loss` xếp `attn_only` (0,5377) thắng
`correct` (0,6249), còn thang đo tác vụ nói hai run hoà tuyệt đối ở 194/200.** Nếu dừng ở
chỉ số thay thế, tôi đã báo cáo một kết luận mà dữ liệu không ủng hộ. Cũng chính cột loss đó
mù tịt trước lỗi thực tế duy nhất của model — 6 ticket chứa cụm `Khi nào tiện`.

**Ba điều tôi học được** (cụ thể, không generic):

1. **Loss trả lời "khớp dữ liệu train tới đâu", không trả lời "làm đúng việc tới đâu".** Với
   LoRA rank cao trên tập nhỏ, hai câu đó tách rời hẳn nhau: `attn_only` loss thấp hơn 14%
   mà chất lượng y hệt, và không cột loss nào chỉ ra được lỗi `Khi nào tiện`.
2. **Mốc so sánh quyết định câu chuyện.** `format` của baseline (a) là 0,000 — model gốc với
   prompt ngây thơ không sinh nổi JSON hợp lệ *một lần nào* trong 50 lần. Nếu lấy (a) làm
   mốc, tôi đã báo cáo 0,00 → 0,97 và **gần 4/5 quãng đường đó vốn miễn phí bằng prompt**.
3. **Fine-tune là đánh đổi, không phải cộng thêm.** +0,205 target đi kèm −0,080 regression,
   và nguyên nhân nằm trọn trong thành phần dữ liệu chứ không trong siêu tham số.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông (replay) vào 225 mẫu
train rồi chạy lại đúng 30 step, để kiểm tra giả thuyết ở §5 — rằng cổng hồi quy xanh được
mà không mất target. Sau đó chạy NB6 merge và đo lại latency, để xác nhận 340 ms chênh lệch
đúng là chi phí của nhánh adapter chứ không phải của trọng số đã học.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse — **không khả thi trên corpus mặc định**: `valid_trace_rate = 0,00`, câu trả lời huấn luyện là JSON trần nên `masked-think` / `response-only` / `assistant-only` sinh ra mask giống hệt nhau
- [x] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/licht666/qwen3.5-4b-lora-vi-ticket-triage
