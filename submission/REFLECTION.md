# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Rằng **train loss có thể xếp hạng ngược với chất lượng thật**. `attn_only` kết thúc ở loss
0,5377, thấp hơn hẳn `correct` (0,6249) — nếu chấm bằng loss thì nó thắng rõ ràng. Trên
thang đo tác vụ, hai run **hoà tuyệt đối**: cùng 194/200 quyết định trường đúng, cùng
`target = 0,9700`. Nghĩa là con số tôi nhìn suốt lúc train — thứ duy nhất hiển thị trên
thanh tiến trình — lại là con số dễ dẫn tôi tới kết luận sai nhất. Lý do thì hợp lý một khi
đã thấy: r = 283 trên 225 mẫu là thừa dung lượng để thuộc lòng tập train, và loss đo đúng
cái sự thuộc lòng đó. Điều làm tôi nhớ lâu là **hoà** chứ không phải **thua**: chỉ số thay
thế không chỉ yếu, nó còn tạo ra một thứ hạng ở nơi thực tế không có thứ hạng nào.

Ngạc nhiên thứ hai, nhỏ hơn nhưng đáng nhớ: **latency là một công cụ chẩn đoán, không chỉ là
chỉ số hiệu năng.** Baseline (a) chậm 2946 ms trong khi (b) chỉ 991 ms — cùng một model.
Model không hiểu phải làm gì thì *viết lan man tới trần token*; model hiểu việc thì trả lời
ngắn và dừng. Sau này khi thấy `wrong_lr` mất 4876 ms/mẫu, tôi đọc ra triệu chứng "không
biết dừng" trước cả khi nhìn điểm.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi dự đoán mất thời gian nhất ở **train**. Thực tế train là phần chạy trơn nhất: bốn run,
mỗi run ~13–16 phút, không phải can thiệp gì.

Thời gian thật đổ vào **hạ tầng và tính đúng đắn của phép đo** — những thứ không xuất hiện
trên bảng kết quả: phiên Colab rớt giữa chừng khiến NB3/NB4 phải chạy lại từ đầu nhiều lần,
và việc truy ngược một khiếm khuyết khiến *cả bốn* adapter đều ra `target = 0,000` dù đường
loss trông hoàn toàn khoẻ mạnh (F-31: dữ liệu train render prompt theo một dạng, còn eval
gửi đi một dạng khác — model trả lời đúng câu hỏi mà nó *thực sự* được hỏi). Mask đã đúng
ngay từ đầu, kiểm chứng đàng hoàng — nhưng "span nào được tính loss?" và "prompt nào sẽ
thực sự được gửi?" là **hai câu hỏi khác nhau**, và tôi chỉ có sẵn công cụ để trả lời câu
đầu.

Tỷ lệ thời gian, ước lượng: ~20% train, ~30% chạy lại vì môi trường, ~50% chẩn đoán và
kiểm chứng số đo. Đúng ra tôi phải đoán được điều này — nhưng không.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Ba niềm tin bị bác bỏ, xếp theo mức độ tôi từng tin chắc:

- **"Loss giảm nghĩa là đang tiến bộ."** Xem câu 1. Loss trả lời "model khớp dữ liệu train
  tới đâu", không trả lời "model làm đúng việc tới đâu". Với LoRA rank cao trên tập nhỏ,
  hai câu đó có thể tách rời hẳn nhau — và cột loss cũng hoàn toàn mù trước lỗi thực tế duy
  nhất mà model của tôi mắc phải (6 ticket chứa cụm `Khi nào tiện`).
- **"Fine-tune tốt hơn thì so với model gốc là đủ."** `format` của baseline (a) là **0,000**
  — model gốc với prompt ngây thơ không sinh nổi JSON hợp lệ *một lần nào* trong 50 lần. Chỉ
  cần đổi prompt, không train gì, `target` đã lên 0,7650. Nếu lấy (a) làm mốc, tôi đã báo
  cáo mức cải thiện 0,00 → 0,97 và **gần 4/5 quãng đường đó vốn miễn phí**. Mốc phải vượt là
  baseline đã được prompt tử tế, không phải baseline tệ nhất tìm được.
- **"Fine-tune là cộng thêm năng lực."** Không — ở đây nó là **đánh đổi**. `target` +0,205 đi
  kèm `regression` −0,080 (gấp 4 lần ngưỡng cho phép), và nguyên nhân nằm trọn trong dữ liệu:
  225/225 mẫu train là ticket → JSON, không có một mẫu phổ thông nào làm đối trọng, nên model
  học rằng *mọi* đầu vào đều có nghĩa là "xuất JSON triage".

Một điều tôi **vẫn** tin, và giờ có số để chống lưng: learning rate là biến quan trọng nhất.
Sai 10× kéo `target` từ 0,9700 xuống 0,0000 — và phá luôn cả `format`.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng vào ba việc: chẩn đoán khi số đo vô lý (F-31 — vì sao loss khoẻ mà điểm bằng 0), dựng
script kiểm chứng, và **soạn bản nháp report từ các file JSON đã đo**.

Chỗ nó sai rõ nhất là **§6 — phân tích định tính**. Bản nháp viết rằng các ca yếu thua vì
"ticket không nói thẳng ý định, phải suy ra `intent`" — nghe rất thuyết phục, vì những ticket
đó đúng là chỉ mô tả triệu chứng ("Chưa thấy tiền", "Thiếu phụ kiện"). Khi đối chiếu với nhãn
thật trong `data/eval_target.jsonl` thì **sai hẳn**: `intent` đúng **50/50**, kể cả các ca
đó. Lỗi thật nằm ở **`urgency`**, cùng một hướng ở cả sáu ca (`thap` bị đoán thành
`trung_binh`), và cụm gây lỗi là **"Khi nào tiện"** — model học hoàn hảo hai cụm đồng nghĩa
khác (`Không vội` 7/7, `Hỏi cho biết thôi` 5/5) nhưng trượt 0/6 ở cụm này, dù nó có **nhiều**
mẫu train nhất trong ba cụm (35 mẫu, tất cả đều gắn nhãn `thap`). Hai câu chuyện đều mạch
lạc; chỉ một câu khớp dữ liệu.

Bài học rút ra không phải "đừng dùng AI" mà cụ thể hơn: **AI rất giỏi dựng một lời giải
thích hợp lý cho con số, và lời giải thích hợp lý thì không kiểm chứng được bằng cách đọc
lại nó.** `qualitative.json` chỉ lưu điểm 0,75 chứ không lưu *trường nào* sai, nên cả tôi
lẫn nó đều đang suy diễn. Tôi phải mở file nhãn ra so từng trường thì mới biết. Quy tắc từ
nay: mọi câu trong report bắt đầu bằng "model yếu ở…" đều phải truy được về một dòng dữ liệu
cụ thể, nếu không thì xoá.

Sai lầm nhỏ hơn nhưng cùng bản chất: ở lượt smoke `EVAL_LIMIT=8`, nó viết chắc nịch về những
con số đo trên 8 mẫu — nơi bước nhảy tối thiểu của tập regression là 1/8 = 0,125, tức **cổng
hồi quy bị trượt bởi đúng một câu hỏi**. Phần cảnh báo đó là do tôi ép thêm vào, không phải
mặc định nó có. Lượt nộp chạy đủ 50 ticket và 15 câu regression chính là để câu trả lời
không còn phụ thuộc vào một mẫu lẻ.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Không** phải chọn model, không phải chọn rank. Bước đầu tiên là **dựng bộ đánh giá và
chốt baseline trước khi train một step nào** — cụ thể theo thứ tự:

1. **Viết tập eval trước tiên**, đủ lớn để bước nhảy tối thiểu `1/n` nhỏ hơn hẳn mức khác
   biệt mà khách hàng thật sự quan tâm. Lab này dạy tôi bài đó theo cách đau nhất ở lượt
   smoke: `n = 8` → bước nhảy 0,125, trong khi ngưỡng hồi quy là 0,020 — cổng bị trượt bởi
   đúng một câu hỏi, và tôi không có cách nào biết độ lớn thật. Kèm theo là tập hồi quy đo
   năng lực phổ thông — thứ mà không ai nghĩ tới cho đến khi nó mất.
2. **Đo baseline (a) *và* (b).** Cách rẻ nhất để biết mình có cần fine-tune hay không là
   thử prompt cho tử tế trước. Nếu prompt tối ưu đã đủ đạt yêu cầu, phần fine-tune là chi
   phí và rủi ro hồi quy không đổi lại gì. Chốt hash prompt để mốc không bị dịch về sau.
3. **Chứng minh phần được giám sát *và* dạng prompt trước khi train** — cả hai. Kiểm tra
   mask (loss có rơi đúng vào câu trả lời không) *và* kiểm tra dạng render lúc train có
   khớp dạng sẽ gửi lúc phục vụ không. Lab này có một chứng minh mask nghiêm chỉnh và vẫn
   mất nhiều giờ vì thiếu chứng minh thứ hai.
4. **Chốt tiêu chí "thế nào là đạt" bằng văn bản, cùng khách hàng, trước khi thấy kết quả** —
   gồm cả mức hồi quy được phép. Nếu không, áp lực nới ngưỡng cho tới khi gate xanh là gần
   như không cưỡng lại được. Báo cáo của tôi kết thúc bằng FAILED, và tôi giữ nguyên nó.

Chỉ sau bốn bước đó tôi mới đụng tới cấu hình — và ở đó, thứ tự ưu tiên đã rõ từ §4:
**learning rate trước** (sai 10× là mất trắng), rồi tới thành phần dữ liệu (replay 1–5%),
còn vị trí adapter và rank là tinh chỉnh biên — `attn_only` chứng minh điều đó bằng cách đạt
đúng cùng điểm với `correct`. QLoRA chỉ dùng khi fp16 **không vừa card**, chứ không dùng mặc
định để tiết kiệm: 41% VRAM nghe hấp dẫn, nhưng ở tier T4 nó chỉ đổi được 3 điểm target lấy
chỗ trống mà tôi không cần.
