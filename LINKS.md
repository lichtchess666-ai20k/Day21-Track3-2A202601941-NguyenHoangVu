# Lab 21 — Links

**Họ tên**: Nguyễn Hoàng Vũ · **MSSV**: 2A202601941

| | |
|---|---|
| Repo code | https://github.com/lichtchess666-ai20k/Day21-Track3-2A202601941-NguyenHoangVu |
| LoRA adapter (HF Hub) | *(dán URL sau khi chạy `python push_to_hub.py <hf-username>`)* |
| Base model | https://huggingface.co/unsloth/Qwen3.5-4B |

## Nội dung nộp

* `submission/REPORT.md` — báo cáo 7 mục, mọi số khớp `results/`
* `submission/REFLECTION.md` — 5 câu phản tư
* `results/` — 9 artefact JSON/CSV mà grader dùng để đối chiếu
* `adapters/correct/` — adapter chính (cũng có trên HF Hub, link ở trên)

## Kết quả một dòng

Fine-tune đạt `target = 0,970` so với `0,765` của base đã prompt tử tế (+0,205), thắng
hoặc hoà trên cả 50/50 ticket — nhưng **trượt cổng hồi quy**: general capability tụt
0,758 → 0,678, gấp 4 lần ngưỡng 0,020. Phán quyết: **FAILED**, chưa deploy được.
