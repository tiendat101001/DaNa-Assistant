# DaNa realtime upgrade: chỉ mục báo cáo (2026-11-05)

**Phạm vi yêu cầu:** đưa mô hình ASR/TTS LiteRT chạy trên máy để rút ngắn thời gian
phản hồi (mục tiêu: âm thanh đầu tiên < 0,6 s); một API registry mô hình duyệt và
tải từ HuggingFace `litert-community` chỉ với một yêu cầu; một giao diện quản lý
tất cả (CPU/WebGPU); LLM qua SGLang hoặc OpenAI/Gemini (cộng thêm Together AI, bổ
sung giữa phiên); computer use tổng quát thay cho các trang cố định; và kế hoạch
điện thoại.

## Tóm tắt điều hành

1. **Nguyên nhân chính làm chậm:** hai khâu chiếm phần lớn. TTS sinh bằng đám mây
   (Gemini TTS, ≥ 1 s đến byte đầu tiên) và LLM suy luận ẩn (Gemini mặc định bật
   thinking; mô hình Together cũng suy luận mặc định). Tổng thể từ lúc nói xong đến
   lúc có âm thanh đầu tiên khoảng 2–4 s.
2. **Giọng nói tự chạy trên máy bằng LiteRT đã hoạt động thật.** Đã kiểm thử trên
   các mô hình `litert-community` thật, không phải mô hình giả lập:
   - **ASR:** Moonshine-tiny (EN) mất **250–480 ms**; Whisper-tiny (EN+VI) mất
     khoảng 0,8–1,5 s. Im lặng chỉ tốn khoảng 1 ms nhờ cổng năng lượng.
   - **TTS:** KittenTTS-nano (EN) đạt tiếng nói đầu tiên trong **169–335 ms**. Trước
     đó khoảng 1 s, cho đến khi chúng tôi phát hiện và cắt bớt 0,8 s khoảng lặng đầu
     do mô hình tạo ra.
   - **TTS tiếng Việt:** dùng Piper (trong tổ chức chưa có mô hình TTS tiếng Việt
     chạy LiteRT).
3. **LLM đã được tinh chỉnh và đo trực tiếp.** Together DeepSeek-V4.1-Flash, tắt
   suy luận, có TTFT khi đã khởi động là khoảng **375 ms** (không gắn công cụ) và
   **431 ms** khi gắn 26 công cụ của DaNa (lần đầu nguội: 7,5 s, nên DaNa giờ tự khởi
   động LLM khi bắt đầu phiên). Một đợt quét 20 mô hình Together serverless khả dụng
   với khóa này xác nhận đây là mô hình nhanh nhất có tiếng Việt tốt. SGLang, vLLM,
   OpenAI, Gemini và Together đều có sẵn dưới dạng cấu hình một chạm, kèm nút đo độ
   trễ "Measure latency" trực tiếp.
4. **Tình trạng độ trễ thực tế:** bộ đo thành phần thật cho thấy **≈ 1,19 s** từ lúc
   nói đến lúc có âm thanh trả lời đầu tiên trên container 4 vCPU này (VAD 200 +
   Moonshine 423 + LLM 426 + Kitten 138), và **≈ 0,99 s** khi bật **speculative STT**
   mới (đo được giảm 240 ms). So với ≈ 2–4 s trước đây, đây là cải thiện 2–4 lần,
   nhưng câu trả lời chưa đạt < 0,6 s. Khi bật **phản hồi tức thì** (opt-in), người
   dùng nghe thấy DaNa sau khoảng **220 ms**. Phần còn lại khoảng 400 ms chủ yếu là
   độ trễ khứ hồi tối thiểu của LLM đám mây (khoảng 370–450 ms trên mọi mô hình
   Together serverless). Các đòn bẩy còn lại là LLM chạy GPU cục bộ qua SGLang (đã có
   cấu hình và script), một CPU máy bàn, và prefix caching. Xem `02-…` §5.
5. **Registry mô hình:** duyệt danh mục 443 repo, xem gợi ý tệp (lọc bỏ các biến thể
   NPU; mô hình nhiều giai đoạn được đầy đủ; thêm tệp tokenizer đi kèm), và **tải +
   nạp trong một yêu cầu**. Đã kiểm chứng trực tiếp trên huggingface.co (KittenTTS
   62,9 MB được tải và nạp trong khoảng 10 s).
6. **Computer use tổng quát:** `browser_snapshot` → `browser_act(ref)` hoạt động trên
   mọi trang web (đã kiểm chứng trên Chromium thật). Có công cụ Windows UI Automation
   cho mọi ứng dụng desktop, và một **cầu MCP** để cắm bất kỳ MCP server nào. Mọi
   thao tác vẫn đi qua cổng an toàn (xác nhận thao tác phá hủy, từ chối thao tác ngoài
   phạm vi, có ghi nhật ký).
7. **Giao diện:** tab 🧩 Models mới (hub, tải xuống, mô hình cục bộ, gán ASR/TTS, đặt
   LLM, ngân sách độ trễ, bộ chạy WebGPU trong trình duyệt) và manifest PWA cho điện
   thoại.
8. **Di động:** kế hoạch 3 giai đoạn. Giai đoạn 1 là client PWA điều khiển từ xa
   (≈ 20–25 người-ngày); giai đoạn 2 là ứng dụng native với LiteRT chạy trên thiết bị;
   giai đoạn 3 là độc lập.

## Tài liệu (đọc theo thứ tự này)

| # | Tệp | Nội dung |
|---|---|---|
| 1 | `01-architecture-deep-dive.md` | Từ khái niệm tổng quát đến cơ chế của từng thành phần, phép tính ngân sách độ trễ |
| 2 | `02-implementation-and-verification.md` | Những gì đã triển khai (tệp, API, cấu hình), bằng chứng kiểm thử, độ trễ đo được, kế hoạch thu hẹp khoảng cách, hướng dẫn vận hành |
| 3 | `litert-community-catalog.md` | Ảnh chụp trực tiếp 443 repo litert-community (toàn bộ ASR/TTS được liệt kê) |
| 4 | `00-team-contract.md` | Cơ cấu nhóm, phân quyền sở hữu và các hợp đồng REST nhóm đã dùng để phát triển |
| 5 | `lead-a-backend-status.md`, `lead-b-product-status.md` | Bảng công việc của trưởng nhóm kèm bằng chứng |
| 6 | `research-latency-llm-computer-use.md` | Nghiên cứu ngân sách độ trễ, SGLang, các chuẩn computer use |
| 7 | `research-litert-ecosystem.md` | Nghiên cứu runtime LiteRT / WebGPU (§1–3 đã được thay thế bởi catalog, xem ghi chú) |
| 8 | `ux-models-ui.md` | UX của tab Models: màn hình, luồng thao tác, trạng thái |
| 9 | `mobile-app-plan.md` | Các giai đoạn ứng dụng di động, hợp đồng WebSocket âm thanh, ràng buộc của cửa hàng ứng dụng |
| 10 | `business-case.md` | Chi phí cho mỗi người dùng nặng (đám mây so với lai so với tự lưu trữ), KPI, triển khai, các trường hợp dùng cho KAM |

## Việc anh/chị cần làm tiếp

1. Thu hồi (rotate) HF token đã dán trong chat; token chưa từng được ghi vào tệp nào trong repo.
2. Đo WER thật bằng giọng nói và micro của anh/chị, vì số liệu hiện tại dùng clip tổng hợp espeak.
3. Quyết định có chấp nhận độ trễ ≈ 1,0–1,2 s hay không, vì mục tiêu < 0,6 s chưa đạt đầu-cuối.
4. Bật `SPECULATIVE_STT` và `ACK_ENABLED` trong `.env` sau khi đo trên máy thật của anh/chị.
5. Tải tệp giọng Piper vi_VN, vì DaNa không đi kèm tệp này.
6. Quyết định đầu tư GPU NVIDIA (WSL2) để chạy LLM cục bộ qua SGLang, khoản tiết kiệm lớn nhất còn lại.
7. Thử nghiệm ASR tiếng Việt tốt hơn (Qwen3-ASR-0.6B hoặc STT đám mây chỉ cho tiếng Việt), vì Whisper-tiny chậm 1–1,5 s.
8. Kiểm thử các công cụ Windows UI Automation trên máy Windows thật; hiện mới chỉ kiểm thử bằng mô phỏng.
9. Phê duyệt Giai đoạn 1 của ứng dụng di động (≈ 20–25 người-ngày) nếu muốn tiếp tục.
10. Nếu cần chạy offline hoàn toàn: đưa wasm của MediaPipe vào `dana-ui/public/` và bổ sung service worker (hiện PWA chưa hoạt động offline).
