# Web bản đồ RAPO

Chạy từ thư mục `E:\1KLCN\tdvrp-rapo`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r web_map\requirements.txt
.\.venv\Scripts\python.exe -m web_map.app
```

Mở http://127.0.0.1:8000. Streamlit cũ vẫn ở `app.py`; bản web này ở
`web_map/app.py` và `web_map/index.html`.

- Lần đầu mở tự chạy RAPO-2709, Hong Kong n=25 HYB BPR-heavy, P=10, NFE=290, seed=42.
- Có thể chọn RAPO-2709, RAPO-Pufferfish và GA, chạy riêng hoặc cùng lượt. Các thuật toán dùng chung quần thể ngẫu nhiên ban đầu. GA được gọi từ `algorithms/ga.py` qua adapter hiện có, hỗ trợ bản đồ OSRM và xuất JSON/CSV.
- Giao diện báo giai đoạn hiện tại; chưa có phần trăm/NFE realtime trong lúc thuật toán đang tối ưu.
- Kết quả hoàn tất lưu trong `results/web_map/<id>.json`. Tải lại trang sẽ kết nối lại job hoặc đọc kết quả đã lưu.
- Đây là lưu kết quả, chưa phải checkpoint tiếp tục giữa một lần tối ưu nếu tắt backend.
- Tọa độ map lấy từ road graph sidecar; không dùng tọa độ cục bộ trong trường `coordinates`.
- OSRM Route giữ thứ tự depot → khách hàng → depot. GeoJSON dùng `[lon, lat]`; Leaflet marker dùng `[lat, lon]`.
- OSRM chỉ vẽ đường, không thay objective hoặc metric của evaluator. Mất OSRM dùng `node_path` của dataset, có nhãn nguồn rõ ràng; nếu thiếu trace mới dùng đường thẳng nét đứt.
- RAPO-2709 dùng departure exact. Adapter Pufferfish và GA hiện xuất trace bằng quét lưới departure. Chưa xem bảng trong app là benchmark với evaluator hoàn toàn đồng nhất.
- Internet cần cho Leaflet CDN, nền OpenStreetMap và OSRM. Request OSRM có timeout, cache trong bộ nhớ và giới hạn tần suất; lỗi đầu tiên chuyển các tuyến còn lại sang dataset để tránh chờ lâu.

API: `GET /api/datasets`, `GET /api/datasets/<id>`, `POST /api/runs`,
`GET /api/runs/<job_id>`. Mỗi lúc nhận một job. Server chỉ bind localhost.

Tài liệu: [OSRM Route API](https://project-osrm.org/docs/v5.5.1/api/),
[Leaflet](https://leafletjs.com/),
[quy định tile OpenStreetMap](https://operations.osmfoundation.org/policies/tiles/).
