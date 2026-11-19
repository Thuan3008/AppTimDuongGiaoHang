# HCMC observed traffic sidecars

Các file này bổ sung dữ liệu vận tốc quan sát trong `data/vantoctb` cho app bản đồ.

- `hcmc.alpha-observed.parquet`: alpha theo từng quan sát và `way_id` OSM. Công thức chính là `freeFlowSpeed / currentSpeed`, có thêm alpha theo travel time.
- `hcmc.road-observed.json`: road graph HCMC có thêm `osm_way_ids` trên các cung ghép được.
- `hcmc.traffic-observed.json`: profile 24 giờ theo giờ địa phương TP.HCM. Cung ghép được dùng median `currentSpeed`; cung chưa ghép dùng lại profile mô phỏng `hcmc.traffic-heavy.json`.
- `hcmc.osm-way-mapping.json`: báo cáo ghép OSM way với edge tổng hợp.

Hiện tại ghép được 130/2033 cung tổng hợp với ngưỡng hình học 500 m. Vì graph HCMC hiện tại là graph tổng hợp, không nên coi phần còn lại là dữ liệu GPS thực.

## Rebuild

```powershell
.venv\Scripts\python.exe scripts\build_hcmc_alpha_observed.py
.venv\Scripts\python.exe scripts\integrate_hcmc_observed_traffic.py
```

Web app tự ưu tiên các sidecar `*-observed.json` khi chạy dataset HCMC; nếu sidecar không tồn tại, app quay về file HCMC gốc.
