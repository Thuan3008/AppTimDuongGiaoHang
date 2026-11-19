# RAPO-2709 for TDVRP

Thuật toán chính của dự án là **RAPO-2709**, một cơ chế lai rời rạc cho Time-Dependent Vehicle Routing Problem. Objective được giữ cố định:

```text
total_route_duration = total_travel_time + total_service_time
```

RAPO-2709 dùng giant tour + Split động, ARO/POA/Elite, Elite AOS, global-best archive, strict NFE ledger và tối ưu thời điểm xuất phát chính xác bằng NDCPWLF. `TD-CR-RAPO-V3` vẫn được giữ để đối chứng và tái lập thực nghiệm; `MS-IRAPO` là mô-đun legacy, không còn là thuật toán mặc định của app hay ExperimentRunner.

Kết quả paired benchmark Hong Kong n=25 HYB/BPR-heavy với 20 seed, population 10 và NFE 290: RAPO-2709 thắng 20/20 seed, fitness trung bình `21939.62` so với `24277.99` của TD-CR-RAPO-V3; Wilcoxon exact hai phía `p = 1.907e-06`. Đây là bằng chứng trên cấu hình n=25 nói trên, không phải kết luận tuyệt đối cho mọi instance.

Phạm vi hiện tại là TDVRP: 1 depot, nhiều khách hàng, nhiều xe, ràng buộc sức chứa, thời gian di chuyển phụ thuộc thời điểm bằng traffic simulation và trực quan hóa XY. **TDVRPTW / time windows chưa thuộc phiên bản này**; không có ready time, due time, waiting hay lateness penalty.

## Chạy trên Windows

Python global không bị thay đổi. Tạo môi trường ảo:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Chạy test:

```powershell
.venv\Scripts\python.exe -m pytest
```

## Kiến trúc hiện tại

Luồng chính: nạp instance + sidecar Poryos chính thức → kiểm tra checksum → materialize ATF → khởi tạo giant tour → Split động → ARO/POA/Elite → exact departure-time evaluator → global-best archive → route trace và export.

- `core/`: mô hình, distance matrix, simulated travel time, evaluator, fitness và validation.
- `algorithms/rapo_2709.py`: thuật toán chính và exact Split evaluator.
- `algorithms/td_cr_rapo_v3.py`: thuật toán đối chứng.
- `experiments/`: runner nhiều seed, checkpoint/resume, log terminal và báo cáo thống kê.
- `data/`: benchmark Poryos Hong Kong cùng road graph và traffic sidecar.
- `tests/`: kiểm thử traffic, validation, repair, population, seed và optimizer.

App mặc định mở Hong Kong n=25 HYB/BPR-heavy và cho phép chọn n=50. Proxy graph bị tắt trong app; nếu thiếu hoặc sai checksum sidecar, app báo lỗi thay vì âm thầm dùng dữ liệu thay thế.

## HCMC TDVRP Dataset

Repository cũng chứa một data layer TDVRP độc lập cho khu vực nghiên cứu quanh Tân Bình, Tân Phú, Phú Nhuận, Quận 10, Quận 11 và một phần Quận 3:

```text
data/hcmc/
  instances/hcmc-n100-heavy.vrp.json
  sidecars/hcmc.road.json
  sidecars/hcmc.traffic-heavy.json
  benchmarks/
```

Instance, road graph và traffic overlay được tách riêng. Traffic là `fixed_hourly_speed`, mô phỏng và cố định theo seed; đây không phải dữ liệu giao thông thực tế của TP.HCM. Road names chỉ nhằm minh họa. Mục tiêu là so sánh thuật toán reproducible với objective `MINIMIZE TOTAL ROUTE DURATION`, trong đó `route_duration = travel_time + service_time`; không có time window/lateness.

Engine mới nằm ở `core/data/` và `core/routing/`: loader kiểm tra sidecar, `FixedTrafficProvider` cung cấp tốc độ theo edge/giờ, còn `find_fastest_path` dùng time-dependent Dijkstra và cập nhật thời điểm sau mỗi edge. Cache dùng khóa `(source, target, departure_time_bucket)` với bucket mặc định 5 phút.

`HCMCTDVRPEvaluator` trong `core/data/evaluator.py` nhận `Solution.routes`, mô phỏng từng xe độc lập từ depot, gọi fastest path cho từng chặng với clock hiện tại, cộng service time sau mỗi customer, kiểm tra capacity/route validity và trả `total_travel_time`, `total_service_time`, `total_route_duration`, `total_distance`, `fitness` cùng timeline từng route. Đơn vị là giây và mét; fitness mặc định là total route duration.

Greedy baseline nằm ở `algorithms/greedy.py`: tại mỗi bước, nó chọn customer chưa phục vụ có fastest travel time nhỏ nhất trong nhóm còn đủ capacity, sau đó đánh giá nghiệm bằng chính `HCMCTDVRPEvaluator`.

Các operator rời rạc nằm ở `algorithms/operators/discrete.py` gồm swap, insertion, relocate, reverse/2-opt và cross-route swap. Mọi candidate đi qua pipeline `repair -> validate -> evaluate` trong `hcmc_pipeline.py`.

Hai bản tối ưu độc lập được thêm dựa trên hai bài báo cung cấp: `algorithms/hcmc_aro_goc.py` giữ ý tưởng ARO về energy shrink, detour foraging khi energy cao và random hiding khi energy thấp; `algorithms/hcmc_poa_goc.py` giữ hai pha POA là predator attack tới nghiệm tốt hơn và defense/escape bằng local move. Vì route TDVRP là biến rời rạc, các phép cập nhật vector liên tục trong bài báo được ánh xạ thành các route operators và luôn được repair/validate/evaluate. Đây là discrete adaptation phục vụ HCMC TDVRP, không phải chép nguyên công thức continuous benchmark của bài báo.

`algorithms/hcmc_aro_c.py` là biến thể ARO random-key bám sát đoạn mã continuous được cung cấp: vector liên tục có một key cho mỗi customer, thứ tự key được giải mã thành permutation rồi repair thành các route có capacity. Các công thức energy, running mask, detour foraging và random hiding được giữ lại trước bước decode/evaluate.

`algorithms/hcmc_poa_c.py` là biến thể POA theo bản cung cấp: predator attack dùng path-relinking tới pufferfish tốt hơn, defense dùng đảo một đoạn route với bán kính giảm dần, và mỗi chu kỳ ghi nhận đúng 2 NFE (`pf_attack` + `pf_defense`).

`algorithms/hcmc_rapo.py` là RAPO độc lập cho HCMC: controller thích nghi chọn họ ARO/POA, có elite refinement, partial restart và giới hạn evaluation budget; mọi candidate đều đi qua repair → validate → evaluate với objective `total_route_duration`.

`algorithms/td_cr_rapo.py` là TD-CR-RAPO: dùng permutation + split decoder, Elite A/B, TD-criticality không phát sinh NFE, ARO exploration, POA destroy/regret-2 reconstruction, Elite AOS và strict branch-level NFE ledger.

`algorithms/td_cr_rapo_v3.py` là bản port đầy đủ hơn theo mã gốc V3: Dual Elite Memory dương/âm, split-aware block displacement, K-candidate pre-screening, recovery cap, AOS và architecture guards cho tỷ lệ NFE.

Sinh lại dữ liệu:

```powershell
.venv\Scripts\python.exe scripts\generate_hcmc_instance.py --customers 100 --seed 42 --traffic heavy
.venv\Scripts\python.exe scripts\generate_hcmc_instance.py --customers 1000 --seed 42 --traffic heavy
```

Chạy app:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

App bản đồ Flask có hai quyền `admin` và `driver`:

```powershell
.venv\Scripts\python.exe -m web_map.app
```

Tài khoản local được tạo ở lần chạy đầu:

```text
admin  / admin123
driver / driver123
```

Admin được tạo tài khoản Driver và chạy tối ưu. Mỗi route/xe trong kết quả được
tách thành một chuyến riêng để Admin gán cho một Driver. Mỗi customer trên route
được lưu thành một điểm giao; Driver bắt đầu chuyến rồi bấm hoàn thành từng điểm
theo thứ tự. Chuyến tự chuyển `assigned → in_progress → completed` khi điểm cuối
cùng được giao xong. Driver không thể xem route của xe khác.

Khi tạo chuyến, Admin có thể dùng toàn bộ instance của dataset hoặc bật
`Tự nhập kho và các điểm giao`. Chế độ tự nhập nhận tọa độ kho, số xe, tải trọng
và từng điểm giao (tọa độ, demand, service time, time window tùy chọn). Các điểm
được ghép vào nút gần nhất của road graph HCMC đã chọn trước khi chạy RAPO.
Kho và điểm giao cũng có thể nhập bằng địa chỉ; app chỉ gọi Nominatim khi Admin
bấm tìm/xem điểm, cache kết quả, rồi điền tọa độ tự động. Ô tọa độ được giữ trong
phần nâng cao để chỉnh thủ công khi cần.
Khi triển khai ngoài máy local, đặt `RAPO_SECRET_KEY`, `RAPO_ADMIN_PASSWORD`
và `RAPO_DRIVER_PASSWORD` trong environment trước lần khởi tạo database.

Giới hạn của riêng bộ HCMC: graph là mạng lưới mô phỏng cố định, chưa dùng OSM/GPS/live API; precompute là tiện ích tùy chọn và chưa được chạy mặc định cho n=1000.

### Benchmark và ablation HCMC

ARO-goc, POA-goc và RAPO dùng chung `HCMCTDVRPEvaluator`, objective
`total_route_duration = total_travel_time + total_service_time`, cùng giờ xuất
phát và cùng strict NFE budget. Chạy so sánh bằng:

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --instance data\hcmc\instances\hcmc-n10-heavy.vrp.json `
  --num-seeds 20 --population-size 10 --budget 290 `
  --start-time 28800 --departure-mode fixed `
  --algorithms ARO-goc POA-goc RAPO `
  --output results\hcmc_algorithms.json
```

Các biến thể ablation hiện có là `RAPO-no-adaptive`, `RAPO-no-elite` và
`RAPO-no-restart`. Có thể chạy cùng runner:

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --instance data\hcmc\instances\hcmc-n10-heavy.vrp.json `
  --num-seeds 20 --population-size 10 --budget 290 `
  --start-time 28800 --departure-mode fixed `
  --algorithms RAPO RAPO-no-adaptive RAPO-no-elite RAPO-no-restart `
  --output results\hcmc_rapo_ablation.json
```

Report lưu cả `node_path`, `edge_path` và trace từng leg để kiểm tra tuyến có
đổi theo traffic hay không. `MS-IRAPO` hiện được giữ làm mã legacy và không
tham gia benchmark HCMC road-graph.

## Poryos2026 chính thức và benchmark nhiều seed

`core/data/load_poryos.py` phân biệt rõ hai chế độ:

- `load_poryos()`: đọc road graph và traffic sidecar chính thức, giải nén `.json.gz`, kiểm tra schema và SHA-256, ánh xạ `node_vertices`, giữ cạnh có hướng và materialize arrival-time functions bằng `mamut-routing-lib`.
- `load_poryos_proxy()`: chỉ dùng cho smoke test; kết quả luôn mang nhãn `proxy_graph=true` và không được tính gap với BKS chính thức.

Bộ Hong Kong n=50 HYB/BPR-heavy hiện có sidecar OSM chính thức trong `data/TDVRP/sidecars/`. Evaluator dùng đường free-flow được ghim theo chuẩn Poryos và ATF đã kiểm tra SHA-256, vì vậy tái hiện đúng tuyệt đối cost của BKS đi kèm. Mỗi route trả về trace theo leg và stop gồm arrival, service start, departure, waiting, load, node path và edge path.

Chạy benchmark nhiều seed:

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --num-seeds 20 `
  --population-size 10 `
  --budget 290 `
  --algorithms RAPO-2709 `
  --output results\rapo_2709_multi_seed.json
```

Runner mặc định chạy RAPO-2709 với exact NDCPWLF và strict NFE. Kết quả gồm từng run, trạng thái lỗi, best/mean/median/standard deviation, runtime, feasible rate, gap BKS và route trace của run tốt nhất.

Các baseline đã tích hợp vào runner cùng bộ thực nghiệm gồm:

```text
GA, GA-LS, DPSO, ILS-VND, PSO-VNS
```

Khi chạy các thuật toán này cùng RAPO-2709, runner tạo P0 chung theo từng seed, dùng cùng objective exact, cùng chính sách unlimited nonempty routes và kiểm tra strict NFE. Source baseline được giữ trong các module tương ứng; `algorithms/poryos_baseline_adapter.py` chuyển kết quả về route trace chuẩn của project hiện tại.

Ví dụ chạy một seed để kiểm tra toàn bộ nhóm:

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --instance data\TDVRP\hong_kong\n=25\poryos-hong_kong-n25-hyb\bpr-heavy\poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json `
  --num-seeds 20 `
  --population-size 10 `
  --budget 290 `
  --algorithms RAPO-2709 GA GA-LS DPSO ILS-VND PSO-VNS `
  --log experiments\rapo_2709_vs_baselines_n25_progress.log `
  --output results\rapo_2709_vs_baselines_n25.json
```

CLI in tiến độ sau mỗi run theo dạng `[17/120] GA seed=2 | completed` và tự lưu cùng nội dung vào file log.

Runner tự tạo thêm checkpoint JSONL cùng tên output. Nếu chạy lại đúng lệnh, các cặp `(algorithm, seed)` đã hoàn tất sẽ được bỏ qua và chương trình tiếp tục từ run chưa có checkpoint. Dùng `--no-resume` nếu muốn xóa logic tiếp tục và chạy lại toàn bộ theo protocol hiện tại. Khi report hoàn tất, runner tự tạo thư mục PNG gồm fitness best/mean, gap BKS, runtime, fitness theo seed và convergence trung bình.

So sánh RAPO-2709 với TD-CR-RAPO-V3 phải dùng runner paired để bảo đảm cùng quần thể ban đầu, cùng exact evaluator và có checkpoint/resume:

```powershell
.venv\Scripts\python.exe experiments\compare_td_cr_rapo_v3_vs_rapo_2709.py `
  --instance data\TDVRP\hong_kong\n=25\poryos-hong_kong-n25-hyb\bpr-heavy\poryos-hong_kong-n25-hyb-bpr-heavy.vrp.json `
  --num-seeds 20 --population-size 10 --budget 290
```


```

Chạy các thuật toán học máy:
MLP : 
1. Chạy giải có lưu checkpoint và ghi log:
powershell
python algorithms/td_mlp.py solve `
  --instance "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy" `
  --model-path "artifacts/td_mlp_model.pt" `
  --top-k 5 `
  --checkpoint-file "artifacts/solve_checkpoint_n50_heavy.json" `
  --output-json "artifacts/results_n50_heavy.json" `
  --log-file "run.log"

2. Nếu bị ngắt giữa chừng, chạy tiếp từ checkpoint:
powershell
python algorithms/td_mlp.py solve `
  --instance "data/TDVRP/hong_kong/n=50/poryos-hong_kong-n50-hyb/bpr-heavy" `
  --model-path "artifacts/td_mlp_model.pt" `
  --checkpoint-file "artifacts/solve_checkpoint_n50_heavy.json" `
  --resume `
  --log-file "run.log"

3. Chạy huấn luyện có lưu & phục hồi checkpoint:
powershell
# Chạy train (lưu tiến độ vào train_checkpoint.pt)
python algorithms/td_mlp.py train `
  --dataset "artifacts/td_mlp_dataset.json" `
  --model-path "artifacts/td_mlp_model.pt" `
  --checkpoint-file "artifacts/train_checkpoint.pt" `
  --epochs 100
# Chạy tiếp nếu bị ngắt:
python algorithms/td_mlp.py train `
  --dataset "artifacts/td_mlp_dataset.json" `
  --model-path "artifacts/td_mlp_model.pt" `
  --checkpoint-file "artifacts/train_checkpoint.pt" `
  --resume `
  --epochs 100
```