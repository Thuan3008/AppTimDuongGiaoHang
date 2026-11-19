# Hong Kong n=50 official multi-seed benchmark

Chạy từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --instance data\TDVRP\hong_kong\n=50\poryos-hong_kong-n50-hyb\bpr-heavy\poryos-hong_kong-n50-hyb-bpr-heavy.vrp.json `
  --num-seeds 10 `
  --population-size 10 `
  --budget 290 `
  --departure-mode optimize `
  --departure-step 900 `
  --output results\hong_kong_n50_multi_seed.json
```

Protocol áp dụng giống nhau cho `ARO-goc`, `POA-goc`, `RAPO` và
`TD-CR-RAPO-V3`: cùng instance, road graph OSM, traffic BPR-heavy, objective,
population, greedy seed policy, seed set, departure-time policy và strict NFE.

Hai sidecar chính thức được ghim SHA-256:

- `poryos-hong_kong-n50-hyb.road.json.gz`
- `poryos-hong_kong-n50-hyb.traffic-bpr-heavy.json.gz`

Runner xuất JSON đầy đủ, CSV từng run và CSV thống kê. Gap BKS chỉ được tính
khi dùng sidecar thật và ATF chính thức; chế độ proxy luôn để gap trống.

`compare_algorithms.py` cũ được giữ lại để tái hiện kết quả preliminary một
seed trên coordinate proxy, không dùng để đưa ra kết luận chính thức.
