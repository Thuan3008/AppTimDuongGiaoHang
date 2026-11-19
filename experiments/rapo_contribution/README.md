# Thực nghiệm chứng minh đóng góp của RAPO

Thư mục này chứa protocol chính thức để so sánh `RAPO-2709` với Greedy, ARO nguyên bản, POA nguyên bản, GA và GA-LS trên dữ liệu Hong Kong.

## Cấu trúc

```text
rapo_contribution/
├── config.json          # Dataset, thuật toán, P=10, NFE=290 và seed 0..29
├── run.py               # Chạy thí nghiệm, resume, ghi log và tổng hợp kết quả
├── exact_greedy.py      # Greedy xác định dùng exact route evaluator
├── analyze.py           # Mean, median, độ lệch chuẩn và so sánh paired
├── README.md            # Hướng dẫn và quy ước khoa học
└── results/             # Checkpoint, kết quả thô, CSV, JSON và biểu đồ
```

## Protocol cố định

- Dataset chính: Hong Kong HYB BPR-heavy, các kích thước n=25, n=50 và n=100.
- Thuật toán ngẫu nhiên: `ARO-Original`, `POA-Original`, `RAPO-2709`, `GA`, `GA-LS`.
- Population size: 10.
- NFE: 290, bao gồm đánh giá quần thể ban đầu.
- Seed: 0 đến 29.
- Với cùng dataset và seed, mọi thuật toán ngẫu nhiên nhận cùng quần thể khởi tạo.
- Mọi thuật toán dùng cùng exact minimum-duration Split; RAPO-2709 được chạy với `exact_split=True` thay cho Split proxy time-zero của app.
- Objective: tổng thời lượng các tuyến, bao gồm travel time và service time.
- Departure policy: exact NDCPWLF, tối ưu giờ xuất phát trong cùng miền benchmark `[0, 86400]`.
- Không dùng proxy graph và có kiểm tra checksum sidecar.

Tên đúng trong mã nguồn là `RAPO-2709`; cụm `rapo_2907` được hiểu là lỗi gõ tên.

## Ngoại lệ bắt buộc đối với Greedy

Greedy là thuật toán xác định, không có quần thể, seed hay ngân sách NFE. Vì vậy Greedy được chạy đúng một lần trên mỗi dataset bằng cùng exact route evaluator. Không sao chép kết quả Greedy thành 30 dòng vì thao tác đó tạo độ lệch chuẩn bằng 0 giả tạo và không cung cấp thêm bằng chứng thống kê.

## Chạy toàn bộ

Từ thư mục gốc project:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run
```

Chạy riêng một kích thước:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n25
```

Tiếp tục lần chạy bị gián đoạn:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n25
```

## Chạy 30 seed trên ba máy

Ba máy phải dùng cùng phiên bản source code và cùng bộ dữ liệu. Mỗi máy chạy một
shard riêng; checkpoint của shard nằm ngay trong thư mục kết quả của shard đó.

Máy 1:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n50 --seeds 0-9 --nfe 580
```

Máy 2:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n50 --seeds 10-19 --nfe 580
```

Máy 3:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n50 --seeds 20-29 --nfe 580
```

Các thư mục tương ứng là `results/n50/nfe_580/shards/seeds_00_09`,
`seeds_10_19` và `seeds_20_29`. Sau khi chạy xong, chép hai thư mục shard từ
máy 2 và máy 3 vào `results/n50/nfe_580/shards/` của máy 1 rồi chạy:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.merge --dataset n50 --nfe 580
```

Lệnh merge kiểm tra đủ seed 0..29, không trùng seed, cùng protocol, đủ 150 run
hoàn thành, sau đó sinh báo cáo tổng tại `results/n50/`. Không ghép file JSONL
bằng tay.

Chạy lại từ đầu:

```powershell
.\.venv\Scripts\python.exe -m experiments.rapo_contribution.run --datasets n25 --no-resume
```

## Kết quả

Mỗi dataset sinh:

- `stochastic_report.json`: toàn bộ 30 seed của năm thuật toán ngẫu nhiên.
- `stochastic_report_runs.csv`: từng lần chạy.
- `stochastic_report_summary.csv`: best, mean, median, standard deviation và gap.
- `greedy.json`: kết quả Greedy duy nhất.
- `summary.csv`: bảng chung sáu thuật toán.
- `paired_comparisons.csv`: RAPO so với từng baseline trên cùng seed.
- `checkpoint.jsonl`: dùng để resume.
- `run.log`: tiến độ và lỗi của từng run.

`paired_comparisons.csv` dùng hiệu `baseline - RAPO`; giá trị dương nghĩa là RAPO tốt hơn. Với các baseline ngẫu nhiên, file cũng báo win/tie/loss và kiểm định dấu hai phía. Greedy chỉ là mốc xác định nên không báo p-value.

