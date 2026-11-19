# Bàn giao mã nguồn và dữ liệu khóa luận CNTT-KLCN135

Ứng dụng tối ưu hóa lộ trình giao hàng phụ thuộc thời gian bằng TD-ERAPO. Giao diện chính là Web Map, với dữ liệu minh họa tại Hong Kong và TP. Hồ Chí Minh.

## Cấu trúc thư mục

| Vị trí | Nội dung |
| --- | --- |
| `algorithms/`, `core/`, `utils/`, `visualization/` | Thuật toán, bộ đánh giá và các mô-đun hỗ trợ. |
| `web_map/` | Ứng dụng Flask và giao diện bản đồ cho quản trị viên, tài xế. |
| `data/` | Bộ dữ liệu Hong Kong Poryos2026 và HCMC, kèm dữ liệu mạng đường và giao thông cần để chạy. |
| `experiments/`, `scripts/` | Kịch bản thực nghiệm, tổng hợp kết quả và xử lý dữ liệu. |
| `results/` | Kết quả ablation n25, so sánh giới hạn 120 giây, Wave và các đối chứng bổ sung có trong project. |
| `tests/` | Kiểm thử thuật toán, dữ liệu và Web Map. |
| `docs/` | Tài liệu kỹ thuật của project. |

## Chạy Web Map trên Windows

Dùng Python 3.11. Mở PowerShell tại thư mục chứa README này rồi chạy các lệnh sau.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r web_map\requirements.txt
.\.venv\Scripts\python.exe -m web_map.app
```

Mở [Web Map](http://127.0.0.1:8000). Tài khoản demo mặc định là `admin / admin123` và `driver / driver123`. Ứng dụng tạo cơ sở dữ liệu mới ở lần khởi động đầu tiên. Nền bản đồ và dịch vụ vẽ đường trực tuyến cần kết nối Internet. Giao diện Streamlit riêng có thể mở bằng `python -m streamlit run app.py` trong môi trường đã cài thư viện.

## Tra cứu kết quả

- Đối sánh nhiều seed trên Hong Kong nằm ở `experiments/rapo_contribution/results/`. Giữ các thư mục chạy từng nhóm seed và checkpoint để đối chiếu. Nhánh n100 có kết quả từng phần; xem manifest tương ứng trước khi tổng hợp.
- Ablation n25 nằm ở `results/rapo_ablation_n25_merged/`. Hai thư mục `rapo_ablation_n25_seed09` và `rapo_ablation_n25_seed1029` chứa kết quả gốc.
- So sánh theo giới hạn 120 giây nằm ở `results/time_comparison_n25_120s_seeds0_29/`, gồm bảng tổng hợp, kết quả từng lượt, checkpoint, nhật ký và biểu đồ.
- Khảo sát độ nhạy (ver33, Bảng 4.10–4.12) nằm ở [results/sensitivity_n10_n25/](results/sensitivity_n10_n25/). Thư mục có dữ liệu 24 lượt chạy, bảng tổng hợp sáu cấu hình và biểu đồ.
- Đối chứng TD-MLP chế độ `heuristic_only` (Bảng 4.15) có [bản ghi JSON seed 0](results/td_mlp/hong_kong_n50_heuristic_only_seed0.json). Tên thuật toán trong JSON gốc là `TD-Insertion-Heuristic`. Trường `mode` ghi `heuristic_only`.
- Thực nghiệm HCMC (Bảng 5.8) có [CSV tổng hợp và ghi chú nguồn](results/hcmc_n10_n100/). Thư mục kèm trích xuất chỉ số của một lượt n100 seed 6 từ Web Map. Chưa tìm thấy đủ bản ghi từng seed để tái tính thống kê của Bảng 5.8 hoặc kiểm tra toàn bộ phép đối chiếu ghép cặp n100.

- Dữ liệu đầu vào giữ nguyên cấu trúc đường dẫn của project. Bản bàn giao dùng sidecar HCMC đi kèm từng instance và bỏ bản trích OSM thô, môi trường ảo, cache, dữ liệu phiên demo và bản thảo Word.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r web_map\requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Có thể xem tham số chạy bằng `python -m experiments.rapo_contribution.run --help` và `python -m experiments.run_time_comparison --help` trong môi trường đã cài thư viện. Các kịch bản vẽ biểu đồ hoặc huấn luyện mô hình có thể cần thư viện tùy theo phương pháp như Matplotlib, SciPy, PyArrow, PyTorch hoặc XGBoost. Xem [tài liệu kỹ thuật](docs/TAI_LIEU_KY_THUAT.md) để tra cứu thêm.


## How to run
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r web_map\requirements.txt
.\.venv\Scripts\python.exe -m web_map.app
npm run --prefix frontend dev

Sau khi chạy thì truy cập link : http://localhost:5173/


