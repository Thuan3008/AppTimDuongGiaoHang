# BÁO CÁO PHÂN TÍCH ĐỘ NHẠY THAM SỐ THUẬT TOÁN RAPO (TDVRP)

- **Thời điểm phân tích:** 2026-11-07 16:03:33.084323
- **Thư mục kết quả:** `C:\KLCN\CODE\THUCNGHIEM\results\sensitivity\rapo\quick`
- **Tổng số lượt chạy hoàn thành:** 24
- **Tỷ lệ khả thi:** 100.00%
- **Số instance:** 2 | **Số seeds:** 2

---

## 1. NGUYÊN TẮC THỐNG KÊ VÀ LƯU Ý MẪU

> [!WARNING]
> **LƯU Ý QUAN TRỌNG VỀ QUY MÔ MẪU NHỎ (QUICK SENSITIVITY):**
> Thực nghiệm hiện tại được thực hiện trên tập mẫu thu hẹp gồm 2 instance và 2 seed. Do kích thước mẫu nhỏ ($N=2$), mọi kết quả so sánh và xu hướng trong báo cáo này chỉ mang tính chất **phân tích mô tả sơ bộ** (preliminary descriptive trends), **KHÔNG** cấu thành bằng chứng có ý nghĩa thống kê tổng quát (confirmatory statistical proof).
> Về mặt toán học, kiểm định phi tham số Wilcoxon Signed-Rank với $N=2$ cặp không thể đạt mức ý nghĩa $p \le 0.05$ (giá trị $p$ tối thiểu về mặt tổ hợp là $0.50$). Để có kết luận suy diễn thống kê đầy đủ, cần thực hiện trên pha Pilot ($N \ge 10$) hoặc Main ($N=40$ instances, 10-30 seeds).

1. **Trung bình hóa hai cấp:**
   - Cấp 1: Tính trung bình Gap to BKS (%) trên các seed trong từng instance.
   - Cấp 2: Lấy trung bình không trọng số của các giá trị trung bình instance trên toàn bộ tập dữ liệu.
2. **Tách biệt độ lệch chuẩn (SD):**
   - `sd_within_instance`: Độ lệch chuẩn trung bình giữa các seed trong cùng một instance (đo lường tính ổn định ngẫu nhiên).
   - `sd_between_instance`: Độ lệch chuẩn giữa các instance khác nhau ($N=2$ instance). Đây là độ phân tán mẫu giữa quy mô mạng lưới, **KHÔNG** phải khoảng tin cậy thống kê (confidence interval).
3. **Kiểm định thống kê ghép cặp:**
   - Sử dụng kiểm định phi tham số Wilcoxon Signed-Rank trên các cặp instance so với cấu hình Baseline.
   - Hiệu chỉnh p-value bằng quy tắc bước giảm Holm-Bonferroni cho từng họ tham số.
4. **Biểu đồ độ nhạy OFAT rời rạc:**
   - Các biểu đồ độ nhạy chỉ đánh giá hai mức (Baseline và OFAT level). Các đường nối đứt nét chỉ có mục đích định hướng trực quan, **không** diễn giải là quan hệ tuyến tính liên tục.

---

## 2. BẢNG TỔNG HỢP THEO THAM SỐ (SUMMARY BY PARAMETER)

| experiment_group   | condition             | level    |   num_instances |   total_runs |   feasible_rate_percent |   mean_gap_percent |   sd_between_instance_gap |   sd_within_instance_gap |   mean_objective |   sd_objective |   mean_runtime_seconds |   sd_runtime_seconds |   mean_nfe |
|:-------------------|:----------------------|:---------|----------------:|-------------:|------------------------:|-------------------:|--------------------------:|-------------------------:|-----------------:|---------------:|-----------------------:|---------------------:|-----------:|
| aos_learning_rate  | aos_alpha             | 0.5      |               2 |            4 |                     100 |            9.82633 |                  10.6413  |                 0.150795 |          18916.3 |        8808.42 |                56.5662 |              72.6857 |        100 |
| baseline           | baseline              | baseline |               2 |            4 |                     100 |            9.6367  |                  10.3731  |                 0.316658 |          18875.7 |        8750.96 |                78.5796 |              98.5149 |        100 |
| elite_quota        | elite_budget_fraction | 0.1      |               2 |            4 |                     100 |           10.9268  |                   9.42114 |                 2.03757  |          19063.5 |        8672.26 |                52.4556 |              65.8298 |        100 |
| nfe_budget         | max_evaluations       | 200      |               2 |            4 |                     100 |            7.30209 |                   8.69378 |                 1.26987  |          18427.2 |        8317.92 |                82.6307 |             103.605  |        200 |
| poa_destroy        | destroy_fraction      | 0.2      |               2 |            4 |                     100 |           10.6831  |                   9.08992 |                 1.55819  |          19011.7 |        8600.68 |                58.6485 |              70.6658 |        100 |
| pop_size           | population_size       | 20       |               2 |            4 |                     100 |           10.3838  |                  12.3498  |                 0.611782 |          19065.1 |        9132.99 |                74.1566 |              90.9254 |        100 |

---

## 3. PHÂN TÍCH RIÊNG THEO QUY MÔ BÀI TOÁN (N=10 VÀ N=25)

|   num_customers | experiment_group   | condition             | level    |   instances_count |   mean_gap_percent |   sd_between_instances |   mean_runtime_seconds |   mean_objective |
|----------------:|:-------------------|:----------------------|:---------|------------------:|-------------------:|-----------------------:|-----------------------:|-----------------:|
|              10 | aos_learning_rate  | aos_alpha             | 0.5      |                 1 |            2.30178 |                      0 |                5.16958 |          12687.8 |
|              10 | baseline           | baseline              | baseline |                 1 |            2.30178 |                      0 |                8.91906 |          12687.8 |
|              10 | elite_quota        | elite_budget_fraction | 0.1      |                 1 |            4.26501 |                      0 |                5.90691 |          12931.3 |
|              10 | nfe_budget         | max_evaluations       | 200      |                 1 |            1.15466 |                      0 |                9.37053 |          12545.5 |
|              10 | poa_destroy        | destroy_fraction      | 0.2      |                 1 |            4.25556 |                      0 |                8.68021 |          12930.1 |
|              10 | pop_size           | population_size       | 20       |                 1 |            1.6512  |                      0 |                9.86259 |          12607.1 |
|              25 | aos_learning_rate  | aos_alpha             | 0.5      |                 1 |           17.3509  |                      0 |              107.963   |          25144.8 |
|              25 | baseline           | baseline              | baseline |                 1 |           16.9716  |                      0 |              148.24    |          25063.5 |
|              25 | elite_quota        | elite_budget_fraction | 0.1      |                 1 |           17.5885  |                      0 |               99.0042  |          25195.7 |
|              25 | nfe_budget         | max_evaluations       | 200      |                 1 |           13.4495  |                      0 |              155.891   |          24308.8 |
|              25 | poa_destroy        | destroy_fraction      | 0.2      |                 1 |           17.1106  |                      0 |              108.617   |          25093.3 |
|              25 | pop_size           | population_size       | 20       |                 1 |           19.1165  |                      0 |              138.451   |          25523.1 |

---

## 4. SO SÁNH GHÉP CẶP VỚI BASELINE (BASELINE COMPARISONS)

| experiment_group   | condition             |   level |   paired_instances |   mean_gap_delta |   sd_gap_delta |   mean_runtime_delta |   wilcoxon_w |   p_value_raw |   p_value_adjusted_holm | statistical_conclusion                         |
|:-------------------|:----------------------|--------:|-------------------:|-----------------:|---------------:|---------------------:|-------------:|--------------:|------------------------:|:-----------------------------------------------|
| aos_learning_rate  | aos_alpha             |     0.5 |                  2 |         0.18963  |       0.268177 |            -22.0135  |          nan |             1 |                       1 | Không có khác biệt ý nghĩa thống kê (p > 0.05) |
| elite_quota        | elite_budget_fraction |     0.1 |                  2 |         1.29006  |       0.952005 |            -26.1241  |          nan |             1 |                       1 | Không có khác biệt ý nghĩa thống kê (p > 0.05) |
| nfe_budget         | max_evaluations       |   200   |                  2 |        -2.33461  |       1.67936  |              4.05103 |          nan |             1 |                       1 | Không có khác biệt ý nghĩa thống kê (p > 0.05) |
| poa_destroy        | destroy_fraction      |     0.2 |                  2 |         1.0464   |       1.28322  |            -19.9312  |          nan |             1 |                       1 | Không có khác biệt ý nghĩa thống kê (p > 0.05) |
| pop_size           | population_size       |    20   |                  2 |         0.747146 |       1.97668  |             -4.42308 |          nan |             1 |                       1 | Không có khác biệt ý nghĩa thống kê (p > 0.05) |

---

## 5. GIỚI HẠN VÀ KHUYẾN NGHỊ KHOA HỌC

1. **Tính thụ động của `criticality_floor`:**
   - Mã nguồn `rapo_2709.py` hiện tại chấp nhận tham số này trong hàm dựng nhưng chưa sử dụng trong logic tìm kiếm.
   - Biến thiên của tham số này không tạo ra sự khác biệt về chất lượng nghiệm trên phiên bản hiện hành.
2. **Ràng buộc sàn xác suất AOS (`aos_epsilon`):**
   - Với 4 toán tử Elite (`relocate`, `swap`, `two_opt`, `two_opt_star`), tổng sàn xác suất là $4 \times \epsilon$.
   - Miền hợp lệ bắt buộc $\epsilon \le 0.25$. Mức $\epsilon = 0.20$ để lại 20% trọng số thích nghi cho cải tiến nghiệm.
3. **Cơ chế tái chèn của POA:**
   - Phiên bản hiện hành sử dụng phép gỡ bỏ và tái chèn ngẫu nhiên trên giant tour; việc tăng `destroy_fraction` làm tăng độ đa dạng nhưng có thể làm gián đoạn các cụm lộ trình tốt.
4. **Tính thuần khiết của ma trận Population x NFE:**
   - Heatmap $P \times \text{NFE}$ được lọc nghiêm ngặt, chỉ chứa các lượt chạy có toàn bộ tham số nội bộ bằng Baseline.
   - Cấu hình $(P=20, \text{NFE}=200)$ được đánh dấu rõ là N/A vì quy trình QUICK chỉ thực nghiệm biến thiên đơn lẻ (OFAT) mà không thực hiện tổ hợp chéo toàn phần.
5. **Phân tách thang đo hội tụ theo kích thước mạng lưới:**
   - Diễn tiến hội tụ của `hong_kong_n10` (~12.500s) và `hong_kong_n25` (~25.000s) được trình bày trên 2 đồ thị con độc lập với đường chuẩn BKS riêng, tránh méo mó trực quan khi gộp các mạng lưới có hàm mục tiêu khác xa nhau.

---

## 6. DANH MỤC BIỂU ĐỒ SINH RA

- `figures/mean_gap_by_parameter.png`: Biểu đồ Mean Gap (%) theo từng tham số (đường đứt nét 2 mức rời rạc).
- `figures/mean_gap_errorbars_by_parameter.png`: Biểu đồ Mean Gap $\pm$ SD giữa các instance (độ phân tán mẫu $N=2$, không phải CI).
- `figures/runtime_by_parameter.png`: Thời gian thực thi trung bình theo từng tham số.
- `figures/heatmap_pop_x_nfe.png`: Heatmap 2D ma trận Population $\times$ Max NFE thuần (ô P=20,NFE=200 là N/A).
- `figures/convergence_curves.png`: Diễn tiến hội tụ Best-so-far theo NFE phân tách riêng cho từng instance (Seed 0) cùng chuẩn BKS.