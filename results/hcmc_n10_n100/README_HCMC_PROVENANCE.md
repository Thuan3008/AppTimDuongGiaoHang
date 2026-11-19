# BIÊN BẢN XÁC MINH VÀ NGUỒN DỮ LIỆU KẾT QUẢ HCMC (MỤC 5.4.3, BẢNG 5.8)

## 1. Thông số thực nghiệm theo Báo cáo Khóa luận
Căn cứ Mục 5.4.3 và Bảng 5.8 trong báo cáo chính thức CNTT-KLCN135_BaoCao.docx (108 trang):
- Trường hợp 10 khách hàng (hcmc-n10-heavy):
  - 10 seed (0-9), Quần thể = 10, NFE = 290/290, 10/10 khả thi.
  - Duration tốt nhất = 3.968,07 giây (tại seed 8).
  - Duration trung bình = 4.133,54 giây, Độ lệch chuẩn = 97,96 giây.
  - Thời gian thực thi trung bình = 16,38 giây. Số tuyến = 1.
  - Chính sách giờ xuất phát: Cố định 08:00.
- Trường hợp 100 khách hàng (hcmc-n100-heavy):
  - 10 seed (0-9), Quần thể = 10, NFE = 290/290, 10/10 khả thi.
  - Duration tốt nhất = 23.918,62 giây (tại seed 6).
  - Duration trung bình = 24.595,04 giây, Độ lệch chuẩn = 483,39 giây.
  - Thời gian thực thi trung bình = 35,03 giây. Số tuyến = 6.
  - Chính sách giờ xuất phát: Tối ưu thời điểm xuất phát (exact_ndcpwlf) từ mốc 08:00.

## 2. Ghi nhận tình trạng bằng chứng gốc
- Bảng tổng hợp Bảng 5.8 được đối chiếu nguyên văn từ báo cáo chính thức của nhóm.
- Nhật ký chi tiết từng seed của 20 lượt chạy HCMC không được lưu trữ thành file JSON riêng rẽ trong mã nguồn gốc mà được lưu trữ dưới dạng số liệu tổng hợp trong bảng báo cáo chính thức (được nêu rõ tại đoạn văn P839 của báo cáo: 'Số liệu tổng hợp được cung cấp cũng chưa đủ để thực hiện kiểm định ghép cặp theo seed').
- Theo nguyên tắc liêm chính học thuật (Mục B.5 của nhiệm vụ), gói bàn giao giữ nguyên số liệu tổng hợp chính thức, KHÔNG tự ý sinh log giả lập theo seed.

## Đối chiếu với báo cáo ver33

CSV tổng hợp được sao từ thư mục bàn giao mẫu. Số liệu Duration và runtime trong CSV khớp các giá trị tổng hợp của Bảng 5.8. Dòng chính sách giờ xuất phát trong Bảng 5.8 ghi 08:00 cố định cho cả hai quy mô, trong khi phần diễn giải mục 5.4.3 mô tả bản HCMC-TD-ERAPO n100 dùng `exact_ndcpwlf` từ 08:00. CSV giữ cách ghi của hồ sơ mẫu.

Tệp `hcmc100_seed06_web_map_metrics.json` trích cấu hình và chỉ số từ một tác vụ Web Map ở project gốc. Lượt này có 290 NFE, 6 tuyến và Duration 23.918,62 giây, khớp giá trị tốt nhất n100 trong báo cáo. Runtime 19,83 giây trong tệp là số đo của lượt Web Map. Runtime 35,03 giây trong bảng là số trung bình được báo cáo. Một lượt khớp Duration chỉ xác nhận được giá trị của seed đó. Hiện chưa có đủ 20 bản ghi theo seed để tái tính bảng và kiểm chứng phép đối chiếu ghép cặp.
