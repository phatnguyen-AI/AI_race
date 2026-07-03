# Bài 2 - Ontological Reasoning in Medical Knowledge Retrieval

Hệ thống AI xử lý văn bản y khoa tự do (ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm, hồ sơ EHR) để phát hiện, chuẩn hóa các khái niệm y tế và suy luận mối liên hệ ngữ cảnh.

---

## 1. Tổng quan Lịch trình & Lộ trình Cuộc thi

| Phase | Tên vòng | Thời gian | Định dạng nộp bài | Tài nguyên hỗ trợ |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Vòng 1 - Sơ loại | 02/07/2026 → 30/07/2026 | Tệp ZIP | GPU |
| **Phase 2** | Vòng 2 - Sơ khảo | 17/08/2026 → 19/08/2026 | API endpoint | GPU |
| **Phase 3** | Vòng 3 - Chung kết | 09/09/2026 → 10/09/2026 | API endpoint | GPU |

---

## 2. Mô tả Bài toán

Mục tiêu là xây dựng một hệ thống AI sử dụng các giải pháp NLP, LLM hoặc Agentic AI có khả năng thực hiện đồng thời các nhiệm vụ sau:
1. **Xác định và chuẩn hóa khái niệm y tế chuyên môn** từ dữ liệu y khoa dạng văn bản tự do (free-form clinical text).
2. **Suy luận ontology (Ontological Reasoning)** trên dữ liệu y khoa nhằm xác định quan hệ/ngữ cảnh giữa các khái niệm y tế trong một ngữ cảnh nhất định.

Hệ thống AI được cung cấp các cơ sở tri thức y khoa là **ICD-10** và **RxNorm**.

### 2.1. Định dạng Đầu vào (Input)
- **Đầu vào**: Một đoạn văn bản y khoa dạng tự do (free-form text) như kết quả khám lâm sàng, giấy xuất viện, ghi chú của bác sĩ, kết quả chẩn đoán hình ảnh, kết quả xét nghiệm, hồ sơ sức khỏe điện tử (EHR),...
- **Đặc điểm**: Chứa thuật ngữ y khoa, từ viết tắt, thông tin bệnh nhân (đã được tổng hợp giả định - synthetic) và nhiều khái niệm xuất hiện đồng thời trong cùng một văn bản.
- **Ví dụ**: 
  > *"Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quan."*

### 2.2. Định dạng Đầu ra (Output)
Đầu ra là danh sách các khái niệm y tế được phát hiện dưới dạng một danh sách các dictionary chứa các trường sau:

- **`text`** (string): Cụm từ trong văn bản đầu vào được xác định là khái niệm y tế.
- **`position`** (list): Gồm 2 phần tử dạng số nguyên biểu diễn vị trí bắt đầu và kết thúc của cụm từ trong văn bản gốc (vị trí tính từ `0` đến `n - 1`, với `n` là độ dài văn bản tính theo ký tự).
- **`type`** (string): Loại khái niệm y tế, thuộc một trong các nhãn sau:
  - `TRIỆU_CHỨNG`: Tên triệu chứng bệnh nhân mắc phải.
  - `TÊN_XÉT_NGHIỆM`: Tên xét nghiệm bệnh nhân thực hiện.
  - `KẾT_QUẢ_XÉT_NGHIỆM`: Kết quả xét nghiệm, bao gồm giá trị và đơn vị.
  - `CHẨN_ĐOÁN`: Tên chẩn đoán của bác sĩ về bệnh mà bệnh nhân mắc phải.
  - `THUỐC`: Tên thuốc điều trị.
- **`assertions`** (list of strings): Các mối liên hệ/ngữ cảnh của khái niệm y khoa trong văn bản (chỉ giới hạn cho `CHẨN_ĐOÁN`, `THUỐC` và `TRIỆU_CHỨNG`). Danh sách có tối đa 3 phần tử gồm các chuỗi:
  - `"isNegated"`: Khái niệm bị phủ định trong văn bản (Ví dụ: *"không ho"*).
  - `"isFamily"`: Liên quan đến tình trạng bệnh lý của người thân/họ hàng (Ví dụ: *"bố bệnh nhân xuất hiện trường hợp đau bụng tương tự"*).
  - `"isHistorical"`: Liên quan đến tiền sử của bệnh nhân (Ví dụ: *"có tiền sử hen suyễn"*).
- **`candidates`** (list of strings): Danh sách mã định danh của chuẩn y tế tương ứng (chỉ áp dụng cho `CHẨN_ĐOÁN` và `THUỐC`):
  - Chuẩn **ICD-10** đối với bệnh (`CHẨN_ĐOÁN`).
  - Chuẩn **RxNorm** đối với thuốc (`THUỐC`).

---

## 3. Ví dụ Minh họa

### Văn bản Đầu vào (Input)
> *"Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"*

### Cấu trúc Kết quả Đầu ra (Output JSON)
```json
[
  {
    "text": "bệnh trào ngược dạ dày - thực quản",
    "position": [99, 133],
    "type": "CHẨN_ĐOÁN",
    "assertions": [],
    "candidates": ["K21.0", "K21.9"]
  },
  {
    "text": "ho đờm xanh",
    "position": [36, 47],
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "tức ngực",
    "position": [49, 57],
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "đau thượng vị",
    "position": [59, 72],
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "ợ hơi",
    "position": [74, 79],
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "Chlorpheniramine 0.4 MG/ML",
    "position": [168, 194],
    "type": "THUỐC",
    "assertions": ["isHistorical"],
    "candidates": ["360047"]
  },
  {
    "text": "Capsaicin 0.38 MG/ML",
    "position": [196, 216],
    "type": "THUỐC",
    "assertions": ["isHistorical"],
    "candidates": ["1660761"]
  },
  {
    "text": "WBC",
    "position": [287, 290],
    "type": "TÊN_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "14,43",
    "position": [291, 296],
    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "NEUT% (Tỷ lệ % bạch cầu trung tính)",
    "position": [298, 333],
    "type": "TÊN_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "76,4",
    "position": [334, 338],
    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "LYPH% (Tỷ lệ bạch cầu lympho)",
    "position": [340, 369],
    "type": "TÊN_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "12,8",
    "position": [370, 374],
    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  }
]
```

> [!NOTE]
> Các giá trị liên quan đến thông tin cá nhân (tên, tuổi, địa chỉ, sđt) đều là những giá trị synthetic, không phải các thông tin người thật.

---

## 4. Dữ liệu & Quy định Nộp bài

### 4.1. Cơ sở Dữ liệu Chuẩn
- Ánh xạ mã **ICD-10** cho các loại bệnh (`CHẨN_ĐOÁN`).
- Ánh xạ mã **RxNorm** cho các loại thuốc (`THUỐC`).

### 4.2. Dữ liệu Khảo sát (Test set)
- **Tập test**: Bao gồm **100 bản ghi**.
- Được lưu trong thư mục `input/input/` dưới dạng các file từ `1.txt` đến `100.txt`.
- Mỗi văn bản free-form text chứa nhiều hơn một khái niệm cần trích xuất.

### 4.3. Yêu cầu Đầu ra
- Với mỗi file `X.txt`, thí sinh cần tạo ra một file JSON kết quả `X.json` tương ứng.
- Cấu trúc file JSON là một list các dictionary của danh sách khái niệm y tế được phát hiện (chi tiết như ví dụ ở phần 3).

### 4.4. Nhiệm vụ Bổ sung
- Các thí sinh cần sử dụng các giải pháp bên ngoài (ví dụ: tạo thêm dữ liệu tổng hợp bằng LLM, sử dụng các tập dữ liệu mở, v.v.) để bổ sung nguồn dữ liệu huấn luyện cho mô hình.
