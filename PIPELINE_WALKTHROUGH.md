# 🚀 Phân tích Luồng Dữ liệu (Pipeline Walkthrough) & Giải thích Kỹ thuật

Tài liệu này giải thích chi tiết **6 bước** trong quy trình xử lý thực tế (Inference Pipeline), lý do tại sao mỗi kỹ thuật lại được áp dụng, và minh họa bằng luồng chạy thực tế của file `8.txt`.

---

## 📄 Văn bản gốc (`8.txt`)

> **Bệnh sử**
> Bệnh nhân được chỉ định nhập viện để đánh giá và xử trí tình trạng kết quả chọc hút bất thường của nốt tuyến giáp thùy trái. Qua thăm khám và cận lâm sàng trước nhập viện, ghi nhận nốt tuyến giáp thùy trái kích thước khoảng 1 cm. Bệnh nhân đã được thực hiện thủ thuật chọc hút bằng kim nhỏ đối với tổn thương này. Kết quả xét nghiệm tế bào học từ dịch chọc hút ghi nhận bất thường.
> Tại thời điểm nhập viện, bệnh nhân **không ghi nhận** triệu chứng chèn ép vùng cổ như **khó nuốt, khó thở hoặc khàn tiếng**. 
>
> **Kết quả khám tại bệnh viện**
> Kết quả xét nghiệm: Kết quả tế bào học từ dịch chọc hút nốt tuyến giáp trái ghi nhận bất thường.
> Thủ thuật đã thực hiện: Chọc hút bằng kim nhỏ nốt tuyến giáp trái. Tổn thương ghi nhận: Nốt tuyến giáp thùy trái kích thước khoảng 1 cm.

---

## ⚙️ Quy trình 6 Bước & Lý do áp dụng kỹ thuật

### Bước 1: Tiền xử lý & Nhận diện Section (Fuzzy Matching)
*   **Lý do áp dụng:** Bác sĩ thường gõ sai chính tả tiêu đề hoặc gõ liền không xuống dòng. Nếu dùng Regex thông thường sẽ bỏ sót cấu trúc văn bản. Kỹ thuật "Khớp mờ" (Fuzzy Matching) giúp chia đoạn văn bản bất chấp lỗi đánh máy, tạo cơ sở ngữ cảnh cho Bước 4.
*   **Thực thi trên `8.txt`:**
    *   Hệ thống quét và cắt văn bản thành 2 Section: 
        1. `HIEN_TAI`: Bắt đầu từ chữ *"Bệnh sử"* đến hết đoạn 1.
        2. `DANH_GIA`: Bắt đầu từ chữ *"Kết quả khám tại bệnh viện"* đến hết bài.

### Bước 2: Quét Ngữ Cảnh Toàn Cục (Cross-file Entity Pool)
*   **Lý do áp dụng:** Chạy 100 file độc lập khiến AI "quên" mất các từ lóng đã gặp. Việc lập danh sách từ vựng chung (Entity Pool) giúp LLM tự tin hơn khi gặp các từ hiếm.
*   **Thực thi trên `8.txt`:** Hệ thống nhìn thấy *"nốt tuyến giáp"*, *"chọc hút bằng kim nhỏ"* đã từng xuất hiện ở file khác. Nó đưa các từ này vào trí nhớ tạm thời (Prompt) để chuẩn bị cho Bước 3.

### Bước 3: Trích xuất Đa Luồng (LLM + Constrained Decoding + Self-Consistency)
*   **Lý do áp dụng:** 
    *   *Constrained Decoding:* Ép LLM 7B sinh ra JSON hợp lệ ngay lập tức, tránh lỗi format (gây 0 điểm).
    *   *Self-Consistency:* Chạy LLM 3 lần và lấy "Bỏ phiếu đa số" để chống "ảo giác" (hallucination) — tức là LLM bịa ra bệnh không có thật.
*   **Thực thi trên `8.txt`:** Mô hình LLM 7B trả về JSON thô:
    ```json
    [
      {"text": "nốt tuyến giáp thùy trái", "type": "CHẨN_ĐOÁN", "assertions": []},
      {"text": "chọc hút bằng kim nhỏ", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
      {"text": "khó nuốt", "type": "TRIỆU_CHỨNG", "assertions": []},
      {"text": "khó thở", "type": "TRIỆU_CHỨNG", "assertions": []},
      {"text": "khàn tiếng", "type": "TRIỆU_CHỨNG", "assertions": []}
    ]
    ```
    *(Lưu ý: LLM 7B đã bỏ quên mất nhãn `isNegated` của 3 triệu chứng khó nuốt, khó thở, khàn tiếng).*

### Bước 4: Căn chỉnh Tọa Độ (Fuzzy Sliding Window)
*   **Lý do áp dụng:** LLM thường tự động chuẩn hóa văn bản (thêm bớt khoảng trắng). Nếu dùng hàm `.find()` thông thường sẽ không tìm thấy tọa độ. Kỹ thuật "Cửa sổ trượt khớp mờ" (Fuzzy Sliding Window) đảm bảo lấy lại đúng tọa độ `[start, end]` từ file gốc một cách chính xác tuyệt đối.
*   **Thực thi trên `8.txt`:** Hệ thống quét các chuỗi text vào bản gốc và lấy được tọa độ.
    *   `"khó nuốt"` -> `[574, 582]`
    *   `"khó thở"` -> `[584, 591]`

### Bước 5: Suy Luận Ngữ Cảnh Bổ Sung (Rule-based Fallback)
*   **Lý do áp dụng:** LLM nhỏ (< 9B) rất hay bỏ sót các từ phủ định hoặc từ chỉ tiền sử ở xa thực thể. Kỹ thuật "Rule-based Regex" đóng vai trò như một "Vệ sĩ" quét lại lần cuối để bù đắp các nhãn này, đảm bảo không bị trừ điểm oan.
*   **Thực thi trên `8.txt`:** 
    *   Hệ thống dùng Regex dò ngược văn bản từ vị trí của chữ *"khó nuốt"*, *"khó thở"*, *"khàn tiếng"*.
    *   Nó đụng ngay chướng ngại vật: **"không ghi nhận triệu chứng..."**.
    *   **Phép màu xảy ra:** Hệ thống tự động bẻ lái, chèn thêm nhãn `isNegated` vào cả 3 triệu chứng này, sửa lại lỗi bỏ sót của LLM ở Bước 3.

### Bước 6: Tìm Mã Y Tế (Hybrid Search Code Mapping)
*   **Lý do áp dụng:** Dùng LLM tự sinh mã ICD-10 là tự sát vì nó hay bịa mã. Dùng Fuzzy Matching thì bị mù từ đồng nghĩa. Giải pháp **Hybrid Search** (Lexical BM25 + Semantic sBERT) kết hợp RRF giúp tìm đúng mã dù từ khóa có bị viết tắt hay gọi bằng từ lóng.
*   **Thực thi trên `8.txt`:**
    *   Entity: `"nốt tuyến giáp thùy trái"` -> Đưa vào FAISS + BM25 -> Tìm trong cơ sở dữ liệu ICD-10.
    *   Trả về mã chuẩn xác (Ví dụ: `E04.1` - Nốt đơn thuần không độc của tuyến giáp).
    *   Entity: `"khó nuốt"`, `"khó thở"` -> Là TRIỆU_CHỨNG nên không cần mã -> `candidates: []`.

---

## 🎯 Kết Quả Cuối Cùng (Output `8.json`)

Sau khi đi qua 6 bước khép kín, file JSON cuối cùng được tạo ra hoàn hảo như sau:

```json
[
  {
    "text": "nốt tuyến giáp thùy trái",
    "position": [108, 132],
    "type": "CHẨN_ĐOÁN",
    "assertions": [],
    "candidates": ["E04.1"]
  },
  {
    "text": "chọc hút bằng kim nhỏ",
    "position": [239, 260],
    "type": "TÊN_XÉT_NGHIỆM",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "khó nuốt",
    "position": [574, 582],
    "type": "TRIỆU_CHỨNG",
    "assertions": ["isNegated"],
    "candidates": []
  },
  {
    "text": "khó thở",
    "position": [584, 591],
    "type": "TRIỆU_CHỨNG",
    "assertions": ["isNegated"],
    "candidates": []
  },
  {
    "text": "khàn tiếng",
    "position": [597, 607],
    "type": "TRIỆU_CHỨNG",
    "assertions": ["isNegated"],
    "candidates": []
  }
]
```

**Nhận xét:** File `8.txt` có một câu bẫy rất dài: *"không ghi nhận triệu chứng chèn ép vùng cổ như khó nuốt, khó thở hoặc khàn tiếng"*. Nếu làm theo cách dùng LLM thuần túy, mô hình chắc chắn sẽ bỏ sót chữ "không ghi nhận" ở tuốt phía trước chữ "khàn tiếng". Nhờ có **Bước 5 (Rule-based Fallback)**, chúng ta đã bọc lót và bắt gọn 100% các nhãn `isNegated` này!
