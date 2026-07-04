import os
import json
import random
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

disease_pool = [
    "trào ngược dạ dày - thực quản", "viêm phổi thùy", "đái tháo đường tuýp 2", 
    "tăng huyết áp", "nhồi máu cơ tim vùng dưới cũ", "xơ gan do rượu", "rung nhĩ", "sỏi đoạn cuối ống mật chủ"
]
symptom_pool = [
    "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi", "chóng mặt", 
    "khó thở khi gắng sức", "phù mắt cá chân", "ngất xỉu", "đau bụng hạ sườn phải", "buồn nôn thoáng qua", "đánh trống ngực"
]
drug_pool = [
    "Chlorpheniramine", "Capsaicin", "metoprolol 25mg po bid", "doxycycline", "atenolol", 
    "NSAID", "omeprazole", "aspirin 325mg"
]
test_pool = [
    "WBC", "NEUT%", "LYPH%", "Siêu âm bụng", "chụp x-quang ngực", 
    "điện tâm đồ", "chụp cắt lớp vi tính sọ não", "Cộng hưởng từ mật tụy", "men gan"
]

def generate_synthetic_medical_record():
    selected_diseases = random.sample(disease_pool, k=random.randint(1, 3))
    selected_symptoms = random.sample(symptom_pool, k=random.randint(2, 5))
    selected_drugs = random.sample(drug_pool, k=random.randint(1, 3))
    selected_tests = random.sample(test_pool, k=random.randint(2, 4))
    
    length_type = random.choice([
        "NGẮN VÀ CỤT LỦN (khoảng 50 - 150 từ, ghi chép nhanh chóng, vắn tắt).",
        "TRUNG BÌNH (khoảng 200 - 300 từ, thông tin vừa đủ).",
        "RẤT DÀI VÀ LÊ THÊ (khoảng 400 - 800 từ, kể lể nhiều chi tiết râu ria, triệu chứng không mắc phải, hỏi đáp lan man)."
    ])
    
    prompt = f"""
Bạn là một chuyên gia tạo dữ liệu y khoa giả lập (Synthetic Medical Data Generator).
Hãy sinh ra một hồ sơ bệnh án tiếng Việt cực kỳ chân thực, lộn xộn và có độ phức tạp cao, mô phỏng chính xác văn phong của bác sĩ gõ vội trong bệnh viện.

**YÊU CẦU CẤU TRÚC VÀ ĐỘ DÀI VĂN BẢN (raw_text):**
Bệnh án này BẮT BUỘC phải có độ dài: {length_type}
Bệnh án BẮT BUỘC phải chia làm 3 phần:
1. Tiền sử bệnh (Ghi chú các bệnh mạn tính, tiền sử gia đình, thuốc đang dùng).
2. Tiền sử bệnh hiện tại (Nêu lý do nhập viện, liệt kê triệu chứng bằng gạch đầu dòng theo form như: '- Vị trí: ...', '- Thời gian: ...', '- Yếu tố làm nặng: ...').
3. Đánh giá tại bệnh viện (Kết quả xét nghiệm, chẩn đoán hình ảnh, các thủ thuật can thiệp).

**YÊU CẦU ĐA DẠNG HÓA LỖI VÀ MÔ PHỎNG THỰC TẾ (BẮT BUỘC):**
Để mô phỏng dữ liệu thực tế, bạn PHẢI áp dụng NGẪU NHIÊN các loại "Data bẩn" và thông tin sau vào `raw_text`:
- Hành chính & Thời gian: Bắt đầu bệnh án bằng thông tin viết tắt như "BN nam, 70t" hoặc "BN nữ, 45t", kết hợp mốc thời gian "cách 2 tuần", "lúc 17h30".
- Chỉ số sinh tồn (Vitals) lộn xộn: Chèn các chuỗi đo lường thực tế hệt như máy xuất ra, VD: "VS98.3 12987 56 18 99RA" hoặc "mạch 83, HA 159/72".
- Sự không chắc chắn (Hedging): Dùng các cụm từ "Theo dõi...", "Nghi ngờ...", "Chưa loại trừ..." trước các CHẨN_ĐOÁN để tăng độ khó.
- Lỗi dính chữ sau dấu câu: VD "dạ dày.Bệnh nhân", "hiện tạiBệnh nhân", "bệnh.Không"
- Lỗi lặp từ ngớ ngẩn (2-3 từ liên tiếp): VD "bình thườngbình thường", "khó thở nhẹ khó thở"
- Lỗi đánh máy: VD "bênh nhân", "tụi mật" (thay vì túi mật), "xươg" (thay vì xương)
- Viết tắt y khoa chuyên ngành dày đặc: "cls" (cận lâm sàng), "cđha" (chẩn đoán hình ảnh), "tbm" (tế bào máu), "spo2 RA", "po bid" (uống ngày 2 lần).

**RÀNG BUỘC VỀ NỘI DUNG (Hồ chứa từ khóa - Bắt buộc dùng):**
- Chọn Bệnh lý (CHẨN_ĐOÁN) từ: {', '.join(selected_diseases)}
- Chọn Triệu chứng (TRIỆU_CHỨNG) từ: {', '.join(selected_symptoms)}
- Chọn Loại thuốc (THUỐC) từ: {', '.join(selected_drugs)}
- Chọn Xét nghiệm (TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM) từ: {', '.join(selected_tests)}

**RÀNG BUỘC VỀ ĐỘ KHÓ & NHÃN (Assertions):**
1. Đưa vào ít nhất 1 câu PHỦ ĐỊNH liên quan đến triệu chứng (VD: "Không có dấu hiệu đau ngực"). -> Gắn nhãn isNegated.
2. Đưa vào ít nhất 1 thông tin TIỀN SỬ (VD: "Tiền sử đã từng sử dụng NSAIDs"). -> Gắn nhãn isHistorical.
3. Trong mảng entities, giá trị `text` trích xuất ra phải LÀ MỘT CHUỖI CON CHÍNH XÁC 100% (exact substring) trích ra từ `raw_text` (Nếu trong raw_text cố tình viết sai chính tả từ đó, thì entity text cũng phải lưu từ sai chính tả y hệt).

Trả về định dạng JSON gồm:
- "raw_text": Đoạn bệnh án
- "entities": Danh sách các thực thể trích xuất.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là AI sinh dữ liệu y khoa."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "medical_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "raw_text": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "type": {"type": "string", "enum": ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"]},
                                    "assertions": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["isNegated", "isFamily", "isHistorical"]}
                                    }
                                },
                                "required": ["text", "type", "assertions"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["raw_text", "entities"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        temperature=0.7,
    )
    
    result = json.loads(response.choices[0].message.content)

    raw_text = result["raw_text"]
    for entity in result["entities"]:
        start_idx = raw_text.find(entity["text"])
        if start_idx != -1:
            entity["position"] = [start_idx, start_idx + len(entity["text"])]
        else:
            entity["position"] = [-1, -1] 
            
    return result

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    num_samples_to_generate = 8000
    print(f"{num_samples_to_generate}")
    
    def process_file(i):
        try:
            txt_path = os.path.join(output_dir, f"synthetic_{i}.txt")
            json_path = os.path.join(output_dir, f"synthetic_{i}.json")
            
            if os.path.exists(txt_path) and os.path.exists(json_path):
                return True
                
            synthetic_data = generate_synthetic_medical_record()
        
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(synthetic_data["raw_text"])
                
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(synthetic_data["entities"], f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            return False


    max_workers = 10
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, i): i for i in range(1, num_samples_to_generate + 1)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples_to_generate, desc="Tiến độ sinh dữ liệu"):
            if future.result():
                success_count += 1
                
    print(f"\n xong {success_count}/{num_samples_to_generate} file.")
