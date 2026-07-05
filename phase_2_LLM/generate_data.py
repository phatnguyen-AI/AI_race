import os
import json
import random
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# HỒ CHỨA IN-DOMAIN (Dành cho 80% dữ liệu)
disease_pool = [
    "trào ngược dạ dày - thực quản", "viêm phổi thùy", "đái tháo đường tuýp 2", 
    "tăng huyết áp", "nhồi máu cơ tim vùng dưới cũ", "xơ gan do rượu", "rung nhĩ", "sỏi đoạn cuối ống mật chủ",
    "bệnh tim thiếu máu cục bộ", "viêm phế quản mạn tính", "hen suyễn", "suy thận mạn", "viêm gan siêu vi B",
    "viêm khớp dạng thấp", "loét dạ dày tá tràng", "rối loạn lipid máu", "suy tim sung huyết", "trĩ nội",
    "viêm amidan cấp", "sốt xuất huyết Dengue", "đau dây thần kinh tọa", "viêm đường tiết niệu", "bướu cổ"
]

symptom_pool = [
    "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi", "chóng mặt", 
    "khó thở khi gắng sức", "phù mắt cá chân", "ngất xỉu", "đau bụng hạ sườn phải", "buồn nôn thoáng qua", "đánh trống ngực",
    "ho khan", "tiêu chảy", "táo bón", "sốt cao liên tục", "rét run", "đau mỏi cơ khớp", "đau nửa đầu",
    "tê bì chân tay", "tiểu buốt", "tiểu dắt", "vã mồ hôi", "sụt cân không rõ nguyên nhân", "mất ngủ"
]

drug_pool = [
    "Chlorpheniramine", "Capsaicin", "metoprolol 25mg po bid", "doxycycline", "atenolol", 
    "NSAID", "omeprazole", "aspirin 325mg", "Paracetamol", "Ibuprofen", "Amoxicillin", "Cefuroxime",
    "Losartan", "Amlodipine", "Metformin", "Atorvastatin", "Salbutamol", "Pantoprazole", "Vitamin C",
    "Insulin", "Clopidogrel", "Furosemide", "Spironolactone", "Levofloxacin"
]

test_pool = [
    "WBC", "NEUT%", "LYPH%", "Siêu âm bụng", "chụp x-quang ngực", 
    "điện tâm đồ", "chụp cắt lớp vi tính sọ não", "Cộng hưởng từ mật tụy", "men gan",
    "RBC", "HGB", "PLT", "Glucose máu", "HbA1c", "Ure", "Creatinin", "AST", "ALT", "Bilirubin",
    "X-quang cột sống thắt lưng", "Siêu âm tim", "Đo chức năng hô hấp", "Nội soi dạ dày", "Tổng phân tích nước tiểu"
]

def generate_synthetic_medical_record(is_indomain=True):
    length_type = random.choice([
        "NGẮN VÀ CỤT LỦN (khoảng 50 - 150 từ, ghi chép nhanh chóng, vắn tắt).",
        "TRUNG BÌNH (khoảng 200 - 300 từ, thông tin vừa đủ).",
        "RẤT DÀI VÀ LÊ THÊ (khoảng 400 - 800 từ, kể lể nhiều chi tiết râu ria, triệu chứng không mắc phải, hỏi đáp lan man)."
    ])
    
    if is_indomain:
        selected_diseases = random.sample(disease_pool, k=random.randint(1, 3))
        selected_symptoms = random.sample(symptom_pool, k=random.randint(2, 5))
        selected_drugs = random.sample(drug_pool, k=random.randint(1, 3))
        selected_tests = random.sample(test_pool, k=random.randint(2, 4))
        
        domain_instruction = f"""
**RÀNG BUỘC VỀ NỘI DUNG (Hồ chứa từ khóa - Bắt buộc dùng để bám sát chuyên khoa Nội/Hô Hấp/Tim mạch):**
- Bệnh lý (CHẨN_ĐOÁN) phải chứa: {', '.join(selected_diseases)}
- Triệu chứng (TRIỆU_CHỨNG) phải chứa: {', '.join(selected_symptoms)}
- Loại thuốc (THUỐC) phải chứa: {', '.join(selected_drugs)}
- Xét nghiệm (TÊN_XÉT_NGHIỆM / KẾT_QUẢ_XÉT_NGHIỆM) phải chứa: {', '.join(selected_tests)}
"""
    else:
        # OUT-OF-DOMAIN: Ép mô hình tự bịa ra các chuyên khoa hiếm
        specialties = ["Sản phụ khoa", "Nhi khoa", "Ung bướu", "Tâm thần", "Răng Hàm Mặt", "Da liễu", "Mắt", "Tai Mũi Họng"]
        random_specialty = random.choice(specialties)
        domain_instruction = f"""
**RÀNG BUỘC VỀ NỘI DUNG (OUT-OF-DOMAIN - Khái quát hóa):**
Bạn KHÔNG ĐƯỢC dùng các bệnh quen thuộc như Tim mạch, Huyết áp, Dạ dày.
Hãy tạo một bệnh án thuộc chuyên khoa **{random_specialty}**. Tự sáng tạo ra các bệnh lý, thuốc, triệu chứng và xét nghiệm chuyên sâu, hiếm gặp của chuyên khoa này.
"""

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
- Hành chính & Thời gian: Bắt đầu bệnh án bằng thông tin viết tắt như "BN nam, 70t" hoặc "BN nữ, 45t".
- Chỉ số sinh tồn (Vitals) lộn xộn: "VS98.3 12987 56 18 99RA" hoặc "mạch 83, HA 159/72".
- Sự không chắc chắn (Hedging): "Theo dõi...", "Nghi ngờ...", "Chưa loại trừ...".
- Lỗi dính chữ sau dấu câu: "dạ dày.Bệnh nhân", "hiện tạiBệnh nhân".
- Lỗi đánh máy: "bênh nhân", "tụi mật", "xươg".
- Viết tắt y khoa chuyên ngành: "cls", "cđha", "tbm", "spo2 RA", "po bid".

{domain_instruction}

**RÀNG BUỘC VỀ ĐỘ KHÓ & NHÃN (Assertions):**
1. Đưa vào ít nhất 1 câu PHỦ ĐỊNH liên quan đến triệu chứng/bệnh (VD: "Không ho, không sốt"). -> Gắn nhãn isNegated.
2. Đưa vào ít nhất 1 thông tin TIỀN SỬ (VD: "Trước đây từng bị viêm loét"). -> Gắn nhãn isHistorical.
3. Đưa vào ít nhất 1 thông tin BỆNH LÝ CỦA NGƯỜI THÂN (VD: "Bố ruột mắc tiểu đường, mẹ bị tăng huyết áp"). -> Gắn nhãn isFamily.
4. Trong mảng entities, giá trị `text` trích xuất ra phải LÀ MỘT CHUỖI CON CHÍNH XÁC 100% trích ra từ `raw_text` (Kể cả sai chính tả).

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
    
    # Đã loại bỏ code gán `position` để đúng với thiết kế v4

    return result

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    num_samples_to_generate = 10000  # Cập nhật thành 10000 theo v4
    print(f"Bắt đầu sinh {num_samples_to_generate} file (80% In-domain, 20% Out-of-domain)...")
    
    def process_file(i):
        try:
            txt_path = os.path.join(output_dir, f"synthetic_{i}.txt")
            json_path = os.path.join(output_dir, f"synthetic_{i}.json")
            
            # Bỏ qua nếu file đã tồn tại
            if os.path.exists(txt_path) and os.path.exists(json_path):
                return True
                
            # Phân bổ 80% In-domain, 20% Out-of-domain
            is_indomain = True if i <= 8000 else False
            synthetic_data = generate_synthetic_medical_record(is_indomain=is_indomain)
        
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(synthetic_data["raw_text"])
                
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(synthetic_data["entities"], f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            return False

    max_workers = 15 # Nâng worker để tối ưu tốc độ gọi API
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, i): i for i in range(1, num_samples_to_generate + 1)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples_to_generate, desc="Tiến độ sinh dữ liệu"):
            if future.result():
                success_count += 1
                
    print(f"\n Xong {success_count}/{num_samples_to_generate} file.")
