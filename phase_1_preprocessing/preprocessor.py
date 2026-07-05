import os
import json
from thefuzz import fuzz

class TextPreprocessor:
    def __init__(self, config_path: str = None):
        self.section_keywords = {}
        self.fuzzy_threshold = 85
        self.hard_overrides = []
        
        # Load cấu hình từ file JSON nếu được cung cấp
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.section_keywords = config.get("section_keywords", {})
                self.fuzzy_threshold = config.get("fuzzy_threshold", 85)
                self.hard_overrides = config.get("hard_overrides", [])
        else:
            # Fallback về mặc định nếu không có file cấu hình
            self.section_keywords = {
                "TIEN_SU": ["tiền sử bệnh", "tiền sử:", "ts:", "bản thân:", "tiền sử nội khoa", "bệnh lý cũ"],
                "HIEN_TAI": ["bệnh sử hiện tại", "bệnh sử:", "quá trình bệnh lý", "diễn biến bệnh"],
                "DANH_GIA": ["đánh giá tại bệnh viện", "khám lâm sàng", "cận lâm sàng", "kết quả xét nghiệm", "tình trạng lúc vào viện", "lâm sàng:"]
            }
            self.fuzzy_threshold = 85
            self.hard_overrides = [
                {
                    "trigger_section": "TIEN_SU",
                    "required_keywords": ["hiện tại", "diễn biến"],
                    "target_section": "HIEN_TAI"
                }
            ]

    def read_file_safely(self, file_path: str) -> str:
        """Đọc file gốc"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def detect_sections_fuzzy(self, text: str) -> list[dict]:
        """
        Phát hiện các section trong văn bản dựa trên từ khóa và độ tương đồng (fuzzy matching).
        """
        # Tách văn bản thành từng dòng
        lines = []
        current_index = 0
        for line in text.split('\n'):
            lines.append({
                'text': line,
                'start': current_index,
                'end': current_index + len(line)
            })
            current_index += len(line) + 1 # +1 cho ký tự '\n'

        detected_sections = []
        
        for line_info in lines:
            line_text = line_info['text'].strip().lower()
            
            # Bỏ qua dòng trống hoặc dòng liệt kê
            if not line_text or line_text.startswith(('-','+','*')): 
                continue
                
            best_match_type = None
            best_score = 0
            
            # 1. Xử lý dòng ngắn (thường là tiêu đề chuẩn): Dùng thuật toán Fuzzy
            if len(line_text) <= 40:
                for sec_type, keywords in self.section_keywords.items():
                    for kw in keywords:
                        score = fuzz.partial_ratio(kw, line_text)
                        if score > best_score:
                            best_score = score
                            best_match_type = sec_type
            
            # 2. Xử lý dòng dài (bị dính chữ, gõ liền không xuống dòng): Tìm chính xác từ khoá mạnh (có dấu hai chấm)
            else:
                for sec_type, keywords in self.section_keywords.items():
                    for kw in keywords:
                        if ":" in kw and kw in line_text:
                            best_score = 100
                            best_match_type = sec_type
                            # Lùi toạ độ start vào đúng vị trí của từ khoá bị dính trong câu
                            line_info['start'] += line_info['text'].lower().find(kw)
                            break
                    if best_score == 100:
                        break

            # Nếu độ giống nhau vượt ngưỡng cho phép -> Ghi nhận đây là mốc bắt đầu section mới
            if best_score >= self.fuzzy_threshold:
                # Áp dụng các luật bẻ lái (Hard Overrides) từ file config
                for override in self.hard_overrides:
                    if best_match_type == override["trigger_section"]:
                        if any(kw in line_text for kw in override["required_keywords"]):
                            best_match_type = override["target_section"]
                            break

                detected_sections.append({
                    'type': best_match_type,
                    'start_index': line_info['start'],
                    'confidence_score': best_score,
                    'matched_line': line_info['text']
                })

        # Sắp xếp các section theo vị trí xuất hiện trong văn bản
        section_ranges = []
        
        # Xử lý đoạn văn bản "mồ côi" ở đầu file (không có tiêu đề)
        if detected_sections and detected_sections[0]['start_index'] > 0:
            section_ranges.append({
                'section_type': 'THONG_TIN_CHUNG',
                'range': [0, detected_sections[0]['start_index']],
                'matched_header': ''
            })
        elif not detected_sections:
            # Nếu không tìm thấy section nào trong toàn văn bản
            section_ranges.append({
                'section_type': 'THONG_TIN_CHUNG',
                'range': [0, len(text)],
                'matched_header': ''
            })

        for i in range(len(detected_sections)):
            start = detected_sections[i]['start_index']
            # End của section này là start của section tiếp theo (hoặc hết văn bản)
            end = detected_sections[i+1]['start_index'] if i + 1 < len(detected_sections) else len(text)
            
            section_ranges.append({
                'section_type': detected_sections[i]['type'],
                'range': [start, end],
                'matched_header': detected_sections[i]['matched_line'].strip()
            })

        return section_ranges
