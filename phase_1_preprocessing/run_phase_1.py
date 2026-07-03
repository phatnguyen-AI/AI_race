import os
import csv
from preprocessor import TextPreprocessor

def main():
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, 'input', 'input')
    if not os.path.exists(input_dir):
        # Dựa vào cấu trúc hiện tại, thử tìm ở input/
        input_dir = os.path.join(base_dir, 'input')
        
    output_csv_path = os.path.join(os.path.dirname(__file__), 'section_results.csv')
    config_path = os.path.join(base_dir, 'configs', 'medical_config.json')

    # Khởi tạo Preprocessor với file cấu hình
    preprocessor = TextPreprocessor(config_path=config_path)

    # Chuẩn bị file CSV để ghi kết quả
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['file_id', 'section_type', 'start_index', 'end_index', 'matched_header']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Ghi tiêu đề cột
        writer.writeheader()

        # Quét qua 100 file
        for i in range(1, 101):
            file_name = f"{i}.txt"
            file_path = os.path.join(input_dir, file_name)

            if not os.path.exists(file_path):
                print(f"File không tồn tại: {file_path}")
                continue

            try:
                # Đọc file an toàn
                text = preprocessor.read_file_safely(file_path)

                # Nhận diện section
                sections = preprocessor.detect_sections_fuzzy(text)

                # Ghi kết quả vào CSV
                if not sections:
                    # Ghi log nếu không tìm thấy section nào
                    writer.writerow({
                        'file_id': file_name,
                        'section_type': 'NONE_FOUND',
                        'start_index': 0,
                        'end_index': len(text),
                        'matched_header': ''
                    })
                else:
                    for sec in sections:
                        writer.writerow({
                            'file_id': file_name,
                            'section_type': sec['section_type'],
                            'start_index': sec['range'][0],
                            'end_index': sec['range'][1],
                            'matched_header': sec['matched_header']
                        })
                print(f"Đã xử lý xong file: {file_name}")

            except Exception as e:
                pass
                # print(f"Lỗi khi xử lý file {file_name}: {e}")

    print(f"\nHoàn thành! Kết quả đã được lưu tại: {output_csv_path}")

if __name__ == "__main__":
    import sys
    # Fix unicode error in windows console
    sys.stdout.reconfigure(encoding='utf-8')
    main()
