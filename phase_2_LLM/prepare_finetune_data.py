import os
import json
import glob

def main():
    import random
    data_dir = "data"
    
    train_file = "train.json"
    test_file = "test.json"
    
    dataset = []
    
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"Tìm thấy {len(txt_files)} file text. Đang tiến hành tổng hợp...")
    
    for txt_path in txt_files:
        json_path = txt_path.replace(".txt", ".json")
        if os.path.exists(json_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()
                
                with open(json_path, "r", encoding="utf-8") as f:
                    entities = json.load(f)
                    
                output_str = json.dumps(entities, ensure_ascii=False)
                
                instruction = "Bạn là chuyên gia NLP y khoa. Trích xuất toàn bộ khái niệm y tế từ đoạn văn sau và trả về JSON hợp lệ theo schema đã cho."
                
                sample = {
                    "instruction": instruction,
                    "input": raw_text,
                    "output": output_str
                }
                
                dataset.append(sample)
                
            except Exception as e:
                print(f"Lỗi khi xử lý cặp file {txt_path}: {e}")
                    
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # Lưu train set
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    # Lưu test set
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
