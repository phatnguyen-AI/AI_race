import os
import json
import torch
from tqdm import tqdm

def main():
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("Thiếu thư viện AWQ hoặc Transformers. Vui lòng chạy: pip install autoawq transformers")
        return

    phase_2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(phase_2_dir)
    
    test_data_path = os.path.join(phase_2_dir, "test.json")
    awq_model_dir = os.path.join(project_dir, "models", "qwen_7b_medical_awq")
    
    if not os.path.exists(test_data_path):
        print("Không tìm thấy file dữ liệu test.")
        return

    if not os.path.exists(awq_model_dir):
        print(f"LỖI: Không tìm thấy model AWQ tại {awq_model_dir}. Cần chạy lượng tử hóa (quantize_awq.py) trước!")
        return

    print(f"Đang tải Tokenizer và mô hình Lượng tử hóa AWQ 4-bit từ {awq_model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(awq_model_dir, trust_remote_code=True)
    # Tải mô hình AWQ
    model = AutoAWQForCausalLM.from_quantized(
        awq_model_dir, 
        fuse_layers=True, 
        safetensors=True, 
        strict=False
    )
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    sample_size = min(20, len(test_data))
    print(f"Bắt đầu chạy đánh giá trên {sample_size} mẫu bằng mô hình siêu nhẹ AWQ...")
    
    results = []
    for i in tqdm(range(sample_size)):
        item = test_data[i]
        prompt = f"<|im_start|>system\n{item['instruction']}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "input": item["input"],
            "ground_truth": item["output"],
            "prediction": response
        })
        
    output_res_path = os.path.join(os.path.dirname(__file__), "evaluation_awq_results.json")
    with open(output_res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Đánh giá hoàn tất. Kết quả sinh text (AWQ) đã lưu tại: {output_res_path}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
