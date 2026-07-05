import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    phase_2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(phase_2_dir)
    
    test_data_path = os.path.join(phase_2_dir, "test.json")
    adapter_dir = os.path.join(project_dir, "models", "qwen_7b_medical_adapter")
    
    if not os.path.exists(test_data_path):
        print("Không tìm thấy file dữ liệu test.")
        return

    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("Đang tải Tokenizer và Base model...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    if os.path.exists(adapter_dir):
        print(f"Đang tải LoRA adapter từ {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        print("Không tìm thấy Adapter. Sẽ chạy đánh giá trên model GỐC chưa fine-tune.")

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    # Chạy thử 20 mẫu để kiểm tra
    sample_size = min(20, len(test_data))
    print(f"Bắt đầu chạy đánh giá trên {sample_size} mẫu...")
    
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
        
    output_res_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(output_res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Đánh giá hoàn tất. Kết quả sinh text đã lưu tại: {output_res_path}")

if __name__ == "__main__":
    main()
