import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("Thiếu thư viện AWQ. Vui lòng chạy: pip install autoawq")
        return

    phase_2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(phase_2_dir)
    
    adapter_dir = os.path.join(project_dir, "models", "qwen_7b_medical_adapter")
    merged_model_dir = os.path.join(project_dir, "models", "qwen_7b_merged_tmp")
    
    quantized_model_dir = os.path.join(project_dir, "models", "qwen_7b_medical_awq")
    
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"

    if not os.path.exists(merged_model_dir):
        print("Bước 1: Nối (Merge) LoRA adapter vào Base Model...")
        if not os.path.exists(adapter_dir):
            print(f"LỖI: Chưa có adapter tại {adapter_dir}. Cần chạy train trước!")
            return
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16, 
            device_map="cpu"
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        print(f"Đã lưu mô hình sau khi merge vào {merged_model_dir}")
    
    print("Bước 2: Bắt đầu Lượng tử hóa AWQ (4-bit)...")
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    
    model = AutoAWQForCausalLM.from_pretrained(merged_model_dir, safetensors=True, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, trust_remote_code=True)

    print("Đang tối ưu hóa weights (sẽ mất khoảng 15-30 phút tùy CPU/GPU)...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    print(f"Đang lưu mô hình lượng tử vào: {quantized_model_dir}")
    model.save_quantized(quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)
    
    print(f"{merged_model_dir}")

if __name__ == "__main__":
    main()
