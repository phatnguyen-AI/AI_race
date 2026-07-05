import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    phase_2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(phase_2_dir)
    
    data_path = os.path.join(phase_2_dir, "train.json")
    
    model_output_dir = os.path.join(project_dir, "models", "qwen_7b_medical_adapter")
    
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu: {data_path}")
        return

    dataset = load_dataset('json', data_files={'train': data_path})['train']
    
    # Hàm format prompt cho Qwen2.5-Instruct
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<|im_start|>system\n{example['instruction'][i]}<|im_end|>\n<|im_start|>user\n{example['input'][i]}<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    print(f"Đang tải mô hình gốc {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Cấu hình LoRA Adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        num_train_epochs=3, # Train 3 epochs
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args
    )
    
    print("Bắt đầu tiến trình huấn luyện (Fine-tuning)...")
    trainer.train()
    
    print(f"Huấn luyện hoàn tất! Đang lưu mô hình vào: {model_output_dir}")
    trainer.model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

if __name__ == "__main__":
    main()
