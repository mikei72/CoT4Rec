import torch
import random
import json
import time
from datetime import timedelta
from transformers import AutoTokenizer, T5ForConditionalGeneration

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_preference_single(model, tokenizer, device, history,
                        max_input_len=256, max_output_len=256):
    prompt = f"The user's previous order of watching movies and their ratings were: {history} Please analyze his preference characteristics based on the previous behavior."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_output_len,
        num_beams=1
    )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred.strip()


def batch_generate(model, tokenizer, device, json_path, output_path, sample_ratio=1.0, seed=42, save_every=50):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(seed)
    sample_size = int(len(data) * sample_ratio)
    sampled_data = random.sample(data, sample_size)

    print(f"原始数据量: {len(data)}  →  采样数量: {sample_size}")

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            output_existing = json.load(f)
        print("检测到已有输出文件，将继续写入未完成条目")
    except:
        output_existing = []

    existing_map = {item["history"]: item for item in output_existing}

    results = []
    completed_count = 0
    exist_count = 0

    start_time = time.time()
    batch_start = time.time()

    print("开始处理：\n")
    for idx, entry in enumerate(sampled_data, 1):
        if entry["history"] in existing_map and "pred_preference" in existing_map[entry["history"]]:
            results.append(existing_map[entry["history"]])
            completed_count += 1
            exist_count += 1
        else:
            pred = generate_preference_single(
                model,
                tokenizer,
                device,
                entry["history"]
            )

            new_entry = dict(entry)
            new_entry["pred_preference"] = pred.strip()
            results.append(new_entry)
            completed_count += 1

        # 中途保存、计时
        if (idx % save_every == 0 and completed_count > exist_count) or idx == sample_size:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            batch_time = time.time() - batch_start
            total_time = time.time() - start_time

            avg_time = total_time / (completed_count - exist_count)
            remaining = avg_time * (sample_size - completed_count)

            batch_time_str = str(timedelta(seconds=int(batch_time)))
            total_time_str = str(timedelta(seconds=int(total_time)))
            remaining_str = str(timedelta(seconds=int(remaining)))

            print(
                f"\n已处理 {completed_count}/{sample_size} 条"
                f"\n本批耗时: {batch_time_str}"
                f"\n累计耗时: {total_time_str}"
                f"\n预计剩余: {remaining_str}\n"
            )

            batch_start = time.time()

    print(f"处理完成，共写入 {len(results)} 条。")


if __name__ == "__main__":
    model_path = "experiments/ml100k-REC-P-stage-1/checkpoint-3415"
    tokenizer, model, device = load_model(model_path)

    json_path = "dataIntegration/ml100k/train.json"
    output_path = "experiments/ml100k-REC-P-stage-1/train_new.json"

    batch_generate(model, tokenizer, device, json_path, output_path, 0.2)
