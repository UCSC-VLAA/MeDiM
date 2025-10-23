import os
import re
import csv
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----------------------------
def score_image_caption(image_path, caption, model, processor, device, max_new_tokens=128):
    """返回图像与 caption 的匹配分数 [0,1]，不规范返回 None"""
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Report: {caption}. Please give a matching score from 0 (mismatch) to 1 (perfect match). Only output a decimal number."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    match = re.search(r"0(\.\d+)?|1(\.0+)?", output_text)
    if match:
        return float(match.group(0))
    else:
        return None


def main(args):
    data_dir = args.data_dir
    output_csv = args.output_csv
    model_dir = args.model_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.float16
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    print("Model loaded.")

    results, our_scores = [], []

    for file_name in tqdm(os.listdir(data_dir)):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        match = re.match(r"image_(\d+)\.png", file_name)
        if not match:
            continue
        idx = match.group(1)

        image_path = os.path.join(data_dir, file_name)
        caption_path = os.path.join(data_dir, f"text_{idx}.txt")

        if not os.path.exists(caption_path):
            continue

        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        score = score_image_caption(image_path, caption, model, processor, device)
        if score is not None:
            results.append({"image": file_name, "score": score})
            our_scores.append(score)

    if our_scores:
        print(f"Average score: {sum(our_scores)/len(our_scores):.4f}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "score"], escapechar="\\", quoting=csv.QUOTE_NONE)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing done. {len(results)} valid scores saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image-caption matching using Qwen2-VL model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing image_*.png and text_*.txt pairs.")
    parser.add_argument("--output_csv", type=str, default="medunidisc_image_caption_scores.csv", help="Output CSV file path.")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model name or path.")
    args = parser.parse_args()
    main(args)
