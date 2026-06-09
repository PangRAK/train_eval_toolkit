import argparse
import itertools
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.multiprocessing as mp
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from src.training.internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
import torchvision.transforms as T
from decord import VideoReader, cpu
import numpy as np
import yaml
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPTS_PATH = ROOT_DIR / "00_preprocessing" / "insturction" / "prompts.yaml"
PROMPT_TEMPLATE = "<image> <CLS_FALLDOWN>"


def load_prompt_config():
    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config["prompts"]


PROMPT_CONFIG = load_prompt_config()


def compose_prompt(prompt_names):
    if not prompt_names:
        raise ValueError("prompt_names must not be empty.")

    missing_names = [name for name in prompt_names if name not in PROMPT_CONFIG]
    if missing_names:
        available_names = ", ".join(sorted(PROMPT_CONFIG))
        missing_text = ", ".join(missing_names)
        raise ValueError(
            f"Unknown prompt name(s): {missing_text}. Available prompts: {available_names}"
        )

    combined = ""
    for name in prompt_names:
        fragment = str(PROMPT_CONFIG[name]).strip()
        if not fragment:
            continue

        if not combined:
            combined = fragment
            continue

        if "\n" not in combined and "\n" not in fragment:
            combined = f"{combined} {fragment}"
        else:
            combined = f"{combined}\n{fragment}"

    return combined

def build_transform(input_size: int = 448):
    """Return torchvision transform matching InternVL pre‑training."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tgt_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt_ar)
        if diff < best_ratio_diff or (diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split arbitrarily‑sized image into ≤12 tiles sized 448×448 (InternVL spec)."""
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ow, oh, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = [
        resized.crop(
            (
                (idx % (tw // image_size)) * image_size,
                (idx // (tw // image_size)) * image_size,
                ((idx % (tw // image_size)) + 1) * image_size,
                ((idx // (tw // image_size)) + 1) * image_size,
            )
        )
        for idx in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- Multi-GPU Model Splitting Function ---
import math
import torch

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = InternVLChatConfig.from_pretrained(model_path)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
def load_model_and_tokenizer(args):
    """Loads the model and tokenizer based on command-line arguments."""
    print(f"Loading model from: {args.checkpoint}")
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    
    if args.multi_gpu:
        print("Using multi-GPU model splitting.")
        device_map = split_model(args.checkpoint)
        model = InternVLChatModel.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            config=config,
            device_map=device_map
        ).eval()
    else:
        print("Loading model in bfloat16 on a single GPU.")
        model = InternVLChatModel.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            config=config
        ).eval().cuda()
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        use_fast=False,
    )
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    
    return model, tokenizer, image_size


def parse_prediction(pred_str: str) -> str:
    """
    모델의 출력 문자열을 1-token yes/no 응답으로 정규화합니다.
    """
    try:
        clean_str = pred_str
        if '```json' in clean_str:
            clean_str = clean_str.split('```json')[1].split('```')[0]
        elif '```' in clean_str:
            clean_str = clean_str.split('```')[1].split('```')[0]
        clean_str = clean_str.strip().lower().strip("\"'")

        if not clean_str:
            return "invalid_prediction"

        first_token = clean_str.split()[0]
        if first_token in {"yes", "no"}:
            return first_token

        if first_token in {"falldown", "normal"}:
            return "yes" if first_token == "falldown" else "no"

        yes_match = re.search(r"\byes\b", clean_str)
        if yes_match:
            return "yes"

        no_match = re.search(r"\bno\b", clean_str)
        if no_match:
            return "no"

        return "invalid_prediction"

    except Exception:
        return "invalid_prediction"


def answer_to_category(answer: str) -> str:
    normalized = answer.strip().lower().strip("\"'")
    if normalized == "yes":
        return "falldown"
    if normalized == "no":
        return "normal"
    return "parsing_error"
# --- NEW: Main Evaluation Function using Batch Inference ---

def evaluate_in_batches(model, tokenizer, image_size, all_data, args):
    """
    Evaluates the dataset using batch inference.
    """
    results_list = []
    y_true = []
    y_pred = []
    token_latency_records = []
    
    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
    )

    # Iterate over the dataset in chunks of batch_size
    for i in tqdm(range(0, len(all_data), args.batch_size), desc="Evaluating Batches"):
        batch_data = all_data[i:i + args.batch_size]
        
        batch_pixel_values = []
        num_patches_list = []
        
        # Prepare inputs for the current batch
        for item in batch_data:
            image_path = os.path.join(args.image_root, item['image'])
            try:
                # Load each image and get its pixel values (tiles)
                pixel_values = load_image(
                    image_path,
                    input_size=image_size,
                    max_num=args.max_dynamic_patch,
                )
                batch_pixel_values.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
            except Exception as e:
                print(f"Skipping unloadable image {image_path}: {e}")
                # Add a placeholder to keep batch items aligned
                num_patches_list.append(-1) 

        # Filter out failed loads
        valid_indices = [idx for idx, num in enumerate(num_patches_list) if num != -1]
        if not valid_indices:
            continue
        
        valid_batch_data = [batch_data[idx] for idx in valid_indices]
        valid_pixel_values = [batch_pixel_values[idx] for idx in valid_indices]
        valid_num_patches = [num_patches_list[idx] for idx in valid_indices]

        # Concatenate all image tiles into a single tensor for the batch
        pixel_values_tensor = torch.cat(valid_pixel_values, dim=0).to(torch.bfloat16)

        # For multi-GPU, inputs must be on the primary device (GPU 0)
        # For single-GPU, .cuda() is fine. `next(model.parameters()).device` is a robust way.
        device = next(model.parameters()).device
        pixel_values_tensor = pixel_values_tensor.to(device)

        # Prepare questions for the batch
        questions = [args.prompt] * len(valid_batch_data)
        
        # Perform batch inference
        if args.token_latency:
            responses, token_latency_info = model.batch_chat(
                tokenizer,
                pixel_values=pixel_values_tensor,
                num_patches_list=valid_num_patches,
                questions=questions,
                generation_config=generation_config,
                collect_token_latency=True
            )
            batch_id = len(token_latency_records)
            token_latency_records.append({
                'batch_id': batch_id,
                'global_start_index': i,
                'effective_batch_size': token_latency_info['batch_size'],
                'inference_latency_sec': token_latency_info['inference_latency_sec'],
                'batch_input_token_total': token_latency_info['batch_input_token_total'],
                'batch_input_token_max': token_latency_info['batch_input_token_max'],
                'input_token_counts': token_latency_info['input_token_counts'],
                'num_patches_list': token_latency_info['num_patches_list'],
                'gpu_memory_stats': token_latency_info['gpu_memory_stats'],
                'gpu_allocated_mb_total': token_latency_info['gpu_allocated_mb_total'],
                'gpu_reserved_mb_total': token_latency_info['gpu_reserved_mb_total'],
                'gpu_peak_allocated_mb_total': token_latency_info['gpu_peak_allocated_mb_total'],
                'gpu_peak_reserved_mb_total': token_latency_info['gpu_peak_reserved_mb_total'],
                'image_paths': [item['image'] for item in valid_batch_data],
            })
        else:
            responses = model.batch_chat(
                tokenizer,
                pixel_values=pixel_values_tensor,
                num_patches_list=valid_num_patches,
                questions=questions,
                generation_config=generation_config
            )
        # Process results for the batch
        for item, response in zip(valid_batch_data, responses):
            predicted_answer = parse_prediction(response)
            predicted_category = answer_to_category(predicted_answer)
            
            try:
                gt_value_str = item['conversations'][0]['value']
                ground_truth_category = answer_to_category(gt_value_str)
            except Exception:
                ground_truth_category = "parsing_error"
            
            results_list.append({
                'image_path_log' : item['image'],
                'prediction': predicted_category,
                'prediction_answer': predicted_answer,
                'ground_truth': ground_truth_category,
                'is_correct': predicted_category == ground_truth_category,
                'pred_result' : response
            })
            
            # Collect labels for final metrics calculation
            if ground_truth_category in {'falldown', 'normal'}:
                y_true.append(ground_truth_category)
                y_pred.append('normal' if predicted_category == 'normal' else 'falldown')
                

    # --- Final Metrics Calculation and Reporting ---
    total_samples = len(results_list)
    valid_samples = len(y_true)
    invalid_gt_samples = len(all_data) - valid_samples # Count from original data size
    correct_predictions = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
    accuracy = (correct_predictions / valid_samples) * 100 if valid_samples > 0 else 0.0
    
    precision = precision_score(y_true, y_pred, pos_label='falldown', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label='falldown', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='falldown', zero_division=0)
    
    print("\n--- Evaluation Summary ---")
    print(f"GPUs used: {torch.cuda.device_count() if args.multi_gpu else 1}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Videos Submitted: {len(all_data)}")
    print(f"Videos Processed (loadable): {total_samples}")
    print(f"Invalid Ground-Truth Samples (excluded from metrics): {invalid_gt_samples}")
    print(f"Valid Videos for Metrics: {valid_samples}")
    print("------------------------------------")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_samples})")
    print(f"Precision (falldown): {precision:.4f}")
    print(f"Recall (falldown):    {recall:.4f}")
    print(f"F1-Score (falldown):  {f1:.4f}")
    print("------------------------------------\n")

    # --- Save Results ---
    time_prefix = time.strftime('%Y_%m_%d', time.localtime())
    model_name = os.path.basename(args.checkpoint)
    dataset_name = os.path.splitext(os.path.basename(args.annotation))[0]
    new_filename = f"{time_prefix}_{model_name}_{dataset_name}.json"
    results_file_path = os.path.join(args.out_dir, new_filename)
    with open(results_file_path, 'w') as f:
        json.dump(results_list, f, indent=4)
    print(f"Detailed results saved to: {results_file_path}")
    
    summary_path = os.path.join(args.out_dir, 'evaluation_summary.txt')
    with open(summary_path, 'a') as f:
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Annotation: {args.annotation}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Token Latency Enabled: {args.token_latency}\n")
        f.write(f"Total Samples: {len(all_data)} (Valid for metrics: {valid_samples})\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        if token_latency_records:
            avg_latency = sum(item['inference_latency_sec'] for item in token_latency_records) / len(token_latency_records)
            avg_tokens = sum(item['batch_input_token_total'] for item in token_latency_records) / len(token_latency_records)
            avg_peak_allocated = sum(item['gpu_peak_allocated_mb_total'] for item in token_latency_records) / len(token_latency_records)
            avg_peak_reserved = sum(item['gpu_peak_reserved_mb_total'] for item in token_latency_records) / len(token_latency_records)
            f.write(f"Avg Batch Input Tokens: {avg_tokens:.2f}\n")
            f.write(f"Avg Batch Inference Time (sec): {avg_latency:.4f}\n")
            f.write(f"Avg Total Peak GPU Allocated (MB): {avg_peak_allocated:.2f}\n")
            f.write(f"Avg Total Peak GPU Reserved (MB): {avg_peak_reserved:.2f}\n")
        f.write("-" * 20 + "\n")
    print(f"Summary saved to: {summary_path}")

    if token_latency_records:
        token_latency_path = os.path.join(args.out_dir, 'token_latency.json')
        with open(token_latency_path, 'w') as f:
            json.dump(token_latency_records, f, indent=4)
        print(f"Token latency results saved to: {token_latency_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--annotation', type=str, required=True, help="Path to the annotation JSONL file.")
    parser.add_argument('--image-root', type=str, default='', help="Root directory for the image/image files.")
    parser.add_argument('--out-dir', type=str, default='results/eval_result', help="Directory to save results.")
    
    parser.add_argument('--batch-size', type=int, default=8, help="Number of images to process in one batch.")
    parser.add_argument('--multi-gpu', action='store_true', help="Enable multi-GPU model splitting.")
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--max-dynamic-patch', type=int, default=12, help="Maximum dynamic image patches used at evaluation time.")
    parser.add_argument('--token-latency', action='store_true', help="Record input token counts and batch inference latency.")
    parser.add_argument('--prompt-list', nargs='+', default=None, help="Prompt names defined in 00_preprocessing/insturction/prompts.yaml.")
    args = parser.parse_args()
    args.prompt = compose_prompt(args.prompt_list) if args.prompt_list else PROMPT_TEMPLATE.strip()
    
    # Setup output directory
    timestamp = time.strftime('%Y_%m_%d_%H-%M-%S')
    new_out_dir = os.path.join(args.out_dir, timestamp)
    args.out_dir = new_out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model and tokenizer using the new unified function
    model, tokenizer, image_size = load_model_and_tokenizer(args)
    
    # Load all data from the annotation file
    all_data = []
    with open(args.annotation, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
            
    print(f"Found {len(all_data)} samples in the annotation file.")

    # Run the evaluation
    evaluate_in_batches(model, tokenizer, image_size, all_data, args)
