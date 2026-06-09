#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.internvl.model.internvl_chat import InternVLChatModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Add special tokens to an InternVL checkpoint and optionally resize model embeddings.'
    )
    parser.add_argument('checkpoint_path', type=Path, help='Path to the checkpoint directory.')
    parser.add_argument(
        '--output-path',
        type=Path,
        default=None,
        help='Where to save the updated checkpoint. Default: <checkpoint_path>_with_special_tokens',
    )
    parser.add_argument(
        '--token',
        action='append',
        default=[],
        help='Additional special token to add. Repeat this flag to add multiple tokens.',
    )
    parser.add_argument(
        '--token-file',
        type=Path,
        default=None,
        help='Optional file containing tokens. Supports one token per line or a JSON string array.',
    )
    parser.add_argument('--pad-token', type=str, default=None, help='Optional pad token.')
    parser.add_argument('--eos-token', type=str, default=None, help='Optional eos token.')
    parser.add_argument('--bos-token', type=str, default=None, help='Optional bos token.')
    parser.add_argument('--unk-token', type=str, default=None, help='Optional unk token.')
    parser.add_argument(
        '--resize-model',
        action='store_true',
        help='Also resize the language-model token embeddings and save model weights.',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Pass trust_remote_code=True when loading the tokenizer.',
    )
    return parser.parse_args()


def load_tokens_from_file(path: Path) -> list[str]:
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        return []

    if raw.startswith('['):
        data = json.loads(raw)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError('--token-file JSON must be a list of strings.')
        return data

    return [line.strip() for line in raw.splitlines() if line.strip()]


def dedupe_keep_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def initialize_new_embeddings_with_mean(model: InternVLChatModel, num_new_tokens: int) -> None:
    if num_new_tokens <= 0:
        return

    input_embeddings = model.language_model.get_input_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_embeddings_avg

    output_layer = model.language_model.get_output_embeddings()
    if output_layer is not None:
        output_embeddings = output_layer.weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint_path.resolve()
    output_path = args.output_path.resolve() if args.output_path else checkpoint_path.parent / f'{checkpoint_path.name}_with_special_tokens'

    tokens = list(args.token)
    if args.token_file is not None:
        tokens.extend(load_tokens_from_file(args.token_file.resolve()))
    tokens = dedupe_keep_order(tokens)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )

    num_added_tokens = 0
    if tokens:
        num_added_tokens += tokenizer.add_special_tokens({'additional_special_tokens': tokens})

    special_token_updates = {}
    if args.pad_token is not None:
        special_token_updates['pad_token'] = args.pad_token
    if args.eos_token is not None:
        special_token_updates['eos_token'] = args.eos_token
    if args.bos_token is not None:
        special_token_updates['bos_token'] = args.bos_token
    if args.unk_token is not None:
        special_token_updates['unk_token'] = args.unk_token

    if special_token_updates:
        num_added_tokens += tokenizer.add_special_tokens(special_token_updates)

    output_path.mkdir(parents=True, exist_ok=True)

    if args.resize_model:
        model = InternVLChatModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        old_vocab_size = model.language_model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(tokenizer)

        if new_vocab_size != old_vocab_size:
            model.language_model.resize_token_embeddings(new_vocab_size)
            initialize_new_embeddings_with_mean(model, new_vocab_size - old_vocab_size)
            model.language_model.config.vocab_size = new_vocab_size
            model.config.llm_config.vocab_size = new_vocab_size

        model.save_pretrained(output_path)

    tokenizer.save_pretrained(output_path)

    print(f'checkpoint_path: {checkpoint_path}')
    print(f'output_path: {output_path}')
    print(f'num_added_tokens: {num_added_tokens}')
    print(f'final_vocab_size: {len(tokenizer)}')

    for token in tokens:
        print(f'{token}: {tokenizer.convert_tokens_to_ids(token)}')

    for field_name in ('pad_token', 'eos_token', 'bos_token', 'unk_token'):
        token_value = getattr(tokenizer, field_name, None)
        token_id = getattr(tokenizer, f'{field_name}_id', None)
        if token_value is not None:
            print(f'{field_name}: {token_value} ({token_id})')


if __name__ == '__main__':
    main()
