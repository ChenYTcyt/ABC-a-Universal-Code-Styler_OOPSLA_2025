import json
from datasets import Dataset
from transformers import AutoTokenizer

from config import get_default_args


def get_dataset(args, split='train'):
    file_path = f'{args.dataset_path}/{args.dataset_name}/{split}.json'
    dataset = Dataset.from_json(file_path)
    return dataset


def get_tokenized_dataset(args, tokenizer, split='train'):
    dataset = get_dataset(args, split)
    def tokenize_function(example):
        if args.instruction:
            src_text = '\n### Instruction ###\n' + example['instruction'] + '\n### Source Code ###\n' + example['input'] + '\n### Converted code ###\n'
        else:
            src_text = '\n### Source Code ###\n' + example['input'] + '\n### Converted code ###\n'
        tgt_text =  example['output']
        src_encode = tokenizer(src_text)
        tgt_encode = tokenizer(tgt_text)
        input_ids = []
        input_ids.extend(src_encode['input_ids'])
        input_ids.extend(tgt_encode['input_ids'])
        src_mask = [1] * len(src_encode['input_ids'])
        tgt_mask = [0] * len(tgt_encode['input_ids'])
        pad_len = 1024 - len(input_ids)
        if pad_len < 0:
            input_ids_mask = []
            input_ids_mask.extend(src_mask)
            input_ids_mask.extend(tgt_mask)
            input_ids_mask = input_ids_mask[:1024]
        else:
            pad_ids = [tokenizer.pad_token_id] * pad_len
            pad_ids.extend(input_ids)
            input_ids = pad_ids
            input_ids_mask = [tokenizer.pad_token_id] * pad_len
            input_ids_mask.extend(src_mask)
            input_ids_mask.extend(tgt_mask)
        input_ids = input_ids[:1024]
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids_mask,
            'labels': input_ids
        }
    fields = ['instruction', 'input', 'output']
    dataset = dataset.map(tokenize_function).remove_columns(fields) 
    return dataset


def get_tokenized_dataset_exemplars_infer(args, tokenizer, split='test'):
    dataset = get_dataset(args, split)
    src_max_length = compute_max_length(args, tokenizer, split) + 2
    print(f"src_max_length: {src_max_length}")
    def tokenize_function(example):
        src_text = example['input']
        src_encode = tokenizer(src_text, padding="max_length", truncation=True, max_length=src_max_length)
        tgt_encode = tokenizer(example['output'], padding="max_length", truncation=True, max_length=512)
        return {
            'input_ids': src_encode['input_ids'],
            'attention_mask': src_encode['attention_mask'],
            'labels': tgt_encode['input_ids']
        }
    fields = ['input', 'output']
    dataset = dataset.map(tokenize_function).remove_columns(fields)
    return dataset


def compute_max_length(args, tokenizer, split):
    dataset = get_dataset(args, split)
    max_length = 0
    for example in dataset:
        text = example['input']
        encode = tokenizer(text)
        max_length = max(max_length, len(encode['input_ids']))
    return max_length
    