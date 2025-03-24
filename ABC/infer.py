import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader

from config import get_default_args
from dataset_utils import get_tokenized_dataset_infer, get_dataset


def infer(args, model_path, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
    test_dataset = get_tokenized_dataset_infer(args, tokenizer, 'test')
    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=DefaultDataCollator())
    result = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch['input_ids'].to("cuda")
        output_ids = model.generate(input_ids, max_new_tokens=512, num_beams=5)
        output_text_list = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(output_text_list)
        result.extend(output_text_list)
    return result