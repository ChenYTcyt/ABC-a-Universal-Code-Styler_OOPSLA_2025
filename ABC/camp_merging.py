import os
import torch
import shutil
import numpy as np
from itertools import combinations


def copy_model(model_path, ties_path):
    files_to_copy = [
        "adapter_config.json",
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json"
    ]
    os.makedirs(ties_path, exist_ok=True)
    for file_name in files_to_copy:
        source_file = os.path.join(model_path, file_name)
        destination_file = os.path.join(ties_path, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        else:
            print(f"File not found: {source_file}")

def trim_weights(model_weights, k_percent, scale_factor):
    trimmed_weights = {}
    for key, value in model_weights.items():
        flat_values = value.view(-1)
        threshold = torch.quantile(flat_values.abs(), 1 - k_percent / 100.0)
        mask = (flat_values.abs() >= threshold).view(value.shape)
        trimmed_values = torch.where(mask, value * scale_factor, torch.tensor(0.0, device=value.device))
        trimmed_weights[key] = trimmed_values
    return trimmed_weights

def elect_sign(models):
    elected_sign = {}
    keys = models[0].keys()
    for key in keys:
        stacked_weights = torch.stack([model[key] for model in models], dim=0)
        elected_sign[key] = torch.sign(stacked_weights.sum(dim=0))
    return elected_sign

def merge(models, elected_sign):
    merged_weights = {}
    keys = models[0].keys()
    for key in keys:
        stacked_weights = torch.stack([model[key] for model in models], dim=0)
        sign_mask = (torch.sign(stacked_weights) == elected_sign[key])
        sign_mask = sign_mask.float()
        valid_weights = stacked_weights * sign_mask
        count = sign_mask.sum(dim=0).clamp(min=1)
        disjoint_mean = valid_weights.sum(dim=0) / count
        merged_weights[key] = disjoint_mean
    return merged_weights

def camp_merging_procedure(model_paths, k_percent, scale_factor=1):
    models = [torch.load(path, weights_only=True) for path in model_paths]
    trimmed_models = [trim_weights(model, k_percent, scale_factor) for model in models]
    elected_sign = elect_sign(trimmed_models)
    merged_model = merge(trimmed_models, elected_sign)
    return merged_model

def parse_string(s):
    parts = s.split('_')
    L = int(parts[0][1:])
    S = int(parts[1][1:])
    V = int(parts[2][1:])
    return L, S, V

def add_strings(str_list):
    L_sum, S_sum, V_sum = 0, 0, 0
    for s in str_list:
        L, S, V = parse_string(s)
        L_sum += L
        S_sum += S
        V_sum += V
    return f"L{L_sum}_S{S_sum}_V{V_sum}"

def generate_combinations(strings, t):
    combs = combinations(strings, t)
    results = []
    for comb in combs:
        summed = add_strings(comb)
        results.append((comb, summed))
    return results

def camp_merging(pretrained_model:str, pretrained_model_path:str, all_model_vectors:dict, t:int, k_percent:int, scale_factor:float, save_path:str):
    combinations_with_sum = generate_combinations(list(all_model_vectors.keys()), t)
    for comb, sum in combinations_with_sum:
        model_vectors = [all_model_vectors[key] for key in comb]
        merged_weights = camp_merging_procedure(model_vectors, k_percent, scale_factor)
        os.makedirs(save_path, exist_ok=True)
        torch.save(merged_weights, os.path.join(save_path, 'adapter_model.bin'))
        copy_model(pretrained_model_path, save_path)