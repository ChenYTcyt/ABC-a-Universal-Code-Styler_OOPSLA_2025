import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def get_default_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--visible_device', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--peft', type=str)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--instruction', type=str)
    parser.add_argument('--infer_model_path', type=str)
    parser.add_argument('--exemplars', type=str)
    
    args = parser.parse_args()
    args.model_path = f'{args.model_path}/{args.model_name}'
    args.peft = str2bool(args.peft)
    args.instruction = str2bool(args.instruction)
    args.exemplars = str2bool(args.exemplars)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device

    return args