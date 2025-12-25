#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.model import MMGNet
from src.utils.config import Config
from utils import util

def inference(config):
    """
    Perform inference for standard SGG task (non-continual learning mode)
    
    Args:
        config_path (str): Path to config file
        model_path (str, optional): Specific model checkpoint path. If None, will use best checkpoint.
    """
    # Load config
    config.MODE = 'eval'  # Set to evaluation mode
    config.EVAL = True
    
    # Set random seed and device
    util.set_random_seed(config.SEED)
    if torch.cuda.is_available():
        config.DEVICE = 'cuda'
    # Initialize model
    model = MMGNet(config)
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {config.model_path}")
    print(f"✅ Loading model weights from {config.model_path}")

    model.load(best=True)
    
    # Perform inference/validation
    print("🚀 Starting inference...")
    results = model.inference(test_triplet=config.use_triplet)
    
    print("\n📊 Inference Results:")
    print(json.dumps(results, indent=2))
    
    return results
def load_config():
    parser = argparse.ArgumentParser(description='SGG Inference Script')
    ##这里的config应该是一个基座config
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--model_name', type=str, default='SGFormer',
                       help='Model name')
    parser.add_argument('--exp', type=str, default='exp_20',
                       help='Experiment name')
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')
    parser.add_argument('--CKPT_PATH', type=str, default='/data_2/lcs/tpami2025/workdir',
                       help='Path to workdir')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers')
    parser.add_argument('--inference_num', type=int, default=37,
                       help='Number of inference')
    parser.add_argument('--SGFormer_attn_structure', type=str, default='1SA+1CA+9SA',
                       help='SGFormer attention structure')
    parser.add_argument('--root', type=str, default='/home/lcs/tpami2025/data/3DSSG_Sub_Not_Split',
                       help='Root path')
    parser.add_argument('--use_triplet', action='store_true', help='Use triplet')
    parser.add_argument('--obj_label_path', type=str, default='/home/lcs/tpami2025/data/3DSSG_subset/classes.txt',
                       help='Object label path')
    parser.add_argument('--rel_label_path', type=str, default='/home/lcs/tpami2025/data/3DSSG_subset/relations.txt',
                       help='Relation label path')
    parser.add_argument('--use_VLM_description', action='store_true', help='Use VLM description')
    parser.add_argument('--dataset_annotation_type', type=str, default="160O26R", choices=["20O8R", "160O26R"])
    parser.add_argument('--task_type', type=str, default='SGCls', choices=['SGCls', 'PredCls'])
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    print(config_path)
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    config = Config(config_path)
    config.exp = args.exp
    config.NAME = args.model_name
    config.DEBUG = args.DEBUG
    config.CKPT_PATH = args.CKPT_PATH
    config.WORKERS = args.num_workers
    config.inference_num = args.inference_num
    config.MODEL.SGFormer_attn_structure = args.SGFormer_attn_structure
    config.dataset.root = args.root
    config.MODEL.obj_label_path = args.obj_label_path
    config.MODEL.rel_label_path = args.rel_label_path
    config.model_path = f'{config.CKPT_PATH}/{config.NAME}_{config.exp}/ckp'
    config.use_triplet = args.use_triplet
    config.use_VLM_description = args.use_VLM_description
    config.dataset_annotation_type = args.dataset_annotation_type
    config.task_type = args.task_type
    # import pdb;pdb.set_trace()
    if config.DEBUG:
        config.MAX_EPOCHES = 6
        config.VALID_INTERVAL = 3
    return config
if __name__ == '__main__':
    config = load_config()
    inference(config)