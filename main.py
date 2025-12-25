#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.model import MMGNet, SGFN
from src.utils.config import Config
from utils import util
import torch
import argparse

def main():
    config = load_config()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)
    # init device
    # config.DEVICE = 'cuda'

    # init continue learning mode
    is_continue_learning = hasattr(config, "continue_learning_mode") and config.continue_learning_mode != 'none'
    # import pdb;pdb.set_trace()
    
    if is_continue_learning:
        task_splits = {}
        task_splits = {
            "0": config.task_0_rel_name,
            "1": config.task_1_rel_name,
            "2": config.task_2_rel_name,
            "3": config.task_3_rel_name,
            "4": config.task_4_rel_name
        }
    # task_splits = config.task_class_split.get(config.continue_learning_mode) if is_continue_learning else None
    # ➤ 普通 SGG 模式
    if not is_continue_learning:
        if config.MODE == 'eval':
            config.EVAL = True
        model = MMGNet(config)
        # save_config(config, model)
        #检查一下是否有已经训练过的的模型，如果有的话，读取其epoch和iter，然后从该epoch和iter开始训

        normal_ckp_path = os.path.join(config.PATH, model.model_name+"_"+model.exp, "ckp", "result.pth")
        best_ckp_path = os.path.join(config.PATH, model.model_name+"_"+model.exp, "ckp", "result_best.pth")
        if os.path.exists(normal_ckp_path) or os.path.exists(best_ckp_path):
            #后缀可能是.pth或者_best.pth
            if os.path.exists(normal_ckp_path):
                result = torch.load(normal_ckp_path)
            else:
                result = torch.load(best_ckp_path)
            model.epoch = result['epoch']#读取epoch
            model.iteration = result['iteration']#
            print(f'✅ Loading pre-trained model...')
            print(f'从epoch {model.epoch}和iteration {model.iteration}开始训练')
            model.load(best=False)#测试的时候肯定是从best测试，但是训练的时候肯定是从iteration开始训练
        else:
            print('没有找到预训练的模型，从0开始训练')
            model.model.epoch = 1
            model.model.iteration = 0
        # import pdb;pdb.set_trace()
        run_training_and_eval(model)
        # return
    else:
    # ➤ 持续学习模式
        for task_id in sorted(task_splits.keys()):
            config = load_config()
            print(f"\n🚀 Starting Task {task_id} in mode {config.continue_learning_mode}...\n")
            # import pdb;pdb.set_trace()
            model = MMGNet(config, int(task_id))
            # import pdb;pdb.set_trace()
            # save_config(config, model)
            if task_id != '0':
                # try:
                #这个地方不能是try，而是必须加载成功，否则会报错
                model.load_continue_learning(task_id=int(task_id)-1)  # 加载前一阶段模型参数，如果有
                # except:
                # print(f'[WARN] Unable to load model for task {task_id}, maybe first task.')
            else:
                print('Start training from original model')
            #修改前一阶段的分类器，权重不变，但是分类器要改变
            #检查是否有该阶段预训练的模型，如果有，跳过训练，直接验证
            # import pdb;pdb.set_trace()
            if os.path.exists(os.path.join(config.PATH, model.model_name+"_"+model.exp, "ckp", f"task_{task_id}", f"result.pth")):
                print(f'✅ Loading pre-trained model for task {task_id}...')
                print(f'跳过{task_id}阶段模型训练')
                # model.train(task_id=task_id)
                model.load_continue_learning(task_id=int(task_id))
            else:
                model.train(task_id=task_id)
                # pass
            #验证第task_id阶段的模型的效果
            config.EVAL = True
            print(f'✅ Validating Task {task_id}...')
            #加载本阶段的模型，但是同样也没加载优化器
            model.load_continue_learning(task_id=task_id)
            #验证本阶段的模型
            # if task_id == '0' or task_id == '1':
            #     pass
            # else:
            model.validation(test_triplet=True)
            #验证全部阶段的模型效果
            if task_id == '0':
                pass
            else:
                print(f'✅ Validating All Tasks...')
                model.validation(task_id=task_id, is_all_task=True, test_triplet=True)
            config.EVAL = False
def save_config(config, model):
    # import pdb;pdb.set_trace()
    # if config.continue_learning_mode is None:
    save_path = os.path.join(config.PATH, model.model_name+'_'+model.exp, 'config')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'config.json')
    if not os.path.exists(save_path):
        # import pdb;pdb.set_trace()
        with open(save_path, 'w') as f:
            json.dump(config, f)

def run_training_and_eval(model):
    if model.config.MODE == 'eval':
        print('start validation...')
        model.load(best=True)
        model.validation(test_triplet=True)
        return

    print('\nstart training...\n')
    model.train()

    model.config.EVAL = True
    print('start validation...')
    model.load()
    model.validation(test_triplet=True)
def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--SGFormer_attn_structure', type=str, default='1SA+1CA+9SA')
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')
    parser.add_argument('--continue_learning_mode', type=str, default=None)
    parser.add_argument('--continue_learning_method', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--obj_label_path', type=str, default=None)
    parser.add_argument('--rel_label_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_VLM_description', action='store_true', help='use VLM description')
    parser.add_argument('--dataset_annotation_type', type=str, default="160O26R", choices=["20O8R", "160O26R"])
    parser.add_argument('--use_triplet', action='store_true', help='use triplet')
    parser.add_argument('--task_type', type=str, default='SGCls', choices=['SGCls', 'PredCls'])
    args = parser.parse_args()
    
    #读取config文件
    # data = json.load(open(args.config, 'r'))
    # import pdb;pdb.set_trace()
    config_path = os.path.abspath(args.config)
    
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.exp = args.exp
    config.NAME = args.model_name
    config.DEBUG = args.DEBUG
    config.dataset.root = args.root
    config.inference_num = 0
    config.MODEL.SGFormer_attn_structure = args.SGFormer_attn_structure
    config.MODEL.obj_label_path = args.obj_label_path
    config.MODEL.rel_label_path = args.rel_label_path
    config.WORKERS = args.num_workers
    config.use_VLM_description = args.use_VLM_description
    config.use_triplet = args.use_triplet
    # import pdb;pdb.set_trace()
    if config.DEBUG:
        config.MAX_EPOCHES = 6
        config.VALID_INTERVAL = 3
    config.continue_learning_method = args.continue_learning_method
    config.continue_learning_mode = args.continue_learning_mode
    config.dataset_annotation_type = args.dataset_annotation_type
    config.task_type = args.task_type
    return config

if __name__ == '__main__':
    main()
