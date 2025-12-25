if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import copy
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset
from src.model.SGFN_MMG.model import Mmgnet
from src.model.SGFN_MMG.baseline_sgfn import SGFN
from src.model.SGFN_MMG.baseline_sgpn import SGPN
from src.model.SGGpoint.model import MMEdgeGCN
from src.model.SGFormer.baeline_sgformer import SGFormer
from src.utils import op_utils
from src.utils.eva_utils_acc import get_mean_recall
from src.utils.eva_utils_acc import get_zero_shot_recall
from src.model.SGFormer.per_class_tracker import PerClassAccuracyTracker
import openpyxl
import time
from src.model.SGFormer.feature_collector import FeatureCollector
class MMGNet():
    def __init__(self, config, task_id=None):
        self.config = config
        self.model_name = self.config.NAME
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.inference_num = config.inference_num
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        self.task_id = task_id
        # self.epoch = 1
        # self.iteration = 0
        ''' Build dataset '''
        dataset = None
        if config.MODE  == 'train' or config.MODE  == 'eval':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL,
                                               task_id=self.task_id,
                                               continue_learning_mode=config.continue_learning_mode,
                                               is_all_task=False)
            self.dataset_train.__getitem__(0)
                
        if config.MODE  == 'train' or config.MODE  == 'trace' or config.MODE  == 'eval':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL,
                                      task_id=self.task_id,
                                      continue_learning_mode=config.continue_learning_mode,
                                      is_all_task=False)
            dataset = self.dataset_valid
        #这个地方需要改，因为持续学习，所以需要根据task_id来获取obj_name和rel_name
        #普通的时候，按这个来弄即可，但是持续学习会改变obj_name和rel_name，不能简单的按原来的满的来创建模型
        if config.continue_learning_mode == "S1":
            #这里用于构造一个截止目前为止所有rel的数据集
            self.dataset_valid_all_task = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL,
                                      task_id=self.task_id,
                                      continue_learning_mode=config.continue_learning_mode,
                                      is_all_task=True)

        # if config.continue_learning_mode is not None:
        #     num_obj_class, num_rel_class = len(self.dataset_train.allowed_obj_name), len(self.dataset_train.allowed_rel_name)
        #     self.num_obj_class = num_obj_class
        #     self.num_rel_class = num_rel_class
        # else:
        num_obj_class = len(self.dataset_valid.classNames)   
        num_rel_class = len(self.dataset_valid.relationNames)
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class
        
        self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
        self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)
        # import pdb;pdb.set_trace()
        ''' Build Model '''
        if self.model_name == 'Mmgnet':
            self.model = Mmgnet(self.config, num_obj_class, num_rel_class, task_id=self.task_id).to('cuda')
        elif self.model_name == 'SGFN':
            self.model = SGFN(self.config, num_obj_class, num_rel_class).to('cuda')
        elif self.model_name == 'MMEdgeGCN':
            self.model = MMEdgeGCN(self.config,
                                   num_node_in_embeddings=80,
                                   num_edge_in_embeddings=80,
                                   AttnEdgeFlag=True,
                                   AttnNodeFlag=True,
                                   num_heads=4).to('cuda')
        elif self.model_name == 'SGFormer':
            self.model = SGFormer(self.config, num_obj_class, num_rel_class, task_id=self.task_id).to('cuda')
        elif self.model_name == 'SGPN':
            self.model = SGPN(self.config, num_obj_class, num_rel_class).to('cuda')
        else:
            raise ValueError(f"Model name {self.model_name} not supported")
        self.samples_path = os.path.join(config.PATH, self.model_name, self.exp,  'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, self.exp, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, self.exp, 'traced')
        # self.writter = None
        # import pdb;pdb.set_trace()
        
        if self.config.continue_learning_mode == 'none':
            if self.config.EVAL:
                self.pth_log = os.path.join(config.PATH, self.model_name+"_"+self.exp, "eval_logs")
            else:
                self.pth_log = os.path.join(config.PATH, self.model_name+"_"+self.exp, "logs")
                self.train_local_task_writter = SummaryWriter(os.path.join(self.pth_log, "train_local"))
            self.valid_local_task_writter = SummaryWriter(os.path.join(self.pth_log, "valid_local"))
        else:
            if self.config.EVAL:
                #只有在专门测试的时候才会触发这一点
                self.pth_log = os.path.join(config.PATH, self.model_name+"_"+self.exp, "eval_logs")
                self.train_local_task_writter = None
            else:
            #在持续学习的setting常规训练下，需要记录三个信息
                self.pth_log = os.path.join(config.PATH, self.model_name+"_"+self.exp, "logs")
                self.train_local_task_writter = SummaryWriter(os.path.join(self.pth_log, "train_local", 'task_'+str(self.task_id)))
            self.valid_local_task_writter = SummaryWriter(os.path.join(self.pth_log, "valid_local", 'task_'+str(self.task_id)))
            self.valid_all_task_writter = SummaryWriter(os.path.join(self.pth_log, "valid_all", 'task_'+str(self.task_id)))
        # import pdb;pdb.set_trace()
    def load(self, best=False):
        return self.model.load(best)
    def load_continue_learning(self, task_id):
        return self.model.load_continue_learning(task_id)
    def to_cuda(self, device):
        self.model.to(device)
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list
    
    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids= \
            self.cuda(obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list
    def train(self, task_id=None):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=self.config.WORKERS,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )

        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it', 'Misc/lr'])
                
        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
                   
        if self.mconfig.use_pretrain != "":
            self.model.load_pretrain_model(self.mconfig.use_pretrain, is_freeze=True)
        
        ''' Train '''
        for epoch in range(self.config.MAX_EPOCHES):

            print('\n\nTraining epoch: %d' % self.model.epoch)
            
            for items in loader:
                self.model.train()
                ''' get data '''
                obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list = self.data_processing_train(items)
                # import pdb;pdb.set_trace()
                #收集所有的gt_class和gt_rel_cls,并保存在文件中
                #
                # import pdb;pdb.set_trace()
                logs = self.model.process_train(obj_points, obj_2d_feats, vlm_description_embedding,gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, with_log=True,
                                                weights_obj=self.dataset_train.w_cls_obj, 
                                                weights_rel=self.dataset_train.w_cls_rel,
                                                ignore_none_rel = False,
                                                )
                
                iteration = self.model.iteration
                logs += [
                    ("Misc/epo", int(self.model.epoch)),
                    ("Misc/it", int(iteration)),
                    ("lr", self.model.lr_scheduler.get_last_lr()[0])
                ]
                
                progbar.add(1, values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                #训练的时候，隔一会儿就往trian local的writter中写入信息
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log_local_task(logs, iteration)
                if self.model.iteration >= self.max_iteration:
                    break

            progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            # import pdb;pdb.set_trace()
            if ('VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                #在训练过程中，我们也要隔一会儿测一下，然后这个测的不会写入eval_logs，但是会写入train_local
                print('start validation...')
                #到了需要保存的时候才保存
                rel_acc_val = self.validation(test_triplet=True)
                #这个地方的valid还记录吗

                self.model.eva_res = rel_acc_val#将验证结果写入eva_res
                if self.config.continue_learning_mode == 'none':
                    self.save()
                else:
                    self.save(task_id)
            
            self.model.epoch += 1
        
        # 训练完成后，保存缺失的VLM文件路径
        if hasattr(self.config, 'use_VLM_description') and self.config.use_VLM_description:
            if hasattr(self.dataset_train, 'save_missing_vlm_files'):
                output_path = os.path.join(self.config.PATH, self.model_name+"_"+self.exp, "missing_vlm_files_train.txt")
                self.dataset_train.save_missing_vlm_files(output_path)

    def cuda(self, *args):
        return [item.to('cuda') for item in args]
    
    def log_local_task(self, logs, iteration):
        '''
        一般在训练的时候使用，记录本阶段训练的指标
        '''
        if self.config.EVAL:
            if self.valid_local_task_writter is not None:
                for i in logs:
                    if not i[0].startswith('Misc'):
                        # print(f"Writing {i[0]} {i[1]} {iteration}")
                        self.valid_local_task_writter.add_scalar(i[0], i[1], iteration)
                        self.valid_local_task_writter.flush()
            
        else:
            #只要不是在推理，就往train里写
            if self.train_local_task_writter is not None:
                for i in logs:
                    if not i[0].startswith('Misc'):
                        self.train_local_task_writter.add_scalar(i[0], i[1], iteration)
                        self.train_local_task_writter.flush()
    def log_all_task(self, logs, iteration):
        '''
        一般在验证的时候使用，记录所有阶段的指标
        '''
        if self.valid_all_task_writter is not None and self.config.EVAL:
            for i in logs:
                if not i[0].startswith('Misc'):
                    # import pdb;pdb.set_trace()
                    self.valid_all_task_writter.add_scalar(i[0], i[1], iteration)
                    self.valid_all_task_writter.flush()
                    
    def save(self, task_id=None):
        if task_id is not None:
            self.model.save_continue_learning(task_id)
        else:
            self.model.save()
#    def save_continue_learning(self, task_id):
#         self.model.save_continue_learning(task_id )
    def inference(self, debug_mode = False,task_id=None,is_all_task=False, is_testing_FWT=False,test_triplet=False):
        '''
        本函数和validation函数基本一致,但是是一个纯单个sample推理的函数,所以主要是用于打印推理出的结果
        '''
        #这里要逐渐去掉is_all_task的限制，非持续学习和持续学习的代码要逐渐分开
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )
        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])
        gt_class_list = []  # 收集gt_class数据用于计算obj mean
        gt_rel_cls_list = []
        # 1. 初始化tracker（只需要一次）
        # import pdb;pdb.set_trace()
        accuracy_tracker = PerClassAccuracyTracker(num_rel_classes=len(self.dataset_valid.relationNames), num_obj_classes=len(self.dataset_valid.classNames))
        feature_collector = FeatureCollector()
        total_obj_class_dict = {}
        # total_rel_class_dict = {}
        # for i in range(len(self.dataset_valid.relationNames)):
        #     total_rel_class_dict[i] = 0
        for i in range(len(self.dataset_valid.classNames)):
            total_obj_class_dict[i] = 0
        #这个是从0开始还是从1开始的？应该是从0开始
        total_rel_class_tensor = torch.zeros(len(self.dataset_valid.relationNames), device='cuda:0')
        total_node_num = 0
        total_scene_num = 0
        max_node_num = 0
        min_node_num = 1e9
        # import pdb;pdb.set_trace()
        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            print(f"processing scene {i}")
            # import pdb;pdb.set_trace()
            obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list = self.data_processing_val(items)
            # import pdb;pdb.set_trace()
            # print(i)
            for m in range(gt_rel_cls.shape[0]):
                total_rel_class_tensor += gt_rel_cls[m]
            for m in range(gt_class.shape[0]):
                total_obj_class_dict[int(gt_class[m].cpu())] += 1
            total_node_num += gt_class.shape[0]
            total_scene_num += 1
            max_node_num = max(max_node_num, gt_class.shape[0])
            min_node_num = min(min_node_num, gt_class.shape[0])
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                        = self.model.process_val(obj_points, obj_2d_feats, vlm_description_embedding, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=test_triplet, is_all_task=is_all_task, \
                            accuracy_tracker=accuracy_tracker, feature_collector=feature_collector)
                        
            #     # import pdb;pdb.set_trace()
            #     ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            # 收集gt_class数据
            # import pdb;pdb.set_trace()
            gt_class_list.extend(gt_class.cpu().numpy())
            gt_rel_cls_list.extend(gt_rel_cls.cpu().numpy())
            if test_triplet:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)
            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@3/obj_cls_acc", (topk_obj_list <= 3).sum() * 100 / len(topk_obj_list)),
                    ("Acc@3/obj_cls_2d_acc", (topk_obj_2d_list <= 3).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),]
            if test_triplet:
                logs.extend([
                        ("Acc@1/triplet_acc", (topk_triplet_list <= 1).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@3/triplet_acc", (topk_triplet_list <= 3).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@5/triplet_acc", (topk_triplet_list <= 5).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                        ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),
                    ])
            else:
                pass
                    

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
        # 检查是否有T-SNE文件夹，没有则创建
        # if not os.path.exists(f"T-SNE"):
        #     os.makedirs(f"T-SNE")
        # feature_collector.save_to_file(f"T-SNE/{self.model_name}++_feature_collector_{self.config.dataset_annotation_type}.pt")
        # # feature_collector.load_from_file(f"{self.model_name}_feature_collector_data.pt")
        # # print(f"Loaded feature collector data from {self.model_name}_feature_collector_data.pt")
        # feature_collector.plot_tsne_clear(num_classes=len(self.dataset_valid.classNames), save_path=f"T-SNE/{self.model_name}++_tsne_node_features_train_{self.config.dataset_annotation_type}.png", perplexity=30, class_name=self.dataset_valid.classNames)
        
        
        #打印每个rel label的次数
        # print("Rel label count:")
        # print("--------------------------------")
        # print("total rel label count: ", total_rel_class_tensor.sum())
        # for i in range(len(self.dataset_valid.relationNames)):
        #     print(f"Rel label {self.dataset_valid.relationNames[i]} count: {total_rel_class_tensor[i]}, frequency: {total_rel_class_tensor[i]/total_rel_class_tensor.sum()}")
        # #打印每个obj label的次数
        # print("Obj label count:")
        # print("--------------------------------")
        # print("total obj label count: ", sum(total_obj_class_dict.values()))
        # #将obj和rel的写成excel表格，包括name,  count, frequency
        # import pandas as pd

        # === 处理对象类别统计 ===
        obj_rows = []
        # for i in range(len(self.dataset_valid.classNames)):
        #     obj_rows.append({
        #         'name': self.dataset_valid.classNames[i],
        #         'count': total_obj_class_dict[i],
        #         'frequency': total_obj_class_dict[i] / sum(total_obj_class_dict.values())
        #     })

        # df_obj = pd.DataFrame(obj_rows)
        # df_obj.to_excel('test_obj_label_count_SGFormer++.xlsx', index=False)

        # # === 处理关系类别统计 ===
        # rel_rows = []
        # for i in range(len(self.dataset_valid.relationNames)):
        #     rel_rows.append({
        #         'name': self.dataset_valid.relationNames[i],
        #         'count': int(total_rel_class_tensor[i].item()),  # 转为 Python 标量
        #         'frequency': float(total_rel_class_tensor[i] / total_rel_class_tensor.sum())
        #     })

        # df_rel = pd.DataFrame(rel_rows)
        # df_rel.to_excel('test_rel_label_count_SGFormer++.xlsx', index=False)

        # # === 打印对象类别信息 ===
        # for i in range(len(self.dataset_valid.classNames)):
        #     freq = total_obj_class_dict[i] / sum(total_obj_class_dict.values())
        #     print(f"Obj label {self.dataset_valid.classNames[i]} count: {total_obj_class_dict[i]}, frequency: {freq:.4f}")
        # #可以算triplet，但是不算mean recall
        # # import pdb;pdb.set_trace()
        # accuracy_tracker.print_comprehensive_results(
        #     relation_names=self.dataset_valid.relationNames,
        #     obj_names=self.dataset_valid.classNames,  # 或者你的物体类别名称
        #     rel_topk=6,
        #     obj_topk=11
        # )

        # 获取具体的指标
        rel_metrics = accuracy_tracker.get_rel_per_class_metrics()
        obj_metrics = accuracy_tracker.get_obj_per_class_metrics()
        overall_metrics = accuracy_tracker.get_overall_accuracy()

        # 访问特定类别的准确率
        for class_id, metrics in obj_metrics.items():
            print(f"Object Class {class_id} Accuracy: {metrics['class_accuracy']:.4f}")
        #将每一类的准确率写入Excel
        #访问特定类别的recall，写入Excel
        # import pdb;pdb.set_trace()
        for class_id, metrics in obj_metrics.items():
            obj_rows.append({
                'name': self.dataset_valid.classNames[class_id],
                'count': total_obj_class_dict[class_id],
                'frequency': total_obj_class_dict[class_id] / sum(total_obj_class_dict.values()),
                "recall": metrics['recall'],
                "recall_at_3": metrics['recall_at_3'],
                "recall_at_5": metrics['recall_at_5'],
            })
        import pandas as pd
        df_obj_metrics = pd.DataFrame(obj_rows)
        import pdb;pdb.set_trace()
        df_obj_metrics.to_excel('test_obj_metrics_SGFormer++_recall_3_5.xlsx', index=False)
        #访问特定rel类别的准确率，写入Excel
        rel_rows = []
        for i in range(len(self.dataset_valid.relationNames)):
            rel_rows.append({
                'name': self.dataset_valid.relationNames[i],
                'count': int(total_rel_class_tensor[i].item()),  # 转为 Python 标量
                'frequency': float(total_rel_class_tensor[i] / total_rel_class_tensor.sum()),
                "recall": rel_metrics[i]['recall'],
                "recall_at_3": rel_metrics[i]['recall_at_3'],
                "recall_at_5": rel_metrics[i]['recall_at_5'],
            })
        df_rel_metrics = pd.DataFrame(rel_rows)
        import pdb;pdb.set_trace()
        df_rel_metrics.to_excel('test_rel_metrics_SGFormer++_recall_3_5.xlsx', index=False)
        # if test_triplet is not None:
        #     cls_matrix_list = np.stack(cls_matrix_list)
        #     sub_scores_list = np.stack(sub_scores_list)
        #     obj_scores_list = np.stack(obj_scores_list)
        #     rel_scores_list = np.stack(rel_scores_list)
        #     mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        #     mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        #     # import pdb;pdb.set_trace()
        #     zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        # import pdb;pdb.set_trace()
        if self.model.config.EVAL and not is_testing_FWT:
            #这个地方怎么
            save_path = os.path.join(self.config.PATH, self.model_name+"_"+self.exp, "result", f"task_{task_id}")
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            f_in = open(os.path.join(self.pth_log, 'result.txt'), 'w')
            # if test_triplet:
            #     np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            #     np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            #     np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            #     np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
        else:
            f_in = None   
        # #这是总的指标，刚刚那个是log的小的指标
        

        #计算per class的准确率
        # class_dict = {}
        # class_acc_dict = {}
        # #首先统计每个类别出现过几次，然后计算每个类别的准确率
        # for i in range(len(gt_class_list)):
        #     if gt_class_list[i] not in class_dict:
        #         class_dict[gt_class_list[i]] = 0
        #     class_dict[gt_class_list[i]] += 1
        #     if gt_class_list[i] not in class_acc_dict:
        #         class_acc_dict[gt_class_list[i]] = 0
        #     if topk_obj_list[i] <= 1:
        #         class_acc_dict[gt_class_list[i]] += 1
        # #计算class_acc_dict
        # class_acc_dict = {k: v / class_dict[k] for k, v in class_acc_dict.items()}
        # print(class_acc_dict)
        # rel_class_dict = {}
        # rel_class_acc_dict = {}
        # import json
        # with open(os.path.join('SGFormer++_obj_class_acc_dict.json'), 'w') as f:
        #     json.dump(class_acc_dict, f)
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_3 = (topk_obj_list <= 3).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_3 = (topk_obj_2d_list <= 3).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        if test_triplet:
            triplet_acc_1 = (topk_triplet_list <= 1).sum() * 100 / len(topk_triplet_list)
            triplet_acc_3 = (topk_triplet_list <= 3).sum() * 100 / len(topk_triplet_list)
            triplet_acc_5 = (topk_triplet_list <= 5).sum() * 100 / len(topk_triplet_list)
            triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
            triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
            triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
            triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)
            obj_acc_mean_1, obj_acc_mean_3, obj_acc_mean_5, obj_acc_mean_10 = self.compute_mean_obj(gt_class_list, topk_obj_list)
            obj_acc_2d_mean_1, obj_acc_2d_mean_3, obj_acc_2d_mean_5, obj_acc_2d_mean_10 = self.compute_mean_obj(gt_class_list, topk_obj_2d_list)
            rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
            rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)
        else:
            triplet_acc_1 = 0
            triplet_acc_3 = 0
            triplet_acc_5 = 0
            triplet_acc_50 = 0
            triplet_acc_2d_50 = 0
            triplet_acc_100 = 0
            triplet_acc_2d_100 = 0
            obj_acc_mean_1 = 0
            obj_acc_mean_3 = 0
            obj_acc_mean_5 = 0
            obj_acc_mean_10 = 0
            obj_acc_2d_mean_1 = 0
            obj_acc_2d_mean_3 = 0
            obj_acc_2d_mean_5 = 0
            obj_acc_2d_mean_10 = 0
            rel_acc_mean_1 = 0
            rel_acc_mean_3 = 0
            rel_acc_mean_5 = 0
            rel_acc_2d_mean_1 = 0
            rel_acc_2d_mean_3 = 0
            rel_acc_2d_mean_5 = 0


        # import pdb;pdb.set_trace()
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@3  : {obj_acc_3}", file=f_in)
        print(f"Eval: 2d obj Acc@3: {obj_acc_2d_3}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d mean obj Acc@1  : {obj_acc_mean_1}", file=f_in)   
        print(f"Eval: 3d mean obj Acc@3  : {obj_acc_mean_3}", file=f_in)
        print(f"Eval: 3d mean obj Acc@5  : {obj_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d mean obj Acc@1: {obj_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 2d mean obj Acc@5: {obj_acc_2d_mean_5}", file=f_in)
        print(f"Eval: 3d mean obj Acc@10  : {obj_acc_mean_10}", file=f_in) 
        print(f"Eval: 2d mean obj Acc@10: {obj_acc_2d_mean_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in)
        print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)    
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        if test_triplet:
            print(f"Eval: 3d triplet Acc@1 : {triplet_acc_1}", file=f_in)
            print(f"Eval: 3d triplet Acc@3 : {triplet_acc_3}", file=f_in)
            print(f"Eval: 3d triplet Acc@5 : {triplet_acc_5}", file=f_in)
            # print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
            # print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
            # print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
            # print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
            # print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
            # print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
            # print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
            # print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
            # print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
            # print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)



        # if self.model.config.EVAL and not is_testing_FWT:
        #     f_in.close()
        
        # logs = [("Acc@1/obj_cls_acc", obj_acc_1),
        #         ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
        #         ("Acc@1/obj_cls_acc_mean", obj_acc_mean_1),
        #         ("Acc@1/obj_2d_cls_acc_mean", obj_acc_2d_mean_1),
        #         ("Acc@5/obj_cls_acc", obj_acc_5),
        #         ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
        #         ("Acc@5/obj_cls_acc_mean", obj_acc_mean_5),
        #         ("Acc@5/obj_2d_cls_acc_mean", obj_acc_2d_mean_5),
        #         ("Acc@10/obj_cls_acc", obj_acc_10),
        #         ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
        #         ("Acc@10/obj_cls_acc_mean", obj_acc_mean_10),
        #         ("Acc@10/obj_2d_cls_acc_mean", obj_acc_2d_mean_10),
        #         ("Acc@1/rel_cls_acc", rel_acc_1),
        #         ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
        #         ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
        #         ("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
        #         ("Acc@3/rel_cls_acc", rel_acc_3),
        #         ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
        #         ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
        #         ("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
        #         ("Acc@5/rel_cls_acc", rel_acc_5),
        #         ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
        #         ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
        #         ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
        #         ("Acc@50/triplet_acc", triplet_acc_50),
        #         ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
        #         ("Acc@100/triplet_acc", triplet_acc_100),
        #         ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
        #         # ("mean_recall@50", mean_recall[0]),
        #         # ("mean_2d_recall@50", mean_recall_2d[0]),
        #         # ("mean_recall@100", mean_recall[1]),
        #         # ("mean_2d_recall@100", mean_recall_2d[1]),
        #         # ("zero_shot_recall@50", zero_shot_recall[0]),
        #         # ("zero_shot_recall@100", zero_shot_recall[1]),
        #         # ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
        #         # ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
        #         # ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
        #         # ("all_zero_shot_recall@100", all_zero_shot_recall[1])
        #         ]
        # if test_triplet:
        #     logs.extend([
        #         ("Acc@50/triplet_mean_recall", mean_recall[0]),
        #         ("Acc@50/triplet_2d_mean_recall", mean_recall_2d[0]),
        #         ("Acc@100/triplet_mean_recall", mean_recall[1]),
        #         ("Acc@100/triplet_2d_mean_recall", mean_recall_2d[1]),
        #         ("Acc@50/zero_shot_recall", zero_shot_recall[0]),
        #         ("Acc@100/zero_shot_recall", zero_shot_recall[1]),
        #         ("Acc@50/non_zero_shot_recall", non_zero_shot_recall[0]),
        #         ("Acc@100/non_zero_shot_recall", non_zero_shot_recall[1]),
        #         ("Acc@50/all_zero_shot_recall", all_zero_shot_recall[0]),
        #         ("Acc@100/all_zero_shot_recall", all_zero_shot_recall[1])
        #     ])
    def validation(self, debug_mode = False,task_id=None,is_all_task=False, is_testing_FWT=False,test_triplet=False):
        if not is_all_task:
            val_loader = CustomDataLoader(
                config = self.config,
                dataset=self.dataset_valid,
                batch_size=1,
                num_workers=self.config.WORKERS,
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn_mmg
            )
            total = len(self.dataset_valid)
        else:
            val_loader = CustomDataLoader(
                config = self.config,
                dataset=self.dataset_valid_all_task,
                batch_size=1,
                num_workers=self.config.WORKERS,
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn_mmg
            )
            total = len(self.dataset_valid_all_task)
        
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])
        gt_class_list = []  # 收集gt_class数据用于计算obj mean

        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, obj_2d_feats, vlm_description_embedding, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id_list = self.data_processing_val(items)            
            
            with torch.no_grad():
                # if self.model.config.EVAL:
                #     top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                #         = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices, use_triplet=True)
                # else:
                # import pdb;pdb.set_trace()
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                        = self.model.process_val(obj_points, obj_2d_feats, vlm_description_embedding, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=test_triplet, is_all_task=is_all_task)
                        
            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            # 收集gt_class数据
            gt_class_list.extend(gt_class.cpu().numpy())
            # print(cls_matrix is None)
            # import pdb;pdb.set_trace()
            if test_triplet:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)
            # else:
            #     #这里不应该是不
            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),]
            if test_triplet:
                logs.extend([
                        ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                        ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                        ("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),
                    ])
            else:
                pass
                    

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
        #可以算triplet，但是不算mean recall
        # import pdb;pdb.set_trace()
        if test_triplet is not None:
            cls_matrix_list = np.stack(cls_matrix_list)
            sub_scores_list = np.stack(sub_scores_list)
            obj_scores_list = np.stack(obj_scores_list)
            rel_scores_list = np.stack(rel_scores_list)
            mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
            mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
            # import pdb;pdb.set_trace()
            zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames, self.config.dataset.root)
        
        if self.model.config.EVAL and not is_testing_FWT:
            #这个地方怎么
            save_path = os.path.join(self.config.PATH, self.model_name+"_"+self.exp, "result", f"task_{task_id}")
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            f_in = open(os.path.join(self.pth_log, 'result.txt'), 'w')
            if test_triplet:
                np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
                np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
                np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
                np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
        else:
            f_in = None   
        #这是总的指标，刚刚那个是log的小的指标
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        if test_triplet:
            triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
            triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
            triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
            triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)
            obj_acc_mean_1, _, obj_acc_mean_5, obj_acc_mean_10 = self.compute_mean_obj(gt_class_list, topk_obj_list)
            obj_acc_2d_mean_1, _, obj_acc_2d_mean_5, obj_acc_2d_mean_10 = self.compute_mean_obj(gt_class_list, topk_obj_2d_list)
            rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
            rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)
        else:
            triplet_acc_50 = 0
            triplet_acc_2d_50 = 0
            triplet_acc_100 = 0
            triplet_acc_2d_100 = 0
            obj_acc_mean_1 = 0
            obj_acc_mean_5 = 0
            obj_acc_mean_10 = 0
            obj_acc_2d_mean_1 = 0
            obj_acc_2d_mean_5 = 0
            obj_acc_2d_mean_10 = 0
            rel_acc_mean_1 = 0
            rel_acc_mean_3 = 0
            rel_acc_mean_5 = 0
            rel_acc_2d_mean_1 = 0
            rel_acc_2d_mean_3 = 0
            rel_acc_2d_mean_5 = 0


     
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d mean obj Acc@1  : {obj_acc_mean_1}", file=f_in)   
        print(f"Eval: 3d mean obj Acc@5  : {obj_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d mean obj Acc@1: {obj_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 2d mean obj Acc@5: {obj_acc_2d_mean_5}", file=f_in)
        print(f"Eval: 3d mean obj Acc@10  : {obj_acc_mean_10}", file=f_in) 
        print(f"Eval: 2d mean obj Acc@10: {obj_acc_2d_mean_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in)
        print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)    
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        if test_triplet:
            print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
            print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
            print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
            print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
            print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
            print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
            print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
            print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
            print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
            print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)



        if self.model.config.EVAL and not is_testing_FWT:
            f_in.close()
        
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@1/obj_cls_acc_mean", obj_acc_mean_1),
                ("Acc@1/obj_2d_cls_acc_mean", obj_acc_2d_mean_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@5/obj_cls_acc_mean", obj_acc_mean_5),
                ("Acc@5/obj_2d_cls_acc_mean", obj_acc_2d_mean_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@10/obj_cls_acc_mean", obj_acc_mean_10),
                ("Acc@10/obj_2d_cls_acc_mean", obj_acc_2d_mean_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                ("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                ("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                # ("mean_recall@50", mean_recall[0]),
                # ("mean_2d_recall@50", mean_recall_2d[0]),
                # ("mean_recall@100", mean_recall[1]),
                # ("mean_2d_recall@100", mean_recall_2d[1]),
                # ("zero_shot_recall@50", zero_shot_recall[0]),
                # ("zero_shot_recall@100", zero_shot_recall[1]),
                # ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                # ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                # ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                # ("all_zero_shot_recall@100", all_zero_shot_recall[1])
                ]
        if test_triplet:
            logs.extend([
                ("Acc@50/triplet_mean_recall", mean_recall[0]),
                ("Acc@50/triplet_2d_mean_recall", mean_recall_2d[0]),
                ("Acc@100/triplet_mean_recall", mean_recall[1]),
                ("Acc@100/triplet_2d_mean_recall", mean_recall_2d[1]),
                ("Acc@50/zero_shot_recall", zero_shot_recall[0]),
                ("Acc@100/zero_shot_recall", zero_shot_recall[1]),
                ("Acc@50/non_zero_shot_recall", non_zero_shot_recall[0]),
                ("Acc@100/non_zero_shot_recall", non_zero_shot_recall[1]),
                ("Acc@50/all_zero_shot_recall", all_zero_shot_recall[0]),
                ("Acc@100/all_zero_shot_recall", all_zero_shot_recall[1])
            ])
        # import pdb;pdb.set_trace()
        # 验证完成后，保存缺失的VLM文件路径
        if hasattr(self.config, 'use_VLM_description') and self.config.use_VLM_description:
            if not is_all_task:
                if hasattr(self.dataset_valid, 'save_missing_vlm_files'):
                    output_path = os.path.join(self.config.PATH, self.model_name+"_"+self.exp, f"missing_vlm_files_validation_task_{task_id if task_id is not None else 'none'}.txt")
                    self.dataset_valid.save_missing_vlm_files(output_path)
            else:
                if hasattr(self.dataset_valid_all_task, 'save_missing_vlm_files'):
                    output_path = os.path.join(self.config.PATH, self.model_name+"_"+self.exp, f"missing_vlm_files_validation_all_task_{task_id if task_id is not None else 'none'}.txt")
                    self.dataset_valid_all_task.save_missing_vlm_files(output_path)
        
        if self.config.continue_learning_mode == 'none':
            self.log_local_task(logs, self.model.iteration)
            self.close_all_writter()
            return rel_acc_1
        else:
            #持续学习才有all和local的区别，
            if not is_testing_FWT:
                if is_all_task:
                    self.log_all_task(logs, self.model.iteration)
                    
                else:
                    self.log_local_task(logs, self.model.iteration)
                self.close_all_writter()
                return rel_acc_1
            else:
                #如果记录FWT，要换出外面去记录，所以这里要返回logs
                self.close_all_writter()
                return logs
    def close_all_writter(self):
        try:
            if self.valid_local_task_writter is not None:
                self.valid_local_task_writter.close()
        except:
            pass
        try:
            if self.train_local_task_writter is not None:
                self.train_local_task_writter.close()
        except:
            pass
        try:
            if self.valid_all_task_writter is not None:
                self.valid_all_task_writter.close()
        except:
            pass
    def compute_mean_obj(self, gt_class_list, topk_obj_list):
        """
        计算物体分类的mean准确率
        gt_class_list: 每个样本的真实类别标签列表
        topk_obj_list: 每个样本的top-k预测结果
        """
        num_classes = len(self.dataset_valid.classNames)
        cls_dict = {}
        for i in range(num_classes):
            cls_dict[i] = []
        
        for idx, gt_class in enumerate(gt_class_list):
            if gt_class != -1:  # 过滤掉无效的样本
                cls_dict[gt_class].append(topk_obj_list[idx])
        
        obj_mean_1, obj_mean_3, obj_mean_5, obj_mean_10 = [], [], [], []
        for i in range(num_classes):
            l = len(cls_dict[i])
            if l > 0:  # 只计算有样本的类别
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                m_10 = (np.array(cls_dict[i]) <= 10).sum() / len(cls_dict[i])
                obj_mean_1.append(m_1)
                obj_mean_3.append(m_3)
                obj_mean_5.append(m_5)
                obj_mean_10.append(m_10) 
           
        obj_mean_1 = np.mean(obj_mean_1) if len(obj_mean_1) > 0 else 0
        obj_mean_3 = np.mean(obj_mean_3) if len(obj_mean_3) > 0 else 0
        obj_mean_5 = np.mean(obj_mean_5) if len(obj_mean_5) > 0 else 0
        obj_mean_10 = np.mean(obj_mean_10) if len(obj_mean_10) > 0 else 0

        return obj_mean_1 * 100, obj_mean_3 * 100, obj_mean_5 * 100, obj_mean_10 * 100
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
