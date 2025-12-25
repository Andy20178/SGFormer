import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
import torch.nn as nn
from src.model.model_utils.network_PointNet import BinaryClassifier
class SGPN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11,task_id=None):
        super().__init__('SGPN', config)
        self.task_id = task_id
        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial
        # dim_point = 3
        # dim_point_rel = 3
        # if mconfig.USE_RGB:
        #     dim_point +=3
        #     dim_point_rel+=3
        # if mconfig.USE_NORMAL:
        #     dim_point +=3
        #     dim_point_rel+=3
            
        # if mconfig.USE_CONTEXT:
        #     dim_point_rel += 1

        dim_point_feature = 512
        self.flow = 'target_to_source'
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        

        self.obj_predictor = PointNetCls(num_obj_class, in_size=dim_point_feature,
                                 batch_norm=with_bn, drop_out=True)
        if self.config.continue_learning_mode == 'none':
            if mconfig.multi_rel_outputs:
                self.rel_predictor = PointNetRelClsMulti(
                    num_rel_class, 
                    in_size=mconfig.edge_feature_size, 
                    batch_norm=with_bn,drop_out=True)
            else:
                self.rel_predictor = PointNetRelCls(
                    num_rel_class, 
                    in_size=mconfig.edge_feature_size, 
                    batch_norm=with_bn,drop_out=True)
        elif self.config.continue_learning_mode == 'S1':
            self.list_rel = [self.config.task_0_rel_name, self.config.task_1_rel_name, self.config.task_2_rel_name, \
                        self.config.task_3_rel_name, self.config.task_4_rel_name]
            if self.config.continue_learning_method == 'uncertainty':
                self.rel_predictor=nn.ModuleList([
                        nn.ModuleList([BinaryClassifier(in_size=512,
                                                        batch_norm=with_bn,
                                                        drop_out=True
                                                        ) 
                                        for _ in range(len(self.list_rel[i]))
                                        ])
                        for i in range(len(self.list_rel))
                    ])
            elif self.config.continue_learning_method == 'finetune' or self.config.continue_learning_method == 'learning_without_forgetting':
                self.rel_predictor = PointNetRelClsMulti(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()


    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        # import pdb;pdb.set_trace()
        rel_feature = self.rel_encoder(edge_feature)

        rel_cls = self.rel_predictor(rel_feature)

        obj_logits = self.obj_predictor(obj_feature)

        return obj_logits, rel_cls

    def process_train(self, obj_points, obj_2d_feats, VLM_description_embedding, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None,
                      old_rel_predictor_3d_weight=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)
        if self.config.continue_learning_mode == 'none':
            pass
        else:
        #获得gt_rel_cls_full为全局标签
            batch_size = rel_pred.shape[0]
            # num_total_classes = rel_cls_3d.shape[1]  # 应为26
            num_total_classes = 26
            # task-specific label，如 task_0 是 [0, 19, 14, 18, 16]
            task_rel_ids = self.config.get(f'task_{self.task_id}_rel_name')
            gt_rel_cls_full = torch.zeros((batch_size, num_total_classes), device=gt_rel_cls.device)
            for i, class_id in enumerate(task_rel_ids):
                gt_rel_cls_full[:, class_id] = gt_rel_cls[:, i]
        if self.mconfig.multi_rel_outputs:
            if self.config.continue_learning_mode == 'none':
                loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
            elif self.config.continue_learning_mode == 'S1':
                if self.config.continue_learning_method == 'uncertainty':
                    # import pdb;pdb.set_trace()
                    loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
                elif self.config.continue_learning_method == 'finetune':
                    loss_rel = F.cross_entropy(rel_pred, gt_rel_cls_full)
                    rel_pred = rel_pred[:, task_rel_ids]
                elif self.config.continue_learning_method == 'learning_without_forgetting':
                    loss_rel = F.cross_entropy(rel_pred['current'], gt_rel_cls_full)
                    rel_pred = rel_pred['current'][:, task_rel_ids]
                    if self.task_id > 0:
                        #读取task_id前面的所有类别
                        old_class_ids = []
                        for i in range(self.task_id):
                            old_class_ids.extend(self.list_rel[i])
                        mask = torch.tensor(old_class_ids, device=rel_pred['current'].device)
                        new_log_probs_masked = rel_pred['current'][:, mask]  # [B, |old_classes|]
                        old_probs_masked = rel_pred['old_prob'][:, mask]
                        # LwF 原始版本使用 soft targets + temperature
                        T = 2.0
                        lwf_loss = F.kl_div(new_log_probs_masked / T, old_probs_masked / T, reduction='mean') * (T * T)
                        # import pdb;pdb.set_trace()
                        loss_rel = loss_rel + lwf_loss
                else:
                    raise NotImplementedError("unknown continue_learning_method type")
         # compute loss for rel
        # if self.mconfig.multi_rel_outputs:
        #     loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        # else:
        #     loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

        
        loss = 0.1 * loss_obj + loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
           
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False,
                    is_all_task=False):
 
        if self.config.continue_learning_mode == 'none':
            obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        else:
            if is_all_task:
                obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False, is_all_task=True)
                all_task_rel_ids = []
                for i in range(int(self.task_id)+1):
                    task_rel_ids = self.config.get(f'task_{i}_rel_name')
                    all_task_rel_ids.extend(task_rel_ids)
            else:
                obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False, is_all_task=False)
                all_task_rel_ids = []
                all_task_rel_ids.extend(self.list_rel[self.task_id])
        if self.config.continue_learning_mode == 'none':
            pass
        elif self.config.continue_learning_mode == 'S1':
            if self.config.continue_learning_method == 'uncertainty':
                if self.mconfig.multi_rel_outputs:
                    rel_pred = rel_pred
                else:
                    rel_pred = rel_pred['logit']
                # rel_pred = F.softmax(rel_pred, dim=-1)
            elif self.config.continue_learning_method == 'finetune':
                if self.mconfig.multi_rel_outputs:
                    rel_pred = rel_pred[:, all_task_rel_ids]  # [B, 5]
                    # 将 rel_pred 按照 all_task_rel_ids 的顺序重新排列
                    # rel_pred = rel_pred[:, all_task_rel_ids]
                    
                    # import pdb;pdb.set_trace()
                else:
                    rel_pred = rel_pred['logit'][:, all_task_rel_ids]  # [B, 5]
            elif self.config.continue_learning_method == 'learning_without_forgetting':
                if self.mconfig.multi_rel_outputs:
                    rel_pred = rel_pred['current'][:, all_task_rel_ids]
                else:
                    rel_pred = rel_pred['current']
        else:
            raise NotImplementedError("unknown continue_learning_mode type")
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores 
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
