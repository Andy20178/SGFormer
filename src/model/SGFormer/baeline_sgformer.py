from enum import Flag
import imp
import clip
from re import X
from tkinter import HIDDEN, Variable
from turtle import Turtle, forward
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti)
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from clip_adapter.model import AdapterModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import dgl.function as fn
from src.model.model_utils.network_PointNet import PointNetfeat
from src.model.SGFormer.help_function import parse_attn_structure
from math import sqrt
from tqdm import tqdm
import os
import sys
from six.moves.urllib.request import urlretrieve
import six
import zipfile
import array
import torch
from src.model.SGFormer.utils import obj_edge_vectors, reporthook, load_word_vectors
from utils import op_utils
from src.model.SGFormer.SGFormer_message_passing_model import SGFormer_message_passing_model
from src.model.model_utils.network_PointNet import EDLClassifier, BinaryClassifier
# from src.model.SGFormer.help_function import edl_mse_loss
from src.model.SGFormer.EDL_Losses import edl_digamma_loss
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk, evaluate_topk_predicate_with_per_class
EMBEDDING_TYPE = ['none', 'glove', 'bert_1024', 'bert_768', 'none_CA', 'none_CA_CLIP']
EMBEDDING_TYPE_DICT = {
    'none': 300,
    'glove': 300,
    'bert_1024': 1024,
    'bert_768': 768,
    'none_CA': 768,
    'none_CA_CLIP': 512
}
class SGFormer(BaseModel):
    def __init__(self,
                 config,
                 num_obj_class,
                 num_rel_class,
                 dim_descriptor=11,
                 task_id=None
                 ):
        super().__init__('SGFormer', config, task_id)
        self.task_id = task_id

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        

        #这是复制mmgnet的代码，有需要的时候再改
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim
        dim_point_feature = 768
        self.momentum = 0.1
        self.model_pre = None

        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder 2D的编码器暂时不需要
        # self.rel_encoder_2d = PointNetfeat(
        #     global_feat=True,
        #     batch_norm=with_bn,
        #     point_size=dim_point_rel,
        #     input_transform=False,
        #     feature_transform=mconfig.feature_transform,
        #     out_size=512)
        
        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)

        self.triplet_projector_3d = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )

        self.clip_adapter = AdapterModel(input_size=512, output_size=512, alpha=0.5)
        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.config.use_VLM_description:
            self.mlp_projection = torch.nn.Sequential(
                torch.nn.Linear(512 - 8 + 2048, 512 - 8),
                torch.nn.BatchNorm1d(512 - 8),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
        # else:
        #     self.mlp_projection = torch.
        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512 + 256, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        # if self.vlm_description_embedding is not None:
        #     self.mlp_vlm = torch.nn.Sequential(
        #         torch.nn.Linear(512 + 256, 512 - 8),
        #         torch.nn.BatchNorm1d(512 - 8),
        #         torch.nn.ReLU(),
        #         torch.nn.Dropout(0.1)
        #     )
        if self.config.continue_learning_mode == 'none':
            #如果没有持续学习的设定，则一切的东西按原计划进行
            if mconfig.multi_rel_outputs:
                self.rel_predictor_3d = PointNetRelClsMulti(
                    num_rel_class, 
                    in_size=512, 
                    batch_norm=with_bn,drop_out=True)
            else:
                self.rel_predictor_3d = PointNetRelCls(
                    num_rel_class, 
                    in_size=512, 
                    batch_norm=with_bn,drop_out=True)
        
        elif self.config.continue_learning_mode == 'S1':
            self.list_rel = [self.config.task_0_rel_name, self.config.task_1_rel_name, self.config.task_2_rel_name, \
                        self.config.task_3_rel_name, self.config.task_4_rel_name]
            #Edge类别的持续
            #多label分类
            if mconfig.multi_rel_outputs:
                if self.config.continue_learning_method == 'uncertainty':
                    #构造连续的分类器
                    # import pdb;pdb.set_trace()
                    #连续的二元分类器
                    self.rel_predictor_3d=nn.ModuleList([
                        nn.ModuleList([BinaryClassifier(in_size=512,
                                                        batch_norm=with_bn,
                                                        drop_out=True
                                                        ) 
                                        for _ in range(len(self.list_rel[i]))
                                        ])
                        for i in range(len(self.list_rel))
                    ])
                    # import pdb;pdb.set_trace()
                elif self.config.continue_learning_method == 'finetune':
                    self.rel_predictor_3d=PointNetRelClsMulti(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
                elif self.config.continue_learning_method == 'learning_without_forgetting':
                    self.current_rel_predictor_3d=PointNetRelClsMulti(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
                    self.old_rel_predictor_3d=PointNetRelClsMulti(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
            else:
                if self.config.continue_learning_method == 'uncertainty':
                    #构造连续的分类器
                    # import pdb;pdb.set_trace()
                    self.rel_predictor_3d=nn.ModuleList(
                        [EDLClassifier(
                        len(self.list_rel[i]), 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
                        for i in range(len(self.list_rel))
                        ]
                    )
                elif self.config.continue_learning_method == 'finetune':
                    self.rel_predictor_3d=PointNetRelCls(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
                elif self.config.continue_learning_method == 'learning_without_forgetting':
                    self.current_rel_predictor_3d=PointNetRelCls(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
                    self.old_rel_predictor_3d=PointNetRelCls(
                        num_rel_class, 
                        in_size=512, 
                        batch_norm=with_bn,drop_out=True)
        
        
        self.init_weight(obj_label_path=mconfig.obj_label_path, \
                         rel_label_path=mconfig.rel_label_path, \
                         adapter_path=mconfig.adapter_path)
        self.attn_list = parse_attn_structure(self.config.MODEL.SGFormer_attn_structure)

        
        self.SGFormer = SGFormer_message_passing_model(self.config)
        
        # 优化器，将传递信息网络和骨干网络的学习率分开
        mmg_obj, mmg_rel = [], []
        for name, para in self.SGFormer.named_parameters():
            if 'nn_edge' in name:
                mmg_rel.append(para)
            else:
                mmg_obj.append(para)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.rel_encoder_2d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_3d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_obj, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor_3d.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            
            {'params':self.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.triplet_projector_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.triplet_projector_2d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        if self.config.continue_learning_mode != 'none':

            if self.config.continue_learning_method == 'uncertainty':
                self.optimizer.add_param_group({'params':self.rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD})
            elif self.config.continue_learning_method == 'finetune':
                self.optimizer.add_param_group({'params':self.rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD})
            elif self.config.continue_learning_method == 'learning_without_forgetting':
                self.optimizer.add_param_group({'params':self.current_rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD})
                self.optimizer.add_param_group({'params':self.old_rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD})
        else:
            self.optimizer.add_param_group({'params':self.rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD})
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

        #引入word_embedding
        self.load_text_feature()
        # self.mlp_projection = torch.nn.Linear(self.config.MODEL.SGFormer_word_in_dim + self.config.MODEL.SGFormer_feat_dim, self.config.MODEL.SGFormer_feat_dim)
        #节点分类
        # import pdb;pdb.set_trace()
        # if self.
        # self.task_rel_ids = self.config.get(f'task_{self.task_id}_rel_name')
        if self.config.task_type == 'PredCls':
            self.obj_cls_embedding = nn.Embedding(self.num_class, 512)
        else:
            self.obj_cls_embedding = None
    def load_text_feature(self):
        if self.config.MODEL.embedding_type == 'none':
            print("不使用word embeding")
        elif self.config.MODEL.embedding_type == 'glove':
            print("抽取个glove word embedding")
            self.obj_classes = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "counter", "shelf", "curtain", "pillow", "clothes", "ceiling", "fridge", "tv", "towel", "plant", "box", "nightstand", "toilet", "sink", "lamp", "bathtub", "blanket"]
            self.GLOVE_DIR = "./"
            embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.GLOVE_DIR)
            self.object_word_embed=nn.Parameter(torch.rand(1,300))  # [1, 300]
            self.register_buffer('word_embedding_26_buffer',embed_vecs)
            self.laye_norm_word_embedding = nn.LayerNorm(self.word_in_dim)
        elif self.config.MODEL.embedding_type == 'bert_1024':
            bert_embedding=torch.load("bert_vector.pth",map_location="cuda")
            bert_embedding = torch.tensor(bert_embedding)
            self.object_word_embed=nn.Parameter(torch.rand(1,1024))  # [1, 1024]
            self.register_buffer('word_embedding_26_buffer',bert_embedding)
            self.laye_norm_word_embedding_bert = nn.LayerNorm(self.word_in_dim)
        elif self.config.MODEL.embedding_type == 'bert_768':
            bert_embedding=torch.load("bert_vector_uncased_H_768.pth",map_location="cuda")
            bert_embedding = torch.tensor(bert_embedding)
            self.object_word_embed=nn.Parameter(torch.rand(1,768))  # [1, 1024]
            self.register_buffer('word_embedding_26_buffer',bert_embedding)
            self.laye_norm_word_embedding_bert = nn.LayerNorm(self.word_in_dim)
        elif self.config.MODEL.embedding_type == 'none_CA':
            bert_embedding=torch.load("/home/lcs/ICCV_2023/dataset_processing/BERT_vector_26R+1_768dim.pth",map_location="cuda")
            bert_embedding = torch.tensor(bert_embedding)
            self.object_word_embed=nn.Parameter(torch.rand(1,768))  # [1, 1024]
            self.register_buffer('word_embedding_26_buffer',bert_embedding)
            self.laye_norm_word_embedding_bert = nn.LayerNorm(self.word_in_dim)
        elif self.config.MODEL.embedding_type == 'CLIP':
            #获取那20种物体的文本特征
            bert_embedding, rel_text_feature = self.get_label_weight(self.config.MODEL.obj_label_path, self.config.MODEL.rel_label_path)
            # import pdb;pdb.set_trace()
            # bert_embedding=torch.load("/home/lcs/tpami2025/data_processing/class_features_512.pth",map_location="cuda")['features']
            #import pdb;pdb.set_trace()
            bert_embedding = torch.tensor(bert_embedding)
            #import pdb;pdb.set_trace()
            # self.object_word_embed=nn.Parameter(torch.rand(1,512))  # [1, 1024]
            #register_buffer：用于保存不需要梯度的持久性数据（如mask、均值、方差等），不会被训练。
            #nn.Parameter：用于需要训练的参数，才会被优化器更新。
            self.register_buffer('word_embedding_160_buffer',bert_embedding)
            # self.word_embedding_160 = nn.Parameter(bert_embedding)  # 可训练
            self.laye_norm_word_embedding_bert = nn.LayerNorm(self.config.MODEL.SGFormer_word_in_dim)
    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[-1].weight)
        obj_text_features, rel_text_feature = self.get_label_weight(obj_label_path, rel_label_path)
        # node feature classifier        
        self.obj_predictor_2d = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_2d.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_2d.parameters():
            param.requires_grad = True
        
        self.obj_predictor_3d = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_3d.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_3d.parameters():
            param.requires_grad = True

        self.clip_adapter.load_state_dict(torch.load(adapter_path, 'cpu'))
        # freeze clip adapter
        for param in self.clip_adapter.parameters():
            param.requires_grad = False
        
        self.obj_logit_scale.requires_grad = True

    def get_label_weight(self, obj_label_path, rel_label_path):
        
        self.obj_label_list = []
        self.rel_label_list = []
        self.clip_model, preprocess = clip.load("ViT-B/32", device='cuda',download_root='/home/lcs/tpami2025/')

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        with open(obj_label_path, "r") as f:
            data = f.readlines()
        for line in data:
            self.obj_label_list.append(line.strip())
        
        with open(rel_label_path, "r") as f:
            data = f.readlines()
        for line in data:
            self.rel_label_list.append(line.strip())
        
        # get norm clip weight
        obj_prompt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.obj_label_list]).cuda()
        rel_prompt = torch.cat([clip.tokenize(f"{c}") for c in self.rel_label_list]).cuda()

        with torch.no_grad():
            obj_text_features = self.clip_model.encode_text(obj_prompt)
            rel_text_features = self.clip_model.encode_text(rel_prompt)
        
        obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)
        rel_text_features = rel_text_features / rel_text_features.norm(dim=-1, keepdim=True)

        return obj_text_features.float(), rel_text_features.float()
    def cosine_loss(self, A, B, t=1):
        return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    def compute_triplet_loss(self, obj_logits_3d, rel_cls_3d, obj_logits_2d, rel_cls_2d, edge_indices):
        triplet_loss = []
        obj_logits_3d_softmax = F.softmax(obj_logits_3d, dim=-1)
        obj_logits_2d_softmax = F.softmax(obj_logits_2d, dim=-1)
        for idx, i in enumerate(edge_indices):
            obj_score_3d = obj_logits_3d_softmax[i[0]]
            obj_score_2d = obj_logits_2d_softmax[i[0]]
            sub_score_3d = obj_logits_3d_softmax[i[1]]
            sub_score_2d = obj_logits_2d_softmax[i[1]]
            rel_score_3d = rel_cls_3d[idx]
            rel_score_2d = rel_cls_2d[idx]
            node_score_3d = torch.einsum('n,m->nm', obj_score_3d, sub_score_3d)
            node_score_2d = torch.einsum('n,m->nm', obj_score_2d, sub_score_2d)
            triplet_score_3d = torch.einsum('nl,m->nlm', node_score_3d, rel_score_3d).reshape(-1)
            triplet_score_2d = torch.einsum('nl,m->nlm', node_score_2d, rel_score_2d).reshape(-1)
            triplet_loss.append(F.l1_loss(triplet_score_3d, triplet_score_2d.detach(), reduction='sum')) 
            
            
        #return torch.sum(torch.tensor(triplet_loss))
        return torch.mean(torch.tensor(triplet_loss))
    def message_passing(self,g,node_feats,edge_feats,CLIP_feats_use_grad=True):
        if self.config.MODEL.embedding_type == 'CLIP':
            embed_vecs_160_buffer = self.get_buffer('word_embedding_160_buffer')
            word_feats = self.laye_norm_word_embedding_bert(embed_vecs_160_buffer)
            # import pdb;pdb.set_trace()
        else:
            word_feats = 0
        node_logits_word = 0
        # import pdb;pdb.set_trace()
        node_logits_word, node_feats,edge_feats = self.SGFormer(g, node_feats,edge_feats,word_feats)

        # import pdb;pdb.set_trace()
        node_feats = node_feats
        edge_feats = edge_feats
        node_logits_word = node_logits_word.squeeze()
        return node_logits_word,node_feats,edge_feats

    def forward(self, obj_points, vlm_description_embedding, edge_indices, descriptor=None, batch_ids=None, istrain=False, is_all_task=False, gt_cls=None):
        # assert (descriptor[:, 6:11] >= 0).all(), "Dimensions, volume, length should be non-negative"
        # assert (descriptor[:, 6:11].min() > 1e-6).all() or print("Some dims/volume/length are near zero!")
        # import pdb;pdb.set_trace()
        try:
            src = edge_indices[0]
            dst = edge_indices[1]
        except:
            import pdb;pdb.set_trace()
        descriptor[:, 6:11] = descriptor[:, 6:11].clamp(min=1e-6)
        
        if self.config.task_type == 'SGCls':
            obj_feature = self.obj_encoder(obj_points)
            

            obj_feature = self.mlp_3d(obj_feature)
            if self.config.use_VLM_description:
                # import pdb;pdb.set_trace()
                obj_feature = torch.cat([obj_feature, vlm_description_embedding], dim=-1)
                #并使用MLP投影至原来的维度
                obj_feature = self.mlp_projection(obj_feature)
            else:
                pass
            #引入空间信息
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor[:,3:].clone()
                tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
                obj_feature = torch.cat([obj_feature, tmp],dim=-1)
        else:
            # import pdb;pdb.set_trace()
            obj_feature = self.obj_cls_embedding(gt_cls)

        #构造图
        g = dgl.graph((src, dst),num_nodes=obj_feature.shape[0])


        ''' Create edge feature '''
        
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        #检查edge_feature是有异常值
        if torch.isnan(edge_feature).any():
            print("Invalid edge_feature")
            import pdb;pdb.set_trace()
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        #拿到位置中心，有助于之后的操作
        obj_center = descriptor[:, :3].clone()

        #经过Transformer传递过的node_feats和edge_feats
        obj_logits_text, obj_feature_3d, edge_feature_3d =self.message_passing(g,obj_feature,rel_feature_3d, \
                                                                                CLIP_feats_use_grad=True)
        # import pdb;pdb.set_trace()
        obj_logits_3d = self.obj_predictor_3d(obj_feature_3d / obj_feature_3d.norm(dim=-1, keepdim=True))
        # import pdb;pdb.set_trace()
        if self.config.continue_learning_mode == 'none':
            rel_cls_3d = self.rel_predictor_3d(edge_feature_3d)
            if torch.isnan(rel_cls_3d).any():
                print("Invalid rel_cls_3d")
                import pdb;pdb.set_trace()
        elif self.config.continue_learning_mode == 'S1':
            # import pdb;pdb.set_trace()
            if self.config.continue_learning_method == 'uncertainty':
                task_id = self.task_id
                if is_all_task:
                    if self.mconfig.multi_rel_outputs:
                        rel_cls_3d = []
                        for i in range(int(task_id) + 1):
                            small_rel_cls_3d = torch.cat(([head(edge_feature_3d).squeeze(0) for head in  self.rel_predictor_3d[i]]),dim=1)
                            rel_cls_3d.append(small_rel_cls_3d)
                        rel_cls_3d = torch.cat(rel_cls_3d, dim=1)
                    else:
                        rel_cls_3d = {}
                        all_probs = []
                        all_weights = []
                        all_unc = []
                        # import pdb;pdb.set_trace()
                        for i in range(int(task_id) + 1):
                            out = self.rel_predictor_3d[i](edge_feature_3d)
                            prob_i = out['logit']  # shape [B, C_i]
                            all_probs.append(prob_i)
                        all_probs = torch.cat(all_probs, dim=1)
                        rel_cls_3d['logit'] = all_probs

                    # import pdb;pdb.set_trace()
                else:
                    # rel_cls_3d = self.rel_predictor_3d[task_id](edge_feature_3d)
                    if self.mconfig.multi_rel_outputs:
                        # import pdb;pdb.set_trace()
                        rel_cls_3d = torch.cat(([head(edge_feature_3d).squeeze(0) for head in  self.rel_predictor_3d[task_id]]),dim=1)
                        # import pdb;pdb.set_trace()
                    else:
                        rel_cls_3d = self.rel_predictor_3d[task_id](edge_feature_3d)
                    # import pdb;pdb.set_trace()
            elif self.config.continue_learning_method == 'learning_without_forgetting':
                rel_cls_3d = {}
                rel_cls_3d['current'] = self.current_rel_predictor_3d(edge_feature_3d)
                with torch.no_grad():
                    rel_cls_3d['old'] = self.old_rel_predictor_3d(edge_feature_3d)  # features 是关系特征输入
                    rel_cls_3d['old_prob'] = rel_cls_3d['old'].exp()
            elif self.config.continue_learning_method == 'finetune':
                rel_cls_3d = self.rel_predictor_3d(edge_feature_3d)
                
        
        logit_scale = self.obj_logit_scale.exp()    
        obj_logits_3d = logit_scale * self.obj_predictor_3d(obj_feature_3d / obj_feature_3d.norm(dim=-1, keepdim=True))
        if torch.isnan(obj_logits_3d).any():
            print("Invalid obj_logits_3d")
            import pdb;pdb.set_trace()
        if self.config.task_type == 'PredCls':
            #直接将gt_cls转化为one_hot编码            
            obj_logits_3d = torch.nn.functional.one_hot(gt_cls, num_classes=len(self.obj_label_list))
            obj_logits_3d = obj_logits_3d.float()
            # import pdb;pdb.set_trace()
        if istrain:
            return obj_logits_3d, rel_cls_3d, obj_logits_text, logit_scale
        else:
            return obj_logits_3d, rel_cls_3d, obj_logits_text, obj_feature_3d
    def process_train(self, obj_points, obj_2d_feats, vlm_description_embedding, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None,
                      old_rel_predictor_3d_weight=None):
        self.iteration +=1  
        #调用forward函数
        if self.config.task_type == 'SGCls':
            obj_logits_3d, rel_cls_3d, obj_logits_text, logit_scale = self(obj_points, vlm_description_embedding, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)  
        elif self.config.task_type == 'PredCls':
            obj_logits_3d, rel_cls_3d, obj_logits_text, _ = self(obj_points, vlm_description_embedding, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True, gt_cls=gt_cls)  
        else:
            raise NotImplementedError("unknown task_type type")
        #compute loss for obj
        loss_obj = F.cross_entropy(obj_logits_3d, gt_cls)
        # compute loss for text
        loss_text = F.cross_entropy(obj_logits_text, gt_cls)
        # compute loss for rel
        if torch.isnan(rel_cls_3d).any():
            print("Invalid rel_cls_3d")
            import pdb;pdb.set_trace()
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                # import pdb;pdb.set_trace()
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]                
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")
            # import pdb;pdb.set_trace()
            if self.config.continue_learning_mode == 'none':
                loss_rel = F.binary_cross_entropy_with_logits(rel_cls_3d, gt_rel_cls, weight=weight)
                if torch.isnan(loss_rel):
                    print("Invalid loss")
                    import pdb;pdb.set_trace()
            elif self.config.continue_learning_mode == 'S1':
                if self.config.continue_learning_method == 'uncertainty':
                    # import pdb;pdb.set_trace()
                    loss_rel = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls)
                elif self.config.continue_learning_method == 'finetune':
                    #在finetune 这确实要构建一下 label 的抬升
                    # 假设 rel_cls_3d 的 shape 是 [B, 26]，gt_rel_cls 是 [B] 或 [B, 5]
                    batch_size = rel_cls_3d.shape[0]
                    num_total_classes = rel_cls_3d.shape[1]  # 应为26
                    # task-specific label，如 task_0 是 [0, 19, 14, 18, 16]
                    task_rel_ids = self.config.get(f'task_{self.task_id}_rel_name')
                    gt_rel_cls_full = torch.zeros((batch_size, num_total_classes), device=gt_rel_cls.device)
                    for i, class_id in enumerate(task_rel_ids):
                        gt_rel_cls_full[:, class_id] = gt_rel_cls[:, i]
                    # 现在 rel_cls_3d 和 gt_rel_cls_one_hot 的 shape 是一致的
                    # import pdb;pdb.set_trace()
                    loss_rel = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls_full)
                    # 将 rel_cls_3d 按照 task_rel_ids 的顺序重新排列
                    #要给下面检测留余地啊
                    rel_cls_3d = rel_cls_3d[:, task_rel_ids]
                    # import pdb;pdb.set_trace()
                    #将 gt 要转化为对应的 26 类的 one hot 编码
                    # loss_rel = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls)
                elif self.config.continue_learning_method == 'learning_without_forgetting':
                    batch_size = rel_cls_3d['current'].shape[0]
                    num_total_classes = rel_cls_3d['current'].shape[1]  # 应为26
                    # task-specific label，如 task_0 是 [0, 19, 14, 18, 16]
                    task_rel_ids = self.config.get(f'task_{self.task_id}_rel_name')
                    gt_rel_cls_full = torch.zeros((batch_size, num_total_classes), device=gt_rel_cls.device)
                    for i, class_id in enumerate(task_rel_ids):
                        gt_rel_cls_full[:, class_id] = gt_rel_cls[:, i]
                    # 现在 rel_cls_3d 和 gt_rel_cls_one_hot 的 shape 是一致的
                    loss_rel = F.binary_cross_entropy(rel_cls_3d['current'], gt_rel_cls_full)
                    #还有学习old 的那个
                    if self.task_id > 0:
                        #读取task_id前面的所有类别
                        old_class_ids = []
                        for i in range(self.task_id):
                            old_class_ids.extend(self.list_rel[i])
                        mask = torch.tensor(old_class_ids, device=rel_cls_3d['current'].device)
                        new_log_probs_masked = rel_cls_3d['current'][:, mask]  # [B, |old_classes|]
                        old_probs_masked = rel_cls_3d['old_prob'][:, mask]
                        # LwF 原始版本使用 soft targets + temperature
                        T = 2.0
                        lwf_loss = F.kl_div(new_log_probs_masked / T, old_probs_masked / T, reduction='mean') * (T * T)
                        # import pdb;pdb.set_trace()
                        loss_rel = loss_rel + lwf_loss
                    rel_cls_3d = rel_cls_3d['current']
                    
            else:
                raise NotImplementedError("unknown continue_learning_mode type")
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = len(self.list_rel[self.task_id]))
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_cls_3d.device, requires_grad=False)
            else:
                if self.config.continue_learning_mode == None:
                    #不使用持续学习
                    loss_rel = F.nll_loss(rel_cls_3d, gt_rel_cls, weight = weight)
                elif self.config.continue_learning_mode == 'S1':
                    # import pdb;pdb.set_trace()
                    # if self.task_id == 1:
                    #     import pdb;pdb.set_trace()
                    #目前只做持续学习的第一个实验
                    if self.config.continue_learning_method == 'uncertainty':
                        # import pdb;pdb.set_trace()
                        #将gt_rel_cls转换为one_hot
                        gt_rel_cls_one_hot = torch.nn.functional.one_hot(gt_rel_cls,num_classes = len(self.list_rel[self.task_id]))
                        loss_rel = edl_digamma_loss(rel_cls_3d['logit'], gt_rel_cls_one_hot, self.epoch, len(self.list_rel[self.task_id]), self.config.annealing_step)
                        rel_cls_3d = F.log_softmax(rel_cls_3d['logit'], dim=1)
                    elif self.config.continue_learning_method == 'finetune':
                        #需要根据task_id来将label抬升
                        if self.task_id == 1:
                            gt_rel_cls = gt_rel_cls + 5
                        elif self.task_id == 2:
                            gt_rel_cls = gt_rel_cls + 10
                        elif self.task_id == 3:
                            gt_rel_cls = gt_rel_cls + 15
                        elif self.task_id == 4:
                            gt_rel_cls = gt_rel_cls + 20
                        # import pdb;pdb.set_trace()
                        loss_rel = F.nll_loss(rel_cls_3d, gt_rel_cls)
                    elif self.config.continue_learning_method == 'learning_without_forgetting':
                        #需要根据task_id来将label抬升
                        if self.task_id == 1:
                            gt_rel_cls = gt_rel_cls + 5
                        elif self.task_id == 2:
                            gt_rel_cls = gt_rel_cls + 10
                        elif self.task_id == 3:
                            gt_rel_cls = gt_rel_cls + 15
                        elif self.task_id == 4:
                            gt_rel_cls = gt_rel_cls + 20
                        # import pdb;pdb.set_trace()
                        #让新的分类器和旧的分类器对齐
                        loss_rel = F.nll_loss(rel_cls_3d['current'], gt_rel_cls)
                        if self.task_id > 0:
                            #读取task_id前面的所有类别
                            old_class_ids = []
                            for i in range(self.task_id):
                                old_class_ids.extend(self.list_rel[i])
                            # import pdb;pdb.set_trace()
                            # old_class_ids = [i for i in range(len(self.list_rel[self.task_id-1]))]
                            # import pdb;pdb.set_trace()
                            mask = torch.tensor(old_class_ids, device=rel_cls_3d['current'].device)
                            new_log_probs_masked = rel_cls_3d['current'][:, mask]  # [B, |old_classes|]
                            old_probs_masked = rel_cls_3d['old_prob'][:, mask]
                            # LwF 原始版本使用 soft targets + temperature
                            T = 2.0
                            lwf_loss = F.kl_div(new_log_probs_masked / T, old_probs_masked / T, reduction='batchmean') * (T * T)
                            loss_rel = loss_rel + lwf_loss
                        rel_cls_3d = rel_cls_3d['current']
                            
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_t = self.mconfig.lambda_t
        lambda_max = max(lambda_r,lambda_o,lambda_t)
        lambda_r /= lambda_max
        lambda_o /= lambda_max
        lambda_t /= lambda_max

        #设置一个断点，检查loss的合理性
        #debugging
        #检查loss是否为nan
        if torch.isnan(loss_obj) or torch.isnan(loss_rel) or torch.isnan(loss_text):
            print("Invalid loss")
            import pdb;pdb.set_trace()
        if self.config.task_type == 'PredCls':
            loss = lambda_r * loss_rel
        else:
            loss = lambda_o * loss_obj + lambda_r * loss_rel + lambda_t * loss_text

        try:
            self.backward(loss)
        except Exception as e:
            print(f"Error in backward pass: {e}")
            # 重点检查 rel 分支（多标签关系预测）
            import pdb;pdb.set_trace()
            if 'rel_cls_3d' in locals() or 'rel_cls_3d' in globals():
                rel_logits = rel_cls_3d
                print(f" Rel Logits: min={rel_logits.min().item():.4f}, "
                    f"max={rel_logits.max().item():.4f}, "
                    f"shape={rel_logits.shape}, "
                    f"nan={torch.isnan(rel_logits).any().item()}, "
                    f"inf={torch.isinf(rel_logits).any().item()}")

                # 检查是否意外经过了 sigmoid 或 log_softmax
                if rel_logits.min() >= 0 and rel_logits.max() <= 1:
                    print("⚠️  WARNING: rel_logits are in [0,1] range — "
                        "did you accidentally apply sigmoid or softmax in forward?")
                if rel_logits.min() < -10 or rel_logits.max() > 10:
                    print("⚠️  WARNING: rel_logits are too large — may cause sigmoid overflow")

        # 检查是否超出 [0,1]
        # if (pred_prob < 0).any() or (pred_prob > 1).any():
        #     print("Prediction out of range!")
        rel_cls_3d = torch.sigmoid(rel_cls_3d)
        # import pdb;pdb.set_trace()
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        top_k_text = evaluate_topk_object(obj_logits_text.detach(), gt_cls, topk=11)
        if not with_log:
            return top_k_obj, top_k_rel, loss_rel.detach(), loss_obj.detach(), loss.detach()

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        text_topk_list = [100 * (top_k_text <= i).sum() / len(top_k_text) for i in [1, 5, 10]]
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/text_loss", loss_text.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Text_R1", text_topk_list[0]),
                ("train/Text_R5", text_topk_list[1]),
                ("train/Text_R10", text_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
    def process_val(self, obj_points, obj_2d_feats, vlm_description_embedding, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False,is_all_task=False, \
        accuracy_tracker=None, feature_collector=None):
        if self.config.continue_learning_mode == 'none':
            if self.config.task_type == 'PredCls':
                obj_pred, rel_pred, obj_pred_text, obj_feature_3d = self(obj_points, vlm_description_embedding, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False, gt_cls=gt_cls)
            else:
                obj_pred, rel_pred, obj_pred_text, obj_feature_3d = self(obj_points, vlm_description_embedding, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        else:
            if not is_all_task:
                obj_pred, rel_pred, obj_pred_text, _  = self(obj_points, vlm_description_embedding,edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
                all_task_rel_ids = []
                all_task_rel_ids.extend(self.list_rel[self.task_id])
            else:
                obj_pred, rel_pred, obj_pred_text, _ = self(obj_points, vlm_description_embedding, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False, is_all_task=True)
                all_task_rel_ids = []
                for i in range(int(self.task_id)+1):
                    task_rel_ids = self.config.get(f'task_{i}_rel_name')
                    all_task_rel_ids.extend(task_rel_ids)
            # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()

        if feature_collector is not None:
            feature_collector.add(
                features=obj_feature_3d,
                labels=gt_cls,  # 真实类别
                preds=obj_pred.argmax(-1)  # 可选：预测类别
            )
        
        
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
        # import pdb;pdb.set_trace()
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls.cpu(), topk=11)
        # import pdb;pdb.set_trace()
        top_k_text = evaluate_topk_object(obj_pred_text.detach().cpu(), gt_cls.cpu(), topk=11)
        gt_edges = get_gt(gt_cls.cpu(), gt_rel_cls.cpu(), edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        if accuracy_tracker is not None:
            accuracy_tracker.update_obj_batch(obj_pred.detach().cpu(), gt_cls.cpu(), topk=11)
            # 更新per-class统计
            accuracy_tracker.update_rel_batch(rel_pred.detach().cpu(), gt_edges, topk=6)
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None
        # import pdb;pdb.set_trace()
        return top_k_obj, top_k_text, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
     
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

class Adaptor(nn.Module):
    def __init__(self,num_edge_in_embeddings,edge_mid_channels):
        super().__init__() 
        self.edge_Linear1 = nn.Linear(num_edge_in_embeddings, edge_mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(edge_mid_channels),nn.LeakyReLU(0.2), nn.Dropout())
        self.edge_Linear2 = nn.Linear(edge_mid_channels, num_edge_in_embeddings, bias=False)
    def forward(self,edge_x):
        edge_logits = self.edge_Linear1(edge_x)
        edge_logits = self.edge_BnReluDp(edge_logits.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_Linear2(edge_logits)
        edge_logits = torch.squeeze(edge_logits,0)
        return edge_logits
class Adaptor_Space(nn.Module):
    def __init__(self,num_edge_in_embeddings,edge_mid_channels,space_dim=11):
        super().__init__() 
        self.edge_Linear1 = nn.Linear(num_edge_in_embeddings+space_dim, edge_mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(edge_mid_channels),nn.LeakyReLU(0.2), nn.Dropout())
        self.edge_Linear2 = nn.Linear(edge_mid_channels, num_edge_in_embeddings, bias=False)
    def forward(self,edge_x,edge_feature):
        edge_feature=edge_feature.reshape(-1,edge_feature.size(0),edge_feature.size(1))
        edge_x=torch.cat((edge_x,edge_feature),dim=-1)
        edge_logits = self.edge_Linear1(edge_x)
        edge_logits = self.edge_BnReluDp(edge_logits.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_Linear2(edge_logits)
        edge_logits = torch.squeeze(edge_logits,0)
        return edge_logits
