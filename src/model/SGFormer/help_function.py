from hashlib import new
from itertools import count
from lib2to3.pytree import Node
from platform import node
from sched import scheduler
from typing import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv
from tqdm import *
import numpy as np
from sklearn.metrics import precision_score
from torch.nn import SyncBatchNorm
from src.model.model_utils.networks_base import BaseNetwork
'''
EdgeGCN used for SGGpoint (Chaoyi Zhang)
'''
class EdgeGCN(torch.nn.Module):
    def __init__(self,num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag,nObjClasses, nRelClasses):
        super(EdgeGCN, self).__init__()
        
        self.node_GConv1 = GCNConv(num_node_in_embeddings, num_node_in_embeddings // 2, add_self_loops=True)
        self.node_GConv2 = GCNConv(num_node_in_embeddings // 2, num_node_in_embeddings, add_self_loops=True)

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.AttnEdgeFlag = AttnEdgeFlag # boolean (for ablaiton studies)
        self.AttnNodeFlag = AttnNodeFlag # boolean (for ablaiton studies)

        # multi-dimentional (N-Dim) node/edge attn coefficients mappings
        self.edge_attentionND = nn.Linear(num_edge_in_embeddings, num_node_in_embeddings // 2) if self.AttnEdgeFlag else None
        self.node_attentionND = nn.Linear(num_node_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        #节点分类
        node_mid_channels = num_node_in_embeddings // 2
        self.node_linear1 = nn.Linear(num_node_in_embeddings, node_mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(nn.BatchNorm1d(node_mid_channels), nn.LeakyReLU(0.2), nn.Dropout())
        self.node_linear2 = nn.Linear(node_mid_channels, nObjClasses, bias=False)

        #边分类
        edge_mid_channels = num_edge_in_embeddings // 2
        self.edge_linear1 = nn.Linear(num_edge_in_embeddings, edge_mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(edge_mid_channels), nn.LeakyReLU(0.2), nn.Dropout())
        self.edge_linear2 = nn.Linear(edge_mid_channels, nRelClasses, bias=False)
        

    def concate_NodeIndicator_for_edges(self, node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)#shape [num_node, node_feature]
        #edge_index_list.shape  [num_edges, 2]
        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]#source
        object_idx_list = edge_index_list[:, 1]#target

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]    # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    def forward(self, node_feats, edge_index):
        node_feats = node_feats.unsqueeze(0)#[1, num_node, 256]
        edge_feats = edge_feats_initialization(node_feats, edge_index)#[num_edge, 512]
        edge_feats = edge_feats.unsqueeze(0)
        edge_feats = edge_feats.float()
        node_feats = node_feats.float()
        edge_index = edge_index.t()
        #### Deriving Edge Attention
        if self.AttnEdgeFlag:
            edge_indicator = self.edge_attentionND(edge_feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            #raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0], dim=2, reduce='mean', dim_size=node_feats.size(1)) # (1, num_mid_channels, num_nodes)
            raw_out_col = scatter(edge_indicator, edge_index.t()[:, 1], dim=2, reduce='mean', dim_size=node_feats.size(1)) # (1, num_mid_channels, num_nodes)
            agg_edge_indicator_logits = raw_out_row * raw_out_col                                        # (1, num_mid_channels, num_nodes)
            agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits).permute(0, 2, 1).squeeze(0)    # (num_nodes, num_mid_channels)
        else:
            agg_edge_indicator = 1

        #### Node Evolution Stream (NodeGCN)
        node_feats = F.relu(self.node_GConv1(node_feats, edge_index)) * agg_edge_indicator # applying EdgeAttn on Nodes
        node_feats = F.dropout(node_feats, training=self.training)
        node_feats = F.relu(self.node_GConv2(node_feats, edge_index))
        #node_feats = node_feats.unsqueeze(0)  # (1, num_nodes, num_embeddings)

        #### Deriving Node Attention
        if self.AttnNodeFlag:
            node_indicator = F.relu(self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))                  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)               # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0,2,1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        ### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.permute(0, 2, 1)                  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)                   # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, training=self.training) * agg_node_indicator    # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)
        
        # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
        node_x = self.node_linear1(node_feats)
        node_x = self.node_BnReluDp(node_x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(node_x)
        node_logits = node_logits.squeeze(0)
        node_logits = F.softmax(node_logits, dim=1)
        #node_logits = node_logits.squeeze(0)

        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        edge_x = self.edge_linear1(edge_feats)
        edge_x = self.edge_BnReluDp(edge_x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(edge_x)
        # return {'node_feats': node_feats, 'edge_feats': edge_feats}
        edge_logits = F.softmax(edge_logits, dim=1)
        edge_logits = edge_logits.squeeze(0)
        return node_logits, edge_logits
def edl_mse_loss(predict_dict, target_onehot, epoch, num_classes, kl_strength=1.0):
    alpha = predict_dict['alpha']
    S = torch.sum(alpha, dim=1, keepdim=True)
    p_hat = alpha / S

    mse = torch.sum((target_onehot - p_hat) ** 2, dim=1)
    var = torch.sum(p_hat * (1 - p_hat) / (S + 1), dim=1)

    loss = mse + var

    # KL 散度正则（鼓励模型在不确定时输出 u=1）
    uniform_alpha = torch.ones_like(alpha)
    kl_div = kl_dirichlet(alpha, uniform_alpha)
    annealing = min(1.0, epoch / 10.0)
    loss += kl_strength * annealing * kl_div
    # import pdb;pdb.set_trace()
    return loss.mean()
def kl_dirichlet(alpha, prior, eps=1e-8):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_prior = torch.sum(prior, dim=1, keepdim=True)

    lnB_alpha = torch.sum(torch.lgamma(alpha + eps), dim=1) - torch.lgamma(S_alpha + eps).squeeze()
    lnB_prior = torch.sum(torch.lgamma(prior + eps), dim=1) - torch.lgamma(S_prior + eps).squeeze()

    digamma_diff = torch.digamma(alpha + eps) - torch.digamma(S_alpha + eps)
    kl = torch.sum((alpha - prior) * digamma_diff, dim=1) + lnB_alpha - lnB_prior
    return kl

def edge_feats_initialization(node_feats, batchwise_edge_index):
    node_feats = node_feats.squeeze(0)
    #connections_from_subject_to_object.shape [num_edge, 2]
    connections_from_subject_to_object = batchwise_edge_index
    subject_idx = connections_from_subject_to_object[:, 0]#拿出源节点
    object_idx = connections_from_subject_to_object[:, 1]#拿出目标节点
    
    subject_feats = node_feats[subject_idx]#源节点的feats拿出来
    object_feats = node_feats[object_idx]#拿出目标节点的feats，这其中我保证了节点的feats的正确性
    diff_feats = object_feats - subject_feats

    edge_feats = torch.cat((subject_feats, diff_feats), dim=1)  # equivalent to EdgeConv (with in DGCNN)

    return edge_feats  # (num_Edges, Embeddings * 2)

class NodeMLP(nn.Module):
    def __init__(self, embeddings, nObjClasses, negative_slope=0.2):
        super(NodeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.node_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(SyncBatchNorm(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.node_linear2 = nn.Linear(mid_channels, nObjClasses, bias=False)

    def forward(self, node_feats):
        # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
        x = self.node_linear1(node_feats)
        x = self.node_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(x)
        node_logits = F.log_softmax(node_logits, dim=1)
        return node_logits

class EdgeMLP(nn.Module):
    def __init__(self, embeddings, nRelClasses, negative_slope=0.2):
        super(EdgeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.edge_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(SyncBatchNorm(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.edge_linear2 = nn.Linear(mid_channels, nRelClasses, bias=False)

    def forward(self, edge_feats):
        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        x = self.edge_linear1(edge_feats)
        x = self.edge_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(x)
        return edge_logits


        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))

        return x



class PointNet(nn.Module):#BaseNetwork
    #from DGCNN's repo
    def __init__(self, input_channel, embeddings):
        super(PointNet, self).__init__()
        #input_channel:9, embedding:256
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embeddings, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embeddings)
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x
    # def __init__(self, global_feat = True, input_transform = False, feature_transform = False, 
    #              point_size=9, out_size=256, batch_norm = True,
    #              init_weights=True, pointnet_str:str=None):
    #     super(PointNet, self).__init__()
    #     self.name = 'pnetenc'
    #     self.use_batch_norm = batch_norm
    #     self.relu = nn.ReLU()
    #     self.point_size = point_size
    #     self.out_size = out_size
        
    #     self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
    #     self.conv2 = torch.nn.Conv1d(64, 128, 1)
    #     self.conv3 = torch.nn.Conv1d(128, out_size, 1)
    #     if batch_norm:
    #         self.bn1 = nn.BatchNorm1d(64)
    #         self.bn2 = nn.BatchNorm1d(128)
    #         self.bn3 = nn.BatchNorm1d(out_size)
    #     self.global_feat = global_feat
    #     self.input_transform = input_transform
    #     self.feature_transform = feature_transform
        
    #     if input_transform:
    #         #import pdb;pdb.set_trace()
    #         assert pointnet_str is not None
    #         # self.pointnet_str=pointnet_str
    #         # self.stn = STN3d(point_size=point_size)
    #     if self.feature_transform:
    #         self.fstn = STNkd(k=64)
            
    #     if init_weights:
    #         self.init_weights('constant', 1, target_op = 'BatchNorm')
    #         self.init_weights('xavier_normal', 1)

    # def forward(self, x, return_meta=False):
    #     assert x.ndim >2
    #     n_pts = x.size()[2]
    #     if self.input_transform:
    #         trans = self.stn(x)
    #         x = x.transpose(2, 1)
    #         if self.pointnet_str is None and self.point_size ==3:
    #             x[:,:,:3] = torch.bmm(x[:,:,:3], trans)
    #         elif self.point_size > 3:
    #             assert self.pointnet_str is not None 
    #             for i in len(self.pointnet_str):
    #                 p = self.pointnet_str[i]
    #                 offset = i*3
    #                 offset_ = (i+1)*3
    #                 if p == 'p' or p == 'n': # point and normal
    #                     x[:,:,offset:offset_] = torch.bmm(x[:,:,offset:offset_], trans)
    #         x = x.transpose(2, 1)
    #     else:
    #         trans = torch.zeros([1])
        
    #     x = self.conv1(x)
    #     if self.use_batch_norm:
    #         self.bn1(x)
    #     x = self.relu(x)
        
    #     if self.feature_transform:
    #         trans_feat = self.fstn(x)
    #         x = x.transpose(2,1)
    #         x = torch.bmm(x, trans_feat)
    #         x = x.transpose(2,1)
    #     else:
    #         trans_feat = torch.zeros([1]) # cannot be None in tracing. change to 0
    #     pointfeat = x
    #     x = self.conv2(x)
    #     if self.use_batch_norm:
    #         self.bn2(x)
    #     x = self.relu(x)
    #     x = self.conv3(x)
    #     if self.use_batch_norm:
    #         self.bn3(x)
    #     x = self.relu(x)
        
    #     x = torch.max(x, 2, keepdim=True)[0]
    #     x = x.view(-1, self.out_size)
        
    #     if self.global_feat:
    #         if return_meta:
    #             return x, trans, trans_feat
    #         else:
    #             return x
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        iden = Variable(torch.eye(self.k).view(1,self.k*self.k).repeat(batchsize,1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

def data_processing(xyz, rgb,normal_xyz,instance_index,node_labels,edge_index,edge_labels,descriptor,device,model_PointNet):
            
            dim9 = torch.cat([xyz, rgb, normal_xyz],dim=2)
            dim9 = dim9.permute(1,2,0)
            dim9 = dim9.to(torch.float32)
            dim9 = dim9.to(device)
            dim256 = model_PointNet(dim9)#[4096,256,4]
            
            dim256 = dim256.permute(2,0,1)#[4,4196,256]
            
            descriptor=descriptor[~(descriptor==0).all(dim=2)]
            
            dim256_1, dim256_2,dim256_3,dim256_4,dim256_5, dim256_6,dim256_7,dim256_8= dim256
            instance_index_1,instance_index_2,instance_index_3,instance_index_4,instance_index_5,instance_index_6,instance_index_7,instance_index_8 = instance_index
            instance_index_1 = instance_index_1.detach().numpy()#这个instance脱离计算图无所谓
            instance_index_2 = instance_index_2.detach().numpy()
            instance_index_3 = instance_index_3.detach().numpy()
            instance_index_4 = instance_index_4.detach().numpy()
            instance_index_5 = instance_index_5.detach().numpy()
            instance_index_6 = instance_index_6.detach().numpy()
            instance_index_7 = instance_index_7.detach().numpy()
            instance_index_8 = instance_index_8.detach().numpy()
            big_instance_index = []
            big_instance_index = np.array(big_instance_index)
            instance_index_list = []
            edge_index_1, edge_index_2, edge_index_3, edge_index_4,edge_index_5,edge_index_6,edge_index_7,edge_index_8 = edge_index
            edge_labels_1,edge_labels_2,edge_labels_3,edge_labels_4, edge_labels_5,edge_labels_6,edge_labels_7,edge_labels_8 =edge_labels
            node_labels_1, node_labels_2,node_labels_3,node_labels_4,node_labels_5, node_labels_6,node_labels_7,node_labels_8 = node_labels
            big_edge_index = []
            big_edge_label = []
            big_node_label = []
            node_index = 0
            node_index_old2new_1 = dict()
            node_index_old2new_2 = dict()
            node_index_old2new_3 = dict()
            node_index_old2new_4 = dict()
            node_index_old2new_5 = dict()
            node_index_old2new_6 = dict()
            node_index_old2new_7 = dict()
            node_index_old2new_8 = dict()
            node_label_index = 0
            for iid in np.unique(instance_index_1):
                #iid=iid.item()
                if iid != -1:
                    indices = (instance_index_1 == iid).nonzero()[0]
                    feats = dim256_1[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    if node_index == 0:
                        node_feats = feats
                        #解决node的问题
                    else:
                        node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_1[node_label_index]))
                    node_index_old2new_1[iid] = node_index
                    node_index = node_index + 1 
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_1:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_1[src],node_index_old2new_1[dst]])#重新映成新的序号
                    big_edge_label.append(int(edge_labels_1[label_index]))
                label_index = label_index + 1
            #之后要对后面的内容进行重新映射，所以就不用补了
            node_label_index = 0
            for iid in np.unique(instance_index_2):
                if iid != -1:
                    indices = (instance_index_2 == iid).nonzero()[0]
                    feats = dim256_2[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_2[node_label_index]))
                    node_index_old2new_2[iid] = node_index#把新序号要用上
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
                    # import pdb;pdb.set_trace()
            label_index = 0
            for edge in edge_index_2:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_2[src],node_index_old2new_2[dst]])
                    big_edge_label.append(int(edge_labels_2[label_index]))
                    label_index = label_index + 1
                    node_label_index = 0
            node_label_index = 0
            for iid in np.unique(instance_index_3):
                if iid != -1:
                    indices = (instance_index_3 == iid).nonzero()[0]
                    feats = dim256_3[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_3[node_label_index]))
                    node_index_old2new_3[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_3:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_3[src],node_index_old2new_3[dst]])
                    big_edge_label.append(int(edge_labels_3[label_index]))
                    label_index = label_index + 1
            
            node_label_index = 0
            for iid in np.unique(instance_index_4):
                if iid != -1:
                    indices = (instance_index_4 == iid).nonzero()[0]
                    feats = dim256_4[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_4[node_label_index]))
                    node_index_old2new_4[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_4:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_4[src],node_index_old2new_4[dst]])
                    big_edge_label.append(int(edge_labels_4[label_index]))
                    label_index = label_index + 1
            
            
            node_label_index = 0
            for iid in np.unique(instance_index_5):
                if iid != -1:
                    indices = (instance_index_5 == iid).nonzero()[0]
                    feats = dim256_5[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_5[node_label_index]))
                    node_index_old2new_5[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_5:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_5[src],node_index_old2new_5[dst]])
                    big_edge_label.append(int(edge_labels_5[label_index]))
                    label_index = label_index + 1
            
            node_label_index = 0
            for iid in np.unique(instance_index_6):
                if iid != -1:
                    indices = (instance_index_6 == iid).nonzero()[0]
                    feats = dim256_6[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_6[node_label_index]))
                    node_index_old2new_6[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_6:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_6[src],node_index_old2new_6[dst]])
                    big_edge_label.append(int(edge_labels_6[label_index]))
                    label_index = label_index + 1
            
            node_label_index = 0
            for iid in np.unique(instance_index_7):
                if iid != -1:
                    indices = (instance_index_7 == iid).nonzero()[0]
                    feats = dim256_7[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_7[node_label_index]))
                    node_index_old2new_7[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_7:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_7[src],node_index_old2new_7[dst]])
                    big_edge_label.append(int(edge_labels_7[label_index]))
                    label_index = label_index + 1
            
            node_label_index = 0
            for iid in np.unique(instance_index_8):
                if iid != -1:
                    indices = (instance_index_8 == iid).nonzero()[0]
                    feats = dim256_8[indices]
                    feats = torch.max(feats, 0, keepdim=True)[0]
                    node_feats= torch.cat((node_feats,feats))#确定是按顺序添加的
                    big_node_label.append(int(node_labels_8[node_label_index]))
                    node_index_old2new_8[iid] = node_index
                    node_index = node_index + 1
                    node_label_index = node_label_index + 1
            label_index = 0
            for edge in edge_index_8:
                src = int(edge[0])
                dst = int(edge[1])
                if src + dst != -2:
                    big_edge_index.append([node_index_old2new_8[src],node_index_old2new_8[dst]])
                    big_edge_label.append(int(edge_labels_8[label_index]))
                    label_index = label_index + 1
            #import pdb;pdb.set_trace()
            return node_feats,big_node_label,big_edge_index,big_edge_label,descriptor


def top_5_cm(preds,labels):
    rank1 = 0
    rank5 = 0
    rank10 = 0
    cm_top_5 = np.zeros(16,16)
    # 遍历数据集
    for (p,gt) in zip(preds,labels):
        # 通过降序对概率进行排序
        p = np.argsort(p)[::-1]
        # 检查真实标签是否落在top5中
        if gt in p[:10]:
            rank10 += 1
        if gt in p[:5]:
            rank5 += 1
        # 检验真实标签是否等于top1
        if gt == p[0]:
            rank1 += 1
            # 计算准确度
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    rank10 /= float(len(labels))
    return rank1,rank5,rank10

def macro_F1_edge(preds, labels):
    labels = [0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15] # 有哪几类
    precision = precision_score(preds, labels, labels=labels, average="macro") # average 指定为macro 


def Recall_function(cm):
    recall_average = 0
    index = 0
    index_sum  = 0
    for i in cm:
        if i.sum() != 0:
            recall_average += cm[index][index]/(i.sum())
            index_sum += 1
        index += 1
    recall_average = recall_average/index_sum
    return recall_average


def precision_function(cm):
    precision_average = 0
    index = 0
    index_sum  = 0
    for j in range(cm.shape[0]):
        if cm[:, j].sum() != 0:
            precision_average += cm[index][index]/(cm[:, j].sum())
            index_sum += 1
        index += 1
    precision_average = precision_average/index_sum
    return precision_average
    
def Recall_function_top_n(out_node, labels, top_k,list_index=None,pre=None):
    # pred_node = out_node.argmax(dim = 1)
    count_node = Counter(labels)
    pred_nodes = np.argsort(out_node,kind='stable') #[num_node, num_class]
    if list_index is not None:
        for i in range(len(list_index)):
            pred_nodes =np.where(pred_nodes == i,list_index[i]+100,pred_nodes)
        pred_nodes-=100
        # if len(list_index)>5:
        #     import pdb;pdb.set_trace()
        # if top_k==1:
        #     # print(pred_nodes[:,-1])
        #     if np.array_equal(pred_nodes[:,-1], pre)==False:
        #         import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    labels_unique = np.unique(labels)
    acc_sum = 0
    num_label = labels_unique.shape[0]
    index = 0
    count_node_acc = dict()
    for label in labels_unique:
        count_node_acc[label] = 0
    for index in range(labels.shape[0]):
        pred_node = pred_nodes[index]
        pred_node = pred_node[::-1]
        pred_node = pred_node[:top_k]
        node_label = labels[index]
        if node_label in pred_node:
            if node_label in count_node_acc:
                count_node_acc[node_label] += 1
            else:
                count_node_acc[node_label] = 1
    #在得到acc和总的数据后，可以开始计算平均了
    count_node_sum = 0
    acc_sum_mean = 0
    for label in labels_unique:
        count_node_sum += count_node[label]
    for label in labels_unique:
        # acc_sum += (count_node_acc[label]/count_node[label])*(count_node[label]/count_node_sum)
        acc_sum += count_node_acc[label]/count_node_sum
    for label in labels_unique:
        acc_sum_mean += count_node_acc[label]/count_node[label]
        #if(top_k==1):
            #print(count_node_acc[label]/count_node[label])
        #print("=========")
    Recall_top_k_mean = acc_sum_mean/num_label
    Recall_top_k = acc_sum
    return Recall_top_k, Recall_top_k_mean


def precision_function_top_n(out_node, labels, top_k):
    count_have_node = dict()
    count_actualy_have_node  = dict()
    pred_nodes = np.argsort(out_node)
    pred_nodes_new = np.zeros((1, top_k))#先开一行起头
    for index in range(labels.shape[0]):
        pred_nodes_a = pred_nodes[index][::-1]
        a = pred_nodes_a[:top_k]
        pred_nodes_new = np.row_stack((pred_nodes_new, a))
    pred_nodes_new = np.delete(pred_nodes_new, 0, 0)#把第一行删了  
    pred_nodes_new = pred_nodes_new.astype(np.int64) 
    for index in range(labels.shape[0]):
        pred_nodes = pred_nodes_new[index] #[3,2,1]
        node_label = labels[index]#[1]
        for number in pred_nodes:
            if number in count_have_node:
                count_have_node[number] += 1

            else:
                count_have_node[number] = 1
        if node_label in pred_nodes:
            if node_label in count_actualy_have_node:
                count_actualy_have_node[node_label] += 1
            else:
                count_actualy_have_node[node_label] = 1
    acc_sum = 0
    count_label = 0
    for label in count_actualy_have_node:
        acc_sum += count_actualy_have_node[label]/count_have_node[label]
    for label in count_have_node:
        count_label += 1
    precision_top_k = acc_sum/count_label
    return precision_top_k

def hand_make_cm_node(pred_node, labels):
    cm = np.zeros((27,27))
    for index in range(pred_node.shape[0]):
        cm[labels[index]][pred_node[index]] += 1
    return cm
def hand_make_cm_edge(pred_node, labels):
    cm = np.zeros((16,16))
    for index in range(pred_node.shape[0]):
        cm[labels[index]][pred_node[index]] += 1
    return cm



import torch
import torch.nn.functional as F

from torch import nn

# class BKDLoss(nn.Module):
#     def __init__(self):
#         super(BKDLoss, self).__init__()
#     def foward(self,new_chance_dic,old_chance_dic,old_stage_Mask,edge_labels,list_n,index,FreM):
#         if index == 0: return 0
#         Loss=0
#         e_new = new_chance_dic[i][old_stage_Mask[i]]
#         e_old = before_chance_dic[i][old_stage_Mask[i]]
        
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, logits, labels,mask=None,list_n=None):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        # logits = logits.permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        #节点这样处理效果好
        # if mask is  None:
        #     logits=torch.log(logits)
        # import pdb;pdb.set_trace()
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        try:
            assert(logits.size(0) == labels.size(0))
            assert(logits.size(1) == labels.size(1))
        except:
            import pdb;pdb.set_trace()
        batch_size = logits.size(0)
        labels_length = logits.size(2)
        seq_length = logits.size(1)
        # import pdb; pdb.set_trace()
        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        
        alpha=self.alpha
        
        if list_n is not None:
            labels_length=27
            alpha=2
       
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).to(self.device)
        
        #import pdb;pdb.set_trace()
        # print('============')
        #import pdb;pdb.set_trace()
        # label_onehot.scatter_(1, new_label, 1-self.kese)
        label_onehot.scatter_(1, new_label, 1)
        label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length
        #import pdb;pdb.set_trace()
        if list_n is not None:
            label_onehot=label_onehot[:,:,list_n]
        if mask is not None:
            mask = torch.tensor(mask).to(self.device)
            if (mask==1).sum()==0:
                return 0
            #import pdb;pdb.set_trace()
            mask = mask.unsqueeze(0)
            label_onehot=label_onehot[mask]
            logits=logits[mask]
        else:
            logits=logits.squeeze(0)
            label_onehot=label_onehot.squeeze(0)
       
        # calculate log
        #import pdb;pdb.set_trace()
        log_p = torch.log(logits+1e-7)
        #import pdb;pdb.set_trace()
        fl = -alpha *(label_onehot*(1-logits)**self.gamma) * log_p
        fl=fl.sum(1)
        #import pdb;pdb.set_trace()
        #cross_entropy = -label_onehot * torch.log(logits + 1e-7)
        #p_t=(logits * label_onehot).sum(dim=1)
        #fl = -self.alpha * (1 - p_t) ** self.gamma * torch.log(logits + 1e-6)
        #p_t = logits.gather(1, labels.long().unsqueeze(1)).squeeze(1)

        #log_p = torch.log(logits)
        



        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        # if mask is not None:
        #     mask = torch.tensor(mask).to(self.device)
        #     if (mask==1).sum()==0:
        #         return 0
        #     mask = mask.unsqueeze(0)
        #     fl=fl[mask]
        #import pdb;pdb.set_trace()
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
def parse_attn_structure(s):
    # s: '1SA+1CA+9SA'
    result = []
    for block in s.split('+'):
        num = int(''.join(filter(str.isdigit, block)))
        attn_type = ''.join(filter(str.isalpha, block))
        result.extend([attn_type] * num)
    return result

def evaluate_topk_predicate_26_classes(rels_preds, gt_edges, multi_rel_outputs, topk, 
                                      confidence_threshold=0.5, epsilon=0.02):
    """
    专门针对26类关系（不包括None类）的评估
    """
    res = []
    # 只统计26个关系类别的准确率
    num_rel_classes = 26
    per_class_stats = {
        i: {
            'total_samples': 0,      # 包含该关系的样本数
            'support': 0,            # 该关系作为正样本的次数
            'correct_top1': 0,       # top-1预测正确的次数
            'correct_topk': 0,       # top-k预测正确的次数
            'predicted_as_top1': 0,  # 被预测为top-1的次数
        } for i in range(num_rel_classes)
    }
    
    for rel_idx, rel_pred in enumerate(rels_preds):
        # 复制预测结果
        rel_pred_copy = rel_pred.clone()
        
        # 处理None类的特殊逻辑（如果最大置信度低于阈值，则认为是None）
        if multi_rel_outputs:
            if rel_pred_copy.max() < confidence_threshold:
                # 这种情况下相当于预测为None（不在26类中）
                max_pred_idx = -1  # 表示None
            else:
                max_pred_idx = torch.argmax(rel_pred_copy).item()
        else:
            max_pred_idx = torch.argmax(rel_pred_copy).item()
        
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred_copy, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel_idx][2] if len(gt_edges[rel_idx]) > 2 else []
        
        # 只统计26个关系类别的表现
        for class_id in range(num_rel_classes):
            # 统计support（该关系在真实标签中出现的次数）
            if class_id in rels_target:
                per_class_stats[class_id]['support'] += 1
                per_class_stats[class_id]['total_samples'] += 1
            # 对于负样本，也计入统计
            elif len(rels_target) == 0 or all(t < num_rel_classes for t in rels_target):
                per_class_stats[class_id]['total_samples'] += 1
        
        # Top-1预测统计
        if max_pred_idx >= 0 and max_pred_idx < num_rel_classes:
            per_class_stats[max_pred_idx]['predicted_as_top1'] += 1
            # 如果预测正确
            if max_pred_idx in rels_target:
                per_class_stats[max_pred_idx]['correct_top1'] += 1
        
        # Top-k预测统计
        topk_indices = sorted_idx[:topk].tolist()
        # 过滤掉可能的None预测（超出26类范围的）
        valid_topk_indices = [idx for idx in topk_indices if 0 <= idx < num_rel_classes]
        
        for class_id in range(num_rel_classes):
            # 检查该类别是否在top-k预测中且是正确的
            if class_id in valid_topk_indices and class_id in rels_target:
                per_class_stats[class_id]['correct_topk'] += 1
        
        # 原始排名计算逻辑（保持原有的处理方式）
        if len(rels_target) == 0:
            # 无真实关系
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item() + 1
            temp_topk.append(index)
        else:
            # 有真实关系，但只处理26类内的关系
            valid_targets = [gt for gt in rels_target if 0 <= gt < num_rel_classes]
            for gt in valid_targets:
                index = 1
                for idx in sorted_idx:
                    if idx >= num_rel_classes:  # 跳过None类
                        continue
                    if rel_pred_copy[gt] >= rel_pred_copy[idx] or index > topk:
                        break
                    index += 1
                temp_topk.append(index)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
    
    return np.asarray(res), per_class_stats

# 简化版本 - 更清晰地处理26类关系
def evaluate_26_class_accuracy(rels_preds, gt_edges, topk=1, confidence_threshold=0.5):
    """
    简化版26类关系准确率计算
    """
    num_rel_classes = 26
    per_class_stats = {
        i: {
            'support': 0,            # 该关系的正样本数
            'correct_topk': 0,       # top-k正确预测数（召回率分子）
            'predicted_count': 0,    # 该关系被预测的次数（精确率分母）
            'correct_predictions': 0, # 该关系被正确预测的次数（精确率分子）
        } for i in range(num_rel_classes)
    }
    
    for rel_idx, rel_pred in enumerate(rels_preds):
        # 获取预测结果
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
        rels_target = gt_edges[rel_idx][2] if len(gt_edges[rel_idx]) > 2 else []
        
        # 只考虑26类关系（0-25）
        valid_targets = [t for t in rels_target if 0 <= t < 26]
        topk_predictions = [idx for idx in sorted_idx[:topk].tolist() if 0 <= idx < 26]
        
        # 统计每个类别的support
        for class_id in range(num_rel_classes):
            if class_id in valid_targets:
                per_class_stats[class_id]['support'] += 1
        
        # 统计预测情况
        for pred_class in topk_predictions:
            per_class_stats[pred_class]['predicted_count'] += 1
            if pred_class in valid_targets:
                per_class_stats[pred_class]['correct_predictions'] += 1
        
        # 统计召回率
        for target_class in valid_targets:
            if target_class in topk_predictions:
                per_class_stats[target_class]['correct_topk'] += 1
    
    return per_class_stats

# 计算最终的per-class指标
def calculate_final_26_class_metrics(per_class_stats):
    """
    根据统计结果计算最终的指标
    """
    results = {}
    for class_id, stats in per_class_stats.items():
        # 精确率：预测为该类且正确的样本 / 预测为该类的样本
        precision = stats['correct_predictions'] / stats['predicted_count'] if stats['predicted_count'] > 0 else 0
        
        # 召回率：该类被正确预测的样本 / 该类的总样本
        recall = stats['correct_topk'] / stats['support'] if stats['support'] > 0 else 0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': stats['support'],
            'predicted_count': stats['predicted_count'],
            'correct_predictions': stats['correct_predictions'],
            'correct_topk': stats['correct_topk']
        }
    
    return results

# 使用示例
def print_26_class_results(per_class_metrics, relation_names):
    """
    打印26类关系的结果
    """
    print("26-Class Relationship Accuracy Analysis")
    print("=" * 100)
    print(f"{'Relation':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 100)
    
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for class_id in range(26):
        metrics = per_class_metrics[class_id]
        name = relation_names[class_id] if class_id < len(relation_names) else f"Relation_{class_id}"
        
        print(f"{name:<25} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['support']:<8}")
        
        # 计算加权平均
        total_support += metrics['support']
        weighted_precision += metrics['precision'] * metrics['support']
        weighted_recall += metrics['recall'] * metrics['support']
        weighted_f1 += metrics['f1_score'] * metrics['support']
    
    print("-" * 100)
    mean_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values() if metrics['support'] > 0])
    mean_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values() if metrics['support'] > 0])
    mean_f1 = np.mean([metrics['f1_score'] for metrics in per_class_metrics.values() if metrics['support'] > 0])
    
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f}")
    
    if total_support > 0:
        print(f"Weighted Precision: {weighted_precision/total_support:.4f}")
        print(f"Weighted Recall: {weighted_recall/total_support:.4f}")
        print(f"Weighted F1-Score: {weighted_f1/total_support:.4f}")