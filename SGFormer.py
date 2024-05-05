from enum import Flag
import imp
#from macpath import dirname
from re import X
from tkinter import HIDDEN, Variable
from turtle import Turtle, forward
from sqlalchemy import false, true
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import dgl.function as fn
#from EdgeGCN import EdgeGCN

from help_function import edge_feats_initialization

from math import sqrt
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import dgl.function as fn
#from EdgeGCN import EdgeGCN
from tqdm import tqdm
import os
import sys
from six.moves.urllib.request import urlretrieve
import six
import zipfile
import array
import torch
#from bert_serving.client import BertClient


class SALayer(nn.Module):
    def __init__(self,in_dim,num_heads,hidden_dim,dropout):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hidden_dim =  hidden_dim
        self.use_bias = False
        self.dropout = dropout
        self.SAattention = MultiHeadAttentionLayer_with_edge(in_dim, self.hidden_dim//self.num_heads, num_heads, self.use_bias)
        self.O_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.O_e = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.layer_norm1_h = nn.LayerNorm(self.hidden_dim)
        self.layer_norm1_e = nn.LayerNorm(self.hidden_dim)
        
        self.FFN_h_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self.FFN_h_layer2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self.FFN_e_layer2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        self.layer_norm2_h = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2_e = nn.LayerNorm(self.hidden_dim)
    def forward(self,g,node_feats, edge_feats):
        node_in1 = node_feats # for first residual connection
        edge_in1 = edge_feats # for first residual connection
        
        # multi-head attention out
        node_attn_out, edge_attn_out = self.SAattention(g, node_feats, edge_feats)
        
        node_feats = node_attn_out.view(-1, self.hidden_dim)
        edge_feats = edge_attn_out.view(-1, self.hidden_dim)
        
        node_feats = F.dropout(node_feats, self.dropout, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout, training=self.training)

        node_feats = self.O_h(node_feats)
        edge_feats = self.O_e(edge_feats)

        node_feats = node_in1 + node_feats # residual connection
        edge_feats = edge_in1 + edge_feats # residual connection

        node_feats = self.layer_norm1_h(node_feats)#layernorm
        edge_feats = self.layer_norm1_e(edge_feats)#layernorm

        node_in2 = node_feats # for second residual connection
        edge_in2 = edge_feats # for second residual connection

        # FFN for h
        node_feats = self.FFN_h_layer1(node_feats)
        node_feats = F.relu(node_feats)
        node_feats = F.dropout(node_feats, self.dropout, training=self.training)
        node_feats = self.FFN_h_layer2(node_feats)

        # FFN for e
        edge_feats = self.FFN_e_layer1(edge_feats)
        edge_feats = F.relu(edge_feats)
        edge_feats = F.dropout(edge_feats, self.dropout, training=self.training)
        edge_feats = self.FFN_e_layer2(edge_feats)
        
        node_feats = node_in2 + node_feats # residual connection       
        edge_feats = edge_in2 + edge_feats # residual connection  

        node_feats = self.layer_norm2_h(node_feats)
        edge_feats = self.layer_norm2_e(edge_feats)

        return node_feats,edge_feats

class MultiHeadAttentionLayer_with_edge(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))#这个分数要拿过去更新e
        
        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))#将节点的V乘以分数，然后再将结果给V
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out



class CA_word_layer(nn.Module):
    def __init__(self,node_in_dim, hidden_dim,word_in_dim,dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_in_dim = word_in_dim
        self.dropout = dropout
        self.node_in_dim = node_in_dim
        self.linear_q = nn.Linear(self.node_in_dim, self.hidden_dim, bias=False)
        self.linear_k = nn.Linear(self.word_in_dim, self.hidden_dim, bias=False)
        self.linear_v = nn.Linear(self.word_in_dim, self.hidden_dim, bias=False)
        self.scaling = 1/sqrt(self.hidden_dim)
        
        # self.layer_norm_word_embedding = nn.LayerNorm(self.hidden_dim)
        self.layer_norm1_node = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2_node = nn.LayerNorm(self.hidden_dim)
        
        self.FFN_node_layer1 = nn.Linear(self.hidden_dim,self.hidden_dim*2,bias=False)
        self.FFN_node_layer2 = nn.Linear(self.hidden_dim*2, self.hidden_dim,bias=False)
        
        self.att_linear = nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)
        
        self.concat_layer1 = nn.Linear(300,80,bias=False)
        self.concat_layer2 = nn.Linear(80*2,80,bias=False)
    def forward(self,feats_Q,word_feats_KV):
        x_in  = feats_Q
        num_Q, dim_in = feats_Q.shape
        num_K, dim_in = word_feats_KV.shape 
        
        dk = self.hidden_dim  # dim_k of each head
        dv = self.hidden_dim  # dim_v of each head

        q = self.linear_q(feats_Q)#num_Q, hidden_dim
        k = self.linear_k(word_feats_KV)
        v = self.linear_v(word_feats_KV)#num_KV,hidden_dim

        dist = torch.matmul(q, k.transpose(1, 0)) * self.scaling  #num_Q, num_K
        node_logits = F.softmax(dist, dim=-1)  # num_Q, num_K

        att = torch.matmul(node_logits, v)  # num_Q, hidden_dim
        
        att = self.att_linear(att)
        
        node_feats = x_in + att
        node_feats = self.layer_norm1_node(node_feats)
        
        x_in_2 = node_feats
        
        node_feats = self.FFN_node_layer1(node_feats)
        node_feats = F.relu(node_feats)
        node_feats = F.dropout(node_feats,self.dropout,training=self.training)
        node_feats = self.FFN_node_layer2(node_feats)
        
        node_feats = x_in_2 + node_feats
        node_feats = self.layer_norm2_node(node_feats)
            
        return node_logits,node_feats

class SGlayer(nn.Module):
    def __init__(self,node_in_dim,hidden_dim,dropout,num_head,word_in_dim,embedding_type):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_head = num_head
        self.word_in_dim = word_in_dim
        self.embedding_type = embedding_type
        self.SALayer = SALayer(self.node_in_dim,self.num_head,self.hidden_dim,self.dropout)
        self.CA_word_layer = CA_word_layer(self.hidden_dim,self.hidden_dim,self.word_in_dim,self.dropout)
    def forward(self,g,node_feats,edge_feats,word_feats):
        node_feats, edge_feats = self.SALayer(g,node_feats,edge_feats)
        if self.embedding_type == 'none':
            return node_feats,edge_feats
        elif self.embedding_type == 'glove':
            node_logits,node_feats= self.CA_word_layer(node_feats,word_feats)
            node_logits = node_logits.unsqueeze(0)
            return node_logits,node_feats,edge_feats
        elif self.embedding_type == 'bert_1024' or self.embedding_type == 'bert_768' :
            node_logits,node_feats= self.CA_word_layer(node_feats,word_feats)
            node_logits = node_logits.unsqueeze(0)
            return node_logits,node_feats,edge_feats
            
class SGFormer(nn.Module):
    def __init__(self,node_in_dim,hidden_dim,dropout,num_head,num_layers,embedding_type = 'none'):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.embedding_type = embedding_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_head = num_head
        if self.embedding_type == 'glove':
            self.word_in_dim = 300
        elif self.embedding_type == 'bert_1024':
            self.word_in_dim  = 1024
        elif self.embedding_type == 'bert_768':
            self.word_in_dim  = 768
        elif self.embedding_type == 'none':
            self.word_in_dim = 300
        elif self.embedding_type == 'none_CA':
            self.word_in_dim = 768
        elif self.embedding_type == 'none_CA_CLIP':
            self.word_in_dim = 512
        
        self.layers = nn.ModuleList([SGlayer(self.hidden_dim,self.hidden_dim,self.dropout,self.num_head,self.word_in_dim,embedding_type=self.embedding_type) for _ in range(self.num_layers)])
        
        self.layer_change_1 = nn.ModuleList([SALayer(self.hidden_dim,self.num_head,self.hidden_dim,self.dropout) for _ in range(self.num_layers-1)])
        self.CAword_layer = CA_word_layer(self.hidden_dim,self.hidden_dim,self.word_in_dim,self.dropout)
        self.layer_change_2 = SALayer(self.hidden_dim,self.num_head,self.hidden_dim,self.dropout)
        
        self.layer_change_3 = nn.ModuleList([SALayer(self.hidden_dim,self.num_head,self.hidden_dim,self.dropout) for _ in range(8)])
        self.CAword_layer = CA_word_layer(self.hidden_dim,self.hidden_dim,self.word_in_dim,self.dropout)
        self.layer_change_4 = nn.ModuleList([SALayer(self.hidden_dim,self.num_head,self.hidden_dim,self.dropout) for _ in range(0)])
        
        self.embedding_node = nn.Linear(self.node_in_dim,self.hidden_dim)
        self.embedding_edge = nn.Linear(self.node_in_dim*2,self.hidden_dim)
        
        self.in_feats_dropout = nn.Dropout(self.dropout)
        
        if self.embedding_type == 'glove':
            self.obj_classes = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "counter", "shelf", "curtain", "pillow", "clothes", "ceiling", "fridge", "tv", "towel", "plant", "box", "nightstand", "toilet", "sink", "lamp", "bathtub", "blanket"]
            self.GLOVE_DIR = "./"
            embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.GLOVE_DIR)
            self.object_word_embed=nn.Parameter(torch.rand(1,300))  # [1, 300]
            self.register_buffer('word_embedding_26_buffer',embed_vecs)
            self.laye_norm_word_embedding = nn.LayerNorm(self.word_in_dim)
        num_node_in_embeddings = self.hidden_dim
        node_mid_channels = num_node_in_embeddings // 2
        self.node_linear1 = nn.Linear(num_node_in_embeddings, node_mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(nn.BatchNorm1d(node_mid_channels), nn.LeakyReLU(0.2), nn.Dropout())
        self.node_linear2 = nn.Linear(node_mid_channels, 27, bias=False)

        num_edge_in_embeddings = self.hidden_dim
        edge_mid_channels = num_edge_in_embeddings // 2
        self.edge_linear1 = nn.Linear(num_edge_in_embeddings, edge_mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(edge_mid_channels),nn.LeakyReLU(0.2), nn.Dropout())
        self.edge_linear2 = nn.Linear(edge_mid_channels, 16, bias=False)
    def forward(self,node_feats,edge_index):
        src = edge_index.t()[0]
        dst = edge_index.t()[1]
        g = dgl.DGLGraph((src, dst))
        edge_feats = edge_feats_initialization(node_feats, edge_index)#[num_edge, 160]
        node_feats = self.embedding_node(node_feats)#[num+_node, 80]
        
        edge_feats = self.embedding_edge(edge_feats)
        g.ndata['feats'] = node_feats
        g.edata['feats'] = edge_feats
        
        if self.embedding_type != 'none':
            embed_vecs_26_buffer = self.get_buffer('word_embedding_26_buffer')
            embed_vecs_25 = embed_vecs_26_buffer[:25,:]
            embed_vecs_No27 = embed_vecs_26_buffer[25,:].unsqueeze(0)
            embedding_object = self.object_word_embed
            embed_vecs_26 = torch.cat([embed_vecs_25,embedding_object],dim=0)
            embed_vecs_27 = torch.cat([embed_vecs_26, embed_vecs_No27],dim=0)
            word_feats = embed_vecs_27
            if self.embedding_type == 'glove':
                word_feats = self.laye_norm_word_embedding(word_feats)
            elif self.embedding_type == 'bert_1024' or self.embedding_type == 'bert_768' or self.embedding_type == 'none_CA' or self.embedding_type == 'none_CA_CLIP' :
                word_feats = self.laye_norm_word_embedding_bert(word_feats)
        else:
            word_feats = 0
        flag = 0
        if self.embedding_type == 'none':
            node_logits_log = 0
            for layer in self.layers:
                node_feats,edge_feats = layer(g,node_feats,edge_feats,word_feats)
            
            node_logits_log = node_logits_log.unsqueeze(0)
        

        node_feats = node_feats.unsqueeze(0)
        edge_feats = edge_feats.unsqueeze(0)  
        

        node_x = self.node_linear1(node_feats)
        node_x = self.node_BnReluDp(node_x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(node_x)
        node_logits = node_logits.squeeze(0)
        node_logits = F.softmax(node_logits, dim=1)

        edge_x = self.edge_linear1(edge_feats)
        edge_x = self.edge_BnReluDp(edge_x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(edge_x)
        # return {'node_feats': node_feats, 'edge_feats': edge_feats}
        edge_logits = torch.squeeze(edge_logits,0)
        edge_logits = F.softmax(edge_logits, dim=1)
        
        
        
        return node_logits_log,node_logits,edge_logits


""" 
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors
def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {

        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
                    object_class = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "counter", "shelf", "curtain", "pillow", "clothes", "ceiling", "fridge", "tv", "towel", "plant", "box", "nightstand", "toilet", "sink", "lamp", "bathtub", "blanket"]
                    if word in object_class:
                        wv_arr.extend(float(x) for x in entries)
                        wv_tokens.append(word)
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            
    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret
def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner