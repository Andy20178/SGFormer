import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
from math import sqrt
import os
import sys
import six
import zipfile
import array
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

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
class Self_Attention_Layer_with_Edge(nn.Module):
    def __init__(self,in_dim,num_heads,hidden_dim,dropout, with_edge_feature=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hidden_dim =  hidden_dim
        self.use_bias = False
        self.dropout = dropout
        self.with_edge_feature = with_edge_feature
        self.MALE = MultiHead_Attention_Layer_with_Edge(in_dim, self.hidden_dim//self.num_heads, num_heads, self.use_bias, with_edge_feature=self.with_edge_feature)
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
    def forward(self, g, node_feats, edge_feats):
        node_in1 = node_feats # for first residual connection
        edge_in1 = edge_feats # for first residual connection
        
        # multi-head attention out
        node_attn_out, edge_attn_out = self.MALE(g, node_feats, edge_feats)
        
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

class MultiHead_Attention_Layer_with_Edge(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, with_edge_feature=True):
        super().__init__()
        self.with_edge_feature = with_edge_feature
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
        
        if self.with_edge_feature:
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

class Cross_Attention_Layer_with_Word(nn.Module):
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
        
        self.att_projection_linear = nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)
        
        self.concat_layer1 = nn.Linear(300,80,bias=False)
        self.concat_layer2 = nn.Linear(80*2,80,bias=False)
    def forward(self,feats_Q,word_feats_KV):
        #使用word_feats之前，需要layernorm
        # word_feats_KV = self.layer_norm_word_embedding(word_feats_KV)
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
        # import pdb;pdb.set_trace()
        #att有问题
        #为了Concat的操作
        # node_index = torch.argmax(node_logits,dim=1)
        # node_feats_concat = word_feats_KV[node_index]
        # node_feats_concat = self.concat_layer1(node_feats_concat)
        # node_feats = torch.cat([x_in,node_feats_concat],dim=1)
        # node_feats = self.concat_layer2(node_feats)
        att = torch.matmul(node_logits, v)  # num_Q, hidden_dim
        
        att = self.att_projection_linear(att)
        
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

class SGFormer_Single_Layer(nn.Module):
    def __init__(self,node_in_dim,hidden_dim,dropout,num_head,word_in_dim,embedding_type=None, layer_type='Self_Attention_Layer_with_Edge'):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_head = num_head
        self.word_in_dim = word_in_dim
        self.embedding_type = embedding_type
        self.layer_type = layer_type
        if layer_type == 'Self_Attention_Layer_with_Edge':
            self.Self_Attention_Layer_with_Edge = Self_Attention_Layer_with_Edge(self.node_in_dim,self.num_head,self.hidden_dim,self.dropout)
        elif layer_type == 'Self_Attention_Layer_with_Edge_no_edge':
            self.Self_Attention_Layer_with_Edge = Self_Attention_Layer_with_Edge(self.node_in_dim,self.num_head,self.hidden_dim,self.dropout,with_edge_feature=False)
        elif layer_type == 'Cross_Attention_Layer_with_Word':
            self.Cross_Attention_Layer_with_Word = Cross_Attention_Layer_with_Word(self.hidden_dim,self.hidden_dim,self.word_in_dim,self.dropout)
    def forward(self, g, node_feats, edge_feats, word_feats):
        if self.layer_type == 'Self_Attention_Layer_with_Edge' or self.layer_type == 'Self_Attention_Layer_with_Edge_no_edge':
            # print(self.layer_type)
            node_feats, edge_feats = self.Self_Attention_Layer_with_Edge(g,node_feats,edge_feats)
            return None, node_feats, edge_feats
        elif self.layer_type == 'Cross_Attention_Layer_with_Word':
            node_logits,node_feats= self.Cross_Attention_Layer_with_Word(node_feats,word_feats)
            node_logits = node_logits.unsqueeze(0)
            return node_logits, node_feats, edge_feats




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