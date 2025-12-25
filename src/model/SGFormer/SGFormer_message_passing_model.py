from torch import nn
from src.model.SGFormer.utils import SGFormer_Single_Layer
from src.model.SGFormer.help_function import parse_attn_structure
class SGFormer_message_passing_model(nn.Module):
    def __init__(self,config):
        super(SGFormer_message_passing_model, self).__init__()
        layer_type_map = {
            'SA': 'Self_Attention_Layer_with_Edge',
            'CA': 'Cross_Attention_Layer_with_Word',
            'SAnoedge': 'Self_Attention_Layer_with_Edge_no_edge'
        }
        self.config = config
        self.attn_list = parse_attn_structure(self.config.MODEL.SGFormer_attn_structure)
        # import pdb;pdb.set_trace()
        self.layers = nn.ModuleList([
            SGFormer_Single_Layer(
                self.config.MODEL.SGFormer_hidden_dim,
                self.config.MODEL.SGFormer_hidden_dim,
                self.config.MODEL.SGFormer_dropout,
                self.config.MODEL.SGFormer_num_head,
                self.config.MODEL.SGFormer_word_in_dim,
                embedding_type=self.config.MODEL.embedding_type,
                layer_type=layer_type_map[attn_type]
            )
            for attn_type in self.attn_list
        ])
    def forward(self,g, node_feats,edge_feats,word_feats):
        for layer in self.layers:
            if layer.layer_type == 'Cross_Attention_Layer_with_Word':
                node_logits_word, node_feats,edge_feats = layer(g, node_feats,edge_feats,word_feats)
                node_logits_word = node_logits_word.unsqueeze(0)
            else:
                _, node_feats,edge_feats = layer(g,node_feats,edge_feats,word_feats)
        return node_logits_word, node_feats,edge_feats