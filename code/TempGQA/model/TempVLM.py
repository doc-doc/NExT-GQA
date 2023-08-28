from model.EncoderVid import EncoderVid
from transformers.activations import gelu
import torch.nn as nn
import numpy as np
import torch
import math
from model.language_model import Bert, LanModel
# from model.video_model import Swin
import copy
from transformers.modeling_outputs import BaseModelOutput
from transformers import BertConfig
# from transformers import DistilBertConfig
from transformers import RobertaConfig
from transformers import DebertaConfig
from util import get_mask
import torch.nn.functional as F

import h5py
import os.path as osp
from .transformer import Transformer, Embeddings


class VQA(nn.Module):
    def __init__(
        self,
        bert_tokenizer,
        feature_dim=1024,
        word_dim=768,
        N=2,
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        Q=20,
        T=20,
        vocab_size=50265,
        baseline="",
        n_negs=1,
        feat_type='CLIP',
        lan='RoBERTa',
        prop_num=1,
        sigma=9
    ):
        """
        :param feature_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param N: number of transformer layers
        :param h: number of transformer heads
        :param d_model: dimension for the transformer and final embedding
        :param d_ff: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param vocab_size: size of the vocabulary for the masked language modeling head
        :param baseline: set as "qa" not to use the video
        """
        super(VQA, self).__init__()
        self.baseline = baseline
        self.Q = Q
        self.T = T
        self.n_negs = n_negs
        d_pos = 128
        
        # It is more efficient to extract the features offline
        # self.encode_vid = EncoderVid(feat_dim=feature_dim,
        #                             bbox_dim=5, 
        #                             feat_hidden=d_model, 
        #                             pos_hidden=d_pos)
        # self.swin_2d = Swin()
        # self.swin_2d.eval()

        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        #####################clip position###################
        self.position_v = Embeddings(d_model, 0, T, dropout, True, d_pos)
        # self.position_qv = Embeddings(d_model, self.Q, T, dropout, True, d_pos)
        #####################hie position###################
        if lan == 'BERT':
            pt_config_name = 'bert-base-uncased'
            config = BertConfig
        elif lan == 'RoBERTa':
            pt_config_name = 'roberta-base'
            config = RobertaConfig
        elif lan == 'DeBERTa':
            pt_config_name = 'microsoft/deberta-base'
            config = DebertaConfig
        
        self.config = config.from_pretrained(
            pt_config_name,
            num_hidden_layers=N,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            intermediate_size=d_ff,
            num_attention_heads=h,
        )

        self.mmt = Transformer(self.config)
        
        
        # self.vproj = nn.Sequential(
        #                           nn.Dropout(dropout), 
        #                           nn.Linear(d_model, d_model)
        #                           )
        self.vpos_proj = nn.Sequential(
                                nn.Dropout(dropout), 
                                nn.Linear(d_model, d_model)
                                )

        # weight initialization
        self.apply(self._init_weights)
        self.answer_embeddings = None

        # answer modules
        self.lang_model = LanModel(bert_tokenizer, lan=lan, out_dim=d_model)

        # self.pred_vec = nn.Parameter(torch.zeros(feature_dim).float(), requires_grad=True)
        
        
        self.satt_pool_frame = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
                nn.Softmax(dim=-2))
        
        if baseline in ['NG+', 'NG']:
            self.prop_num = prop_num
            self.sigma = sigma
            
            self.logit_gauss = nn.Sequential(
                                    nn.Dropout(dropout), 
                                    nn.Linear(d_model, self.prop_num*2),
                                    nn.Sigmoid(),
                                    )

            self.satt_pool_frame_gs = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.Tanh(),
                    nn.Linear(d_model // 2, 1),
                    nn.Softmax(dim=-2))
        # self.vtrans = Transformer(self.config)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    def get_answer_embedding(self, answer):
        answer_g, answer = self.lang_model(answer)
        return answer_g, answer 

   
    def get_answer_seg_embedding(self, answer, seq_len, seg_feats, seg_num):
        answer_g, answer = self.lang_model(answer, seq_len, seg_feats, seg_num)
        return answer_g, answer 


    def get_question_embedding(self, question):
        question = self.linear_question(question)
        question = gelu(question)
        question = self.norm_question(question)
        return question
    

    def get_vframe_embedding(self, video):
        
        # _B, _T, _C, _H, _W = video.shape
        # video_f = self.swin_2d(video.view(_B*_T, _C, _H, _W))
        # print(video.shape)
        # video_f = video.view(_B, _T, -1)
        
        # bsize, fnum, fdim = video.shape
        # pred_vec = self.pred_vec.view(1, 1, -1).expand(bsize, 1, -1)
        # video = torch.cat([pred_vec, video], dim=1)
        
        video_f = self.linear_video(video)
        video_f = gelu(video_f)
        video_f = self.norm_video(video_f) #(bs, numc, numf, dmodel)
        
        return video_f

    def generate_gauss_weight(self, center, width):
        # code copied from https://github.com/minghangz/cpl
        weight = torch.linspace(0, 1, self.T)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma
        
        w = 0.3989422804014327 #1/(math.sqrt(2*math.pi))
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    
    def forward(self, video, video_mask, question=None, question_mask=None, 
                answer=None, answer_id=None, gauss_weight=None, stage='GD'):
        """
        """
        answer_g, answer_w = (
            self.get_answer_embedding(answer)
            if answer is not None
            else self.answer_embeddings
        )
        
        video_f = self.get_vframe_embedding(video)
        video_fp = self.position_v(video_f)

        if self.baseline == 'posthoc':
            trans_vpos = self.mmt(x=video_fp, attn_mask=video_mask)[0]
            fatt = self.satt_pool_frame(trans_vpos)
            global_feat = torch.sum(trans_vpos*fatt, dim=1)
            fusion_proj = self.vpos_proj(global_feat)
            outs = {"fatt": fatt}
        else:
            query_mask = video_mask[:,0].unsqueeze(1) #borrow from video mask
            qv_mask = torch.cat([query_mask, video_mask], dim=1)

            if stage == 'GD':
                # bsize, n_ans, seq_l, _ = answer_w.shape
                # ids = [(i*n_ans)+idx for i, idx in enumerate(answer_id)]
                # qsn_g = answer_g.view(bsize*n_ans, -1)[ids].view(bsize, -1)
                qsn_g = answer_g[torch.arange(answer_g.shape[0]), answer_id.squeeze(), :]
            else:
                # Max pool over candidate QAs to approximate the query QAs
                qsn_g = answer_g.max(dim=1)[0]
                # qsn_g, qsn_w = self.lang_model(question)
            
            query_proxy = qsn_g.unsqueeze(1)

            qv_cat = torch.cat([video_fp, query_proxy], dim=1)
            
            attended_qv = self.mmt(x=qv_cat, attn_mask=qv_mask)[0]

            fatt_gs = self.satt_pool_frame_gs(attended_qv)
            pooled_qv_feat = torch.sum(attended_qv*fatt_gs, dim=1)
            
            logits_cw = self.logit_gauss(pooled_qv_feat).view(-1, 2)
            
            gauss_c, gauss_w = logits_cw[:, 0], logits_cw[:, 1]
            gauss_weight = self.generate_gauss_weight(gauss_c, gauss_w)

            # align the query with the positive temporal proposals
            bsize = video_f.shape[0]
            props_feat = video_fp.unsqueeze(1) \
                .expand(bsize, self.prop_num, -1, -1).contiguous().view(bsize*self.prop_num, self.T, -1)
            props_mask = video_mask.unsqueeze(1) \
                .expand(bsize, self.prop_num, -1).contiguous().view(bsize*self.prop_num, -1)
            
            trans_vpos = self.mmt(x=props_feat, attn_mask=props_mask, gauss_weight=gauss_weight)[0]
            
            fatt = self.satt_pool_frame(trans_vpos)
            global_feat = torch.sum(trans_vpos*fatt, dim=1)
            fusion_proj = self.vpos_proj(global_feat)

            answer_g = answer_g.unsqueeze(1) \
                .expand(bsize, self.prop_num, -1, -1).contiguous().view(bsize*self.prop_num, answer_g.shape[-2], -1)
            
            outs = {'gcenter':gauss_c, 'gwidth':gauss_w, 'gweight':gauss_weight, 'fatt':fatt}

        if fusion_proj is not None and answer_g.device != fusion_proj.device:
            answer_g = answer_g.to(fusion_proj.device)
        else:
            return fusion_proj, answer_g, outs

        