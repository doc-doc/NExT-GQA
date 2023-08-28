'''
Author: Xiao Junbin
Date: 2022-11-21 22:10:54
LastEditTime: 2022-11-26 22:18:17
LastEditors: Xiao Junbin
Description: 
FilePath: /CoVQA/model/video_model.py
'''
from model.swin_transformer import SwinTransformer
import torch
from model.config import get_config

class Swin(torch.nn.Module):
    def __init__(self, use_head=False):
        super(Swin, self).__init__()
        
        model_path = "../data/pretrain_models/swin"
        model_name = "swin_base_patch4_window7_224_22kto1k_finetune"
        cfg_file = f'{model_path}/{model_name}.yaml'
        config = get_config(cfg_file)
         
        self.swin = SwinTransformer(embed_dim=config.MODEL.SWIN.EMBED_DIM, 
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    use_head=use_head)
        pt_weights = torch.load(f'{model_path}/{model_name}.pth', map_location="cpu")["model"]
        self.swin.load_state_dict(pt_weights)
        
        self.freeze()
        # self.emb_cls = T.nn.Parameter(0.02*T.randn(1, 1, 1, 768))
        # self.emb_pos = T.nn.Parameter(0.02*T.randn(1, 1, 1+14**2, 768))
        # self.emb_len = T.nn.Parameter(0.02*T.randn(1, 6, 1, 768))
        # self.norm = T.nn.LayerNorm(768)
    
    def forward(self, img):
        # _B, _C, _H, _W = img.shape
        f_img = self.swin(img)
        
        return f_img
    

    def freeze(self):
        for name, param in self.swin.named_parameters():
            param.requires_grad = False

