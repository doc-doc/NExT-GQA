# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch

class EncoderVid(nn.Module):
    def __init__(self, feat_dim, bbox_dim, feat_hidden, pos_hidden, input_dropout_p=0.3):
        
        super(EncoderVid, self).__init__()
        self.dim_feat = feat_dim
        self.dim_bbox = bbox_dim
        self.dim_hidden = feat_hidden
        self.input_dropout_p = input_dropout_p

        input_dim = feat_dim

        input_dim += pos_hidden
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.dim_bbox, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            nn.Conv2d(pos_hidden, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            
        )

        self.tohid = nn.Sequential(
            nn.Linear(feat_dim+pos_hidden, feat_hidden),
            nn.ELU(inplace=True))
        
        # self.roi_conv = nn.Sequential(
        #     nn.Conv1d(feat_dim, feat_hidden, kernel_size=3, padding=1),
        #     nn.ELU(inplace=True)
        # )

        # self.roi_conv = nn.Sequential(
        #     nn.Conv2d(4, 4, kernel_size=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        # )


    def forward(self, video_o):
        
        bsize, numc, numf, numr, fdim =  video_o.shape
       
        video_o = video_o.view(bsize, numc*numf, numr, fdim)
        roi_feat = video_o[:,:,:, :self.dim_feat]
        roi_bbox = video_o[:,:,:, self.dim_feat:(self.dim_feat+self.dim_bbox)]
        
        bbox_pos = self.bbox_conv(roi_bbox.permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
 
        bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)

        bbox_feat = self.tohid(bbox_features)
        
        return bbox_feat
