'''
Author: Xiao Junbin
Date: 2022-11-28 15:17:27
LastEditTime: 2022-11-28 17:23:42
LastEditors: Xiao Junbin
Description: test swin feature
FilePath: /CoVQA/tools/extract_feature.py
'''
import sys
sys.path.insert(0, '../')

import torch
import os.path as osp
import h5py
import numpy as np
import pandas as pd
from util import load_file
from dataloader.prepare_video import video_sampling, prepare_input

import argparse


def get_vlist(filename):
    data = pd.read_csv(filename)
    vids = list(set(list(data['video_id'])))
    return vids


def split_dataset_feat(filename, out_dir, train_list, val_list, test_list):

    train_fd = h5py.File(osp.join(out_dir, 'app_feat_train.h5'), 'w')
    val_fd = h5py.File(osp.join(out_dir, 'app_feat_val.h5'), 'w')
    test_fd = h5py.File(osp.join(out_dir, 'app_feat_test.h5'), 'w')
    val_feat_dset, val_ids_dset = None, None
    test_feat_dset, test_ids_dset = None, None
    train_feat_dset, train_ids_dset = None, None

    feat_name = 'swin_2d_224' #'resnext_features'
    string_dt = h5py.special_dtype(vlen=str)
    # vids = video_list
    with h5py.File(filename, 'r') as fp:
        # for k in fp.keys():
        #     print(fp[k].name)
        vids = fp['vid']
        feats = fp[feat_name]
        # print(vids.shape, feats.shape)
        for vid, feat in zip(vids, feats):
            
            if vid in val_list:
                if val_feat_dset is None:
                    dataset_size = len(val_list)
                    F, D = feat.shape
                    # C, D = feat.shape
                    val_feat_dset = val_fd.create_dataset(feat_name, (dataset_size, F, D),
                                                      dtype=np.float32)
                    val_ids_dset = val_fd.create_dataset('ids', shape=(dataset_size,), dtype='int') #dtype=string_dt)
                    ival = 0
                val_feat_dset[ival:(ival+1)] = feat
                val_ids_dset[ival:(ival+1)] = vid
                ival += 1
            if vid in test_list:
                if test_feat_dset is None:
                    dataset_size = len(test_list)
                    F, D = feat.shape
                    # C, D = feat.shape
                    test_feat_dset = test_fd.create_dataset(feat_name, (dataset_size, F, D),
                                                      dtype=np.float32)
                    test_ids_dset = test_fd.create_dataset('ids', shape=(dataset_size,), dtype='int') #dtype=string_dt)
                    itest = 0

                test_feat_dset[itest:(itest + 1)] = feat
                test_ids_dset[itest:(itest + 1)] = str(vid)
                itest += 1
            if vid in train_list:
                if train_feat_dset is None:
                    dataset_size = len(train_list)
                    F, D = feat.shape
                    # C, D = feat.shape
                    train_feat_dset = train_fd.create_dataset(feat_name, (dataset_size, F, D),
                                                      dtype=np.float32)
                    train_ids_dset = train_fd.create_dataset('ids', shape=(dataset_size,), dtype='int') #dtype=string_dt)
                    itrain = 0

                train_feat_dset[itrain:(itrain + 1)] = feat
                train_ids_dset[itrain:(itrain + 1)] = vid
                itrain += 1


def split_feat():

    data_dir = '../data/msvd/'
    filename = f'{data_dir}/swin_2d_224.h5'
    if not osp.exists(filename):
        print(f'{filename} is not existed')
        # extract_feat()
    else:
        out_dir = '../data/msvd/swin/'
        dset_dir = '../data/datasets/msvd/'
        train_file = f'{dset_dir}/train.csv'
        val_file = f'{dset_dir}/val.csv'
        test_file = f'{dset_dir}/test.csv'

        train_list = get_vlist(train_file)
        val_list = get_vlist(val_file)
        test_list = get_vlist(test_file)

        split_dataset_feat(filename, out_dir, train_list, val_list, test_list)


def time2id(durations, fps=30):
    if len(durations) == 1:
        duration = durations[0]
    else:
        max_id = 0
        max_du = 0
        for id, duration in enumerate(durations):
            du = duration[1] - duration[0]
            if du > max_du:
                max_du = du
                max_id = id
        duration = durations[max_id]
    start = int(np.round(float(duration[0]) * fps/5))
    end = int(np.round(float(duration[1]) * fps/5))
    return [start, end]


def extract_feat(feat_type, mode):

    dataset = 'nextqa'
    data_dir = f'/raid/jbxiao/data/{dataset}/'
    
    frame_dir = osp.join(data_dir, 'frames/')
   
    qset_file = f'../../data/datasets/nextqa/{mode}.csv'
    vlist = get_vlist(qset_file)
    vnum = len(vlist)
    # qset = load_file(qset_file)
    # qnum = len(qset)

    # gsub_file = f'../../data/datasets/nextgqa/gsub_{mode}.json'
    # gsub = load_file(gsub_file)    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if feat_type == 'CLIP':
        #CLIP 
        import clip as clip_model
        model, preprocess = clip_model.load('ViT-B/32', device=device)
    elif feat_type == 'CLIPL':
        #CLIP Large
        import clip as clip_model
        model, preprocess = clip_model.load('ViT-L/14', device=device)
    elif feat_type == 'BLIP':
        #BLIP
        from lavis.models import load_model_and_preprocess
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    elif feat_type == 'Swin':
        #Swin
        from model.video_model import Swin
        swin_encoder = Swin()
        swin_encoder.eval()
        swin_encoder.cuda()
    elif feat_type == 'BLIP2':
        #BLIP2
        from transformers import Blip2Processor, Blip2Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        model.to(device)

    app_feat_file = f'../../data/{dataset}/{feat_type}/{feat_type}_I_{mode}.h5'
    fp = h5py.File(app_feat_file, 'w')
    feat_dset = None
    vid_dset = None
    fnum = 32

    i1 = 0
    model.eval()
    with torch.no_grad():
        i = 0 
        for i, vid in enumerate(vlist):
        # for vid, anno in gsub.items():
        #     for qid, dus in anno['location'].items():
            i += 1
            vframe_dir = f'{frame_dir}/{vid}/'
            # duration = time2id(dus, fps=float(anno['fps']))
            frames, _ = video_sampling(vframe_dir, frame_num=fnum, mode='uniC')
            
            if feat_type in ['CLIP', 'CLIPL']:
                frames = [preprocess(img).numpy() for img in frames]
                images = torch.from_numpy(np.asarray(frames)).to(device)
                feat = model.encode_image(images) #fnum x 512
            if feat_type == 'Swin':
                video_inputs = prepare_input(frames).cuda()
                feat = swin_encoder(video_inputs) #fnumx1024
            elif feat_type == 'BLIP':
                frames = [vis_processors["eval"](img).numpy() for img in frames]
                images = torch.from_numpy(np.asarray(frames)).to(device)
                sample = {"image": images, "text_input": None}
                image_feats = model.extract_features(sample, mode="image")
                feat = image_feats.image_embeds[:, 0, :] #fnum x 768
            elif feat_type == 'BLIP2':
                inputs = processor(images=frames, return_tensors="pt").to(device, torch.float16)
                qformer_outs = model.get_qformer_features(**inputs)
                feat = qformer_outs.pooler_output
                

            feat = feat.detach().cpu().numpy()
            if feat_dset is None:
                feat_dset = fp.create_dataset(f'{feat_type}_I', (vnum, fnum, feat.shape[-1]), dtype=np.float32)
                dt = h5py.special_dtype(vlen=str)
                vid_dset = fp.create_dataset('vid',(vnum, ), dtype=dt)
            i2 = i1 + 1
            feat_dset[i1:i2] = feat
            vid_dset[i1:i2] = str(vid) #+"_"+str(qid)
            i1 = i2
            if i % 100 == 0:
                print(i, feat.shape)
            
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='CLIP')
    parser.add_argument('--mode', type=str, default='val')
    args = parser.parse_args()
    extract_feat(args.model_type, args.mode)