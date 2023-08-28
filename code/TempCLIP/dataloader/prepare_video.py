'''
Author: Xiao Junbin
Date: 2022-11-20 15:58:33
LastEditTime: 2022-11-28 15:58:43
LastEditors: Xiao Junbin
Description: sample video frames and load them into memory
FilePath: ./NExT-GQA/dataloader/prepare_video.py
'''
import os
import os.path as osp
import numpy as np
from PIL import Image
import h5py
import torch
import torchvision as tv


def sample_clips(total_frames, num_clips, num_frames_per_clip):
    clips = []
    frames = list(range(total_frames)) #[str(f+1).zfill(6) for f in range(total_frames)]
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1: num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        clip_start = 0 if clip_start < 0 else clip_start
        clip_end = total_frames if clip_end > total_frames else clip_end
        clip = frames[clip_start:clip_end] 
        if clip_start == 0 and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_start])
            if len(added_fids) > 0:
                clip = added_fids + clip
        if clip_end == total_frames and len(clip) < num_frames_per_clip:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_fids = []
            for _ in range(shortage):
                added_fids.append(frames[clip_end-1])
            if len(added_fids) > 0:
                clip += added_fids
        cid = clip[len(clip)//2] #[::4] use the center frame
        clips.append(cid)
    # clips = clips[::2]
    return clips



def video_sampling(vframe_dir, frame_num=32, mode='uniC', duration=None):
    
    frames = sorted(os.listdir(vframe_dir))
    #previously, we extract the videos by 6fps, here we further sparsely sample by 3fps for nextqa
    frame_ids = frames[::3] 
    if duration != None:
        frame_ids = []
        for frame in frames:
            fid = int(frame.split('.')[0])
            #sample inside ground-truth temporal segment.
            if fid >= duration[0] and fid <= duration[1]:  
                frame_ids.append(frame)

    frame_count = len(frame_ids)
    if mode == 'uniF':
        sampled_ids = np.linspace(0, frame_count-1, frame_num).astype('int')
    elif mode == 'uniC':
        sampled_ids = sample_clips(frame_count, frame_num, num_frames_per_clip=4)
    elif mode == 'rad':
        sampled_ids = np.random.choice(frame_count, frame_num, replace=False).astype('int')
        sampled_ids = sorted(sampled_ids)
    
    sampled_frames = np.asarray(frame_ids)[sampled_ids]
    # print(sampled_frames)
    images = []
    for vid in sampled_frames:
        frame_path = osp.join(vframe_dir, vid)
        img_input = Image.open(frame_path).convert("RGB")
        images.append(img_input)
        
    return images, sampled_frames


def prepare_input(raw_images):
    images = []
    target_size = 224
    for img in raw_images:
        w, h = img.size
        img_input = tv.transforms.Compose([tv.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                     tv.transforms.Resize([target_size, target_size]), 
                                     ])(img)
        images.append(np.array(img_input))
    images = np.asarray(images).transpose(0, 3, 1, 2)/255
    images = torch.from_numpy(images.astype(np.float32))
    video_inputs = tv.transforms.Normalize([0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])(images)

    
    return video_inputs                                 




