import sys
sys.path.insert(0, '../')
from lavis.datasets.data_utils import load_video_demo
import json
import os.path as osp

def fid2time(vid_dir, res_file, map_file):
    with open(res_file, 'r') as fp:
        data = json.load(fp)
    with open(map_file, 'r') as fp:
        mapID = json.load(fp)
    image_size = 224
    gqa = {}
    for cnt, item in enumerate(data):
        qid = item['qid']
        pred = item['prediction']
        target = item['target']
        frame_index = item['frame_idx']
        vid = qid.split('_')[1]
        qs_id = qid.split('_')[-1]
        vpath = f'{vid_dir}/{mapID[vid]}.mp4'
        raw_clip, indice, fps, vlen = load_video_demo(
            video_path=vpath,
            n_frms=32,
            height=image_size,
            width=image_size,
            sampling="uniform",
            clip_proposal=None
        )
        tspan = []
        video_len = vlen/fps # seconds
        for i in frame_index:
            select_i = indice[i]
            time = round((select_i / vlen) * video_len, 2)
            tspan.append(time)
        key_id = vid+'_'+qs_id
        gqa[key_id] = {'prediction':pred, 'answer':target, 'location':tspan}
        if cnt % 500 == 0:
            print(gqa[key_id])
    
    with open(osp.dirname(res_file)+'/test_ground.json', 'w') as fp:
        json.dump(gqa, fp)
    

def main():
    res_dir = '../../data/sevila/results/nextqa_infer/result/'
    vid_dir = '/storage/jbxiao/workspace/data/nextqa/videos/'
    map_file = '../../data/datasets/nextqa/map_vid_vidorID.json'
    res_file = f'{res_dir}/test_epochbest.json'
    fid2time(vid_dir, res_file, map_file)

if __name__ == "__main__":
    main()
