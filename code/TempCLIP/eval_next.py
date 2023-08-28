import os.path as osp
from util import load_file, save_to
import argparse

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 
            'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 
            'T': 'Acc_T', 'D': 'Acc_D'}

def accuracy_metric(sample_list, result):
    
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video_id']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': 
            qtype = 'TN'
        group[qtype].append(qns_id)

    preds = result
    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        # print(qtype, len(qns_ids))
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt


    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    for qtype in group_acc:
        if group_cnt[qtype] == 0: continue
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0: continue
        print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))


def accuracy_metric_sub(sample_list, result, sub_ids):
    
    sub_ids = [int(id) for id in sub_ids]
    subset = sample_list.iloc[sub_ids]

    accuracy_metric(subset, result)


def eval_ground(sample_list, gsubset, result):

    gids = []
    vids = []
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        if vid in gsubset and qid in gsubset[vid]['location'].keys():
            vids.append(vid)
            gids.append(idx)
    print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
    subset = sample_list.iloc[gids]
    accuracy_metric(subset, result)


def eval_ground_nov(sample_list, gsubset, result, nov):
    gids = []
    vids = []
    ks = []
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        k = f'{vid}_{qid}'
        if nov[k]['answer'] == nov[k]['prediction']: continue
        if vid in gsubset and qid in gsubset[vid]['location'].keys():
            vids.append(vid)
            gids.append(idx)
            ks.append(k)
    print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
    subset = sample_list.iloc[gids]
    accuracy_metric(subset, result)
    # save_to('./aba/vqa_sub.json', ks)

def eval_ground_nov_nosig(sample_list, gsubset, result, nov, sigF):

    gids = []
    vids = []
    ks = []
    # print(len(sample_list), len(sigF))
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        k = f'{vid}_{qid}'
        if nov[k]['answer'] == nov[k]['prediction'] or sigF[k]['answer'] == sigF[k]['prediction']: continue
        if vid in gsubset and qid in gsubset[vid]['location'].keys():
            vids.append(vid)
            gids.append(idx)
            ks.append(k)
    print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
    subset = sample_list.iloc[gids]
    accuracy_metric(subset, result)
    # save_to('./aba/vidqa_sub.json', ks)


def eval_ground_vqa(sample_list, gsubset, result, nov, noq, novq):
    gids = []
    vids = []
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        k = f'{vid}_{qid}'
        if nov[k]['answer'] == nov[k]['prediction'] or noq[k]['answer'] == noq[k]['prediction'] or novq[k]['answer'] == novq[k]['prediction']: 
            continue
        if vid in gsubset and qid in gsubset[vid]['location'].keys():
            vids.append(vid)
            gids.append(idx)
    print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
    subset = sample_list.iloc[gids]
    accuracy_metric(subset, result)


def eval_ground_gqa(sample_list, gsubset, result, nov, neg, pos):
    gids = []
    vids = []
    ks = []
    for idx, row in sample_list.iterrows():
        vid, qid = str(row['video_id']), str(row['qid'])
        k = f'{vid}_{qid}'
        if nov[k]['answer'] == nov[k]['prediction'] or neg[k]['answer'] == neg[k]['prediction'] or pos[k]['answer'] != pos[k]['prediction']: 
            continue
        if vid in gsubset and qid in gsubset[vid]['location'].keys():
            vids.append(vid)
            gids.append(idx)
        ks.append(k)
    print("Video: {}  Questions: {}".format(len(set(vids)), len(gids)))
    subset = sample_list.iloc[gids]
    accuracy_metric(subset, result)
    # save_to('./aba/gdqa_sub.json', ks)


def main(result_file, mode='val'):
    dataset_dir = '../data/datasets/nextgqa/'
    data_set = mode
    sample_list_file = osp.join(dataset_dir, data_set+'.csv')
    print('Evaluating {}'.format(result_file))

    sample_list = load_file(sample_list_file)
    result = load_file(result_file)
    print(len(result))

    # accuracy_metric(sample_list, result)
    # if mode == 'val':
    #     print('==============ATP-hard Subset================')
    #     hard_subset = osp.join(dataset_dir, 'atp-hard-ct4.txt')
    #     sub_ids = load_file(hard_subset)
    #     accuracy_metric_sub(sample_list, result, sub_ids)
    
    print('===============Ground Subset=================')
    gsub_file = osp.join(dataset_dir, f'gsub_{mode}.json')
    gsubset = load_file(gsub_file) 
    eval_ground(sample_list, gsubset, result)
    
    # res_dir = '../data/save_models/nextqa/'
    res_dir = '../data/gmodels/nextgqa/'
    
    # print('==============Ground VQA Subset==============')
    # nov_file = f'{res_dir}/BlindQA/{mode}-res.json'
    # nov = load_file(nov_file)
    # eval_ground_nov(sample_list, gsubset, result, nov)

    # print('==============Ground GQA Subset==============')
    # res_dir = '../data/gmodels/NG+/'
    # neg_file = f'{res_dir}/TempCLIP-NEG/{mode}-res.json'
    # neg = load_file(neg_file)
    # pos_file = f'{res_dir}/TempCLIP-POS/{mode}-res.json'
    # pos = load_file(pos_file)
    # eval_ground_gqa(sample_list, gsubset, result, nov, neg, pos)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='val', choices=['val','test'])
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()
    res_dir = '../data/gmodels/'+args.folder
    #res_dir = '../data/models/nextqa/'
    mode = args.mode
    model_prefix = 'res'
    result_file = '{}/{}-{}.json'.format(res_dir, mode, model_prefix)
    main(result_file, mode)
