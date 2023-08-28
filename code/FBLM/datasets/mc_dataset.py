import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math
import os.path as osp
import h5py
import numpy as np
import random as rd
import sys
sys.path.insert(0, '../')
from util.utils import get_qsn_type, group, load_file
rd.seed(1)

class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path,
        max_feats=10,
        features_dim=768,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
        feat_type="CLIPL",
        vg_loss = 0
    ):
        self.data = pd.read_csv(csv_path)
        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        # self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        mc = 0
        while f"a{mc}" in self.data:
            mc += 1
        self.mc = mc
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix
        self.vg_loss = vg_loss
        self.frame_feats = {}
        self.v_questions = {}
        self.mode = osp.basename(csv_path).split('.')[0]
        self.agu = False
        if self.mode == 'train':
            if self.agu:
                self.qsn_agu = load_file(osp.join(osp.dirname(csv_path), 'train_gpt4_sub.json'))
            self.all_answers = set(self.data['answer'])
            self.all_question = set(self.data['question'])
            self.ans_group, self.qsn_group = group(self.data, gt=False)
            if self.vg_loss: self._gather_by_v()

        app_feat_file = osp.join(features_path, f'{feat_type}/{feat_type}_I_{self.mode}.h5')
        print('Load {}...'.format(app_feat_file))
        self.frame_feats = {}
        with h5py.File(app_feat_file, 'r') as fp:
            vids = fp['vid']
            feats = fp[f'{feat_type}_I']
            print(feats.shape) #v_num, clip_num, feat_dim
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                vid = vid.decode("utf-8")
                self.frame_feats[str(vid)] = feat

        # with h5py.File(app_feat_file, 'r') as fp:
        #     vqids = fp['qid']
        #     feat_key = f'{feat_type}_I'
        #     feats = fp[feat_key]
        #     print(feats.shape) #v_num, clip_num, feat_dim
        #     for id, (vqid, feat) in enumerate(zip(vqids, feats)):
        #         vqid = vqid.decode("utf-8")
        #         self.frame_feats[str(vqid)] = feat


    def __len__(self):
        return len(self.data)

    
    def _gather_by_v(self):
        for idx, row in self.data.iterrows():
            vid, qsn, qtype = str(row['video_id']), row['question'], row['type']
            if qtype[0] == 'D': continue #exclude descriptive question
            if vid not in self.v_questions:
                self.v_questions[vid] = [qsn]
            else:
                self.v_questions[vid].append(qsn)


    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text_q(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: What question can the video clip answer? Is it '{question}' {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = th.zeros(1, self.features_dim)
        else:
            if start is not None and not math.isnan(start):
                video = self.features[video_id][int(start) : int(end) + 1].float()
            else:
                video = self.features[video_id].float()
            if not len(video):
                print(video_id, start, end)
                video = th.zeros(1, self.features_dim)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, video_len

    def _get_video_feats(self, vid_id):
        feat =  self.frame_feats[vid_id]
        fnum = feat.shape[0]
        # sp_fids = [fnum // 2]
        sp_fids = np.linspace(0, fnum-1, self.max_feats, dtype=int)
        feat = feat[sp_fids]
        
        return th.from_numpy(feat), self.max_feats
    
    def _get_vqid_feats(self, vqid):

        feat = self.frame_feats[vqid]
        vlen = feat.shape[0]
        
        return th.from_numpy(feat), vlen


    def __getitem__(self, idx):

        cur_data = self.data.loc[idx]
        video_id = str(cur_data['video_id'])
        qid = video_id+"_"+str(cur_data['qid'])

        # get start, end
        start = 0 #self.data["start"].values[idx]
        end = 0 #self.data["end"].values[idx]

        # get question
        question = str(cur_data['question'])
        if self.mode == 'train' and self.agu:
            if qid in self.qsn_agu:
                cur_agus = self.qsn_agu[qid]['gen']
                #cur_agus.append(question)
                question = rd.sample(cur_agus, 1)[0] if rd.random() < 0.3 else question
        question = question.capitalize().strip()
        #print(question)
        if question[-1] != "?":
            question = str(question) + "?"
        qtype = cur_data['type']
        # if "type" in self.data:
        #     type = self.data["type"].values[idx]
        # get subs
        if self.subs:
            subs = self._get_subtitles(video_id, start, end)
        else:
            subs = ""

        # get features
        video, video_len = self._get_video_feats(video_id)
        
        # get answer id
        answer_id = -1  # for hidden set testing

        # text = []
        # for i in range(self.mc):
        #     ai = cur_data['a'+f'{i}'].capitalize().strip()
        #     text.append(self._get_text(subs, ai, self.mask, question))
        #     if cur_data['a'+f'{i}'] == cur_data['answer']:
        #         answer_id = i

        ans = cur_data['answer']
        choices = [str(cur_data["a" + str(i)]) for i in range(self.mc)]
        answer_id = choices.index(ans) if ans in choices else -1

        qtype = get_qsn_type(cur_data['question'], qtype)
        if self.mode=='train' and rd.random() < 0.3:
            if qtype not in self.ans_group or len(self.ans_group[qtype]) < self.mc-1:
                valid_anscans = self.all_answers
            else:
                valid_anscans = self.ans_group[qtype]
            
            cand_answers = valid_anscans - set(ans)
            choices = rd.sample(list(cand_answers), self.mc-1)
            choices.append(ans)

            rd.shuffle(choices)
            answer_id = choices.index(ans)
        
        q_text, qsn_id = [], -1
        if self.mode=='train' and self.vg_loss:
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < self.mc-1:
                valid_qsncans = self.all_questions
            else:
                valid_qsncans = self.qsn_group[qtype]
             
            if rd.random() < 0.3:
                same_v_qsn = set(self.v_questions[video_id])
                same_v_other_qsn = list(same_v_qsn - set(question))
                num_other = len(same_v_other_qsn)
                if num_other >= self.mc-1:
                    qchoices = rd.sample(same_v_other_qsn, self.mc-1)
                else:
                    cand_qsn = valid_qsncans - same_v_qsn
                    if len(cand_qsn) < self.mc-1-num_other:
                        cand_qsn = set(self.all_question) - same_v_qsn
                    qchoices = same_v_other_qsn + rd.sample(list(cand_qsn), self.mc-1-num_other)
            else:
                cand_qsns = valid_qsncans - set(question)
                qchoices = rd.sample(list(cand_qsns), self.mc-1)
            """
            same_v_qsn = set(self.v_questions[video_id])
            same_v_other_qsn = list(same_v_qsn - set(question))
            num_other = len(same_v_other_qsn)
            cand_qsn = valid_qsncans - same_v_qsn
            
            if num_other >= 2:
                qchoices = rd.sample(same_v_other_qsn, 2)
            else:
                add = rd.sample(list(cand_qsn), 2-num_other)
                qchoices = same_v_other_qsn + add
                cand_qsn = cand_qsn - set(add)
            qchoices.extend(rd.sample(list(cand_qsn), 2))
            """
            qchoices.append(question)
            rd.shuffle(qchoices)
            qsn_id = qchoices.index(question)
            
            # q_text = [qsn.capitalize().strip()+f'? {self.mask}' for qsn in qchoices]
            q_text = [self._get_text_q(subs, '', self.mask, qsn) for qsn in qchoices]
            

        qa_text = [self._get_text(subs, ai.capitalize().strip(), self.mask, question) for ai in choices]
        
        return {
            "video": video,
            "video_len": video_len,
            "text": qa_text,
            "qid": qid,
            "answer_id": answer_id,
            "type": cur_data['type'],
            "qtext": q_text,
            "qsn_id": qsn_id
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    q_text = [
        [batch[i]["qtext"][j] for i in range(bs)] for j in range(len(batch[0]["qtext"]))
    ]
    qsn_id = default_collate([batch[i]["qsn_id"] for i in range(bs)])

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
        "qtext": q_text,
        "qsn_id": qsn_id
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "nextqa":
        if split == "train":
            csv_path = args.nextqa_train_csv_path
        elif split == "val":
            csv_path = args.nextqa_val_csv_path
        elif split == "test":
            csv_path = args.nextqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = ""
        features_path = args.nextqa_features_path
    elif dataset_name == "nextgqa":
        if split == "train":
            csv_path = args.nextgqa_train_csv_path
        elif split == "val":
            csv_path = args.nextgqa_val_csv_path
        elif split == "test":
            csv_path = args.nextgqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = "" 
        features_path = args.nextgqa_features_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
        feat_type=args.feat_type,
        vg_loss = args.vg_loss
    )
