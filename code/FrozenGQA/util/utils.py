import os.path as osp
import json
import pandas as pd



def get_qsn_type(qsn, ans_rsn):
    dos = ['does', 'do', 'did']
    bes = ['was', 'were', 'is', 'are']
    w5h1 = ['what', 'who', 'which', 'why', 'how', 'where']
    qsn_sp = qsn.split()
    type = qsn_sp[0].lower()
    if type == 'what':
        if qsn_sp[1].lower() in dos:
            type = 'whata'
        elif qsn_sp[1].lower() in bes:
            type = 'whatb'
        else:
            type = 'whato'
    elif type == 'how':
        if qsn_sp[1].lower() == 'many':
            type = 'howm'
    elif type not in w5h1:
        type = 'other'
    if ans_rsn in ['pr', 'cr']:
        #for causalVid, we distiguish answer and reason 
        type += 'r'
    return type


def group(csv_data, gt=True):
    ans_group, qsn_group = {}, {}
    for idx, row in csv_data.iterrows():
        qsn, ans = row['question'], row['answer']
        if gt:
            type = row['type']
            if type == 'TP': type = 'TN'
        else:
            type = 'null' if 'type' not in row else row['type']
            type = get_qsn_type(qsn, type)
        if type not in ans_group:
            ans_group[type] = {ans}
            qsn_group[type] = {qsn}
        else:
            ans_group[type].add(ans)
            qsn_group[type].add(qsn)
    return ans_group, qsn_group


def load_model_by_key(cur_model, model_path):
    model_dict = torch.load(model_path)
    new_model_dict = {}
    for k, v in cur_model.state_dict().items():
        if k in model_dict:
            v = model_dict[k]
        else:
            pass
            # print(k)
        new_model_dict[k] = v
    return new_model_dict


def load_file(filename):
    file_type = osp.splitext(filename)[-1]
    if file_type == '.csv':
        data = pd.read_csv(filename)
    else:
        with open(filename, 'r') as fp:
            if file_type == '.json':
                data = json.load(fp)
            elif file_type == '.txt':
                data = fp.readlines()
                data = [datum.rstrip('\n') for datum in data]
    return data
