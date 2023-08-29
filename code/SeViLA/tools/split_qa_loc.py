import json
import os.path as osp
import pandas as pd


def split_file(res_file, anno_file):
    with open(res_file, 'r') as fp:
        res = json.load(fp)
    # anno = pd.read_csv(anno_file)
    test_qa = {}
    test_gd = {}
    for key, value in res.items():
        test_qa[key] = {"prediction":value['prediction'], "answer":value['answer']}
        test_gd[key] = [value['location'][0], value['location'][-1]]
    dirname = osp.dirname(res_file)
    qa_file = f'{dirname}/test-res.json'
    gd_file = f'{dirname}/test_ground.json'
    with open(qa_file, 'w') as fp:
        json.dump(test_qa, fp)
    with open(gd_file, 'w') as fp:
        json.dump(test_gd, fp)

def main():
    res_file = '../../data/sevila/results/nextqa_infer/result/test_ground.json'
    anno_file = '../../data/datasets/nextaqa/test.csv'
    split_file(res_file, anno_file)

if __name__ == "__main__":
    main()
