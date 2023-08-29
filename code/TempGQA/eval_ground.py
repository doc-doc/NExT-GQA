from util import *

def get_tIoU(loc, span):
    
    if span[0] == span[-1]:
        if loc[0] <= span[0] and span[0] <= loc[1]:
            return 0, 1
        else:
            return 0, 0
    
    span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
    span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
    dis_i = (span_i[1] - span_i[0])
    if span_u[1] > span_u[0]:
        IoU = dis_i / (span_u[1] - span_u[0]) 
    else: 
        IoU = 0.0
    if span[-1] > span[0]:
        IoP = dis_i / (span[-1] - span[0]) 
    else:
        IoP = 0.0

    return IoU, IoP


def eval_ground(gt_ground, pred_ground, pred_qa=None, subset=None, gs=False):
    
    mIoU, mIoP = 0, 0
    cnt, cqt = 0, 0
    crt3, crt5 = 0, 0
    crtp3, crtp5 = 0, 0
    for vid, anno in gt_ground.items():
        for qid, locs in anno['location'].items():
            if not (f'{vid}_{qid}' in pred_ground):
                # print(vid, qid)
                continue
            if subset != None:
                # Non-Blind and Non-Sig QA subset
                if not (f'{vid}_{qid}' in subset):
                    continue
            max_tIoU, max_tIoP = 0, 0
            for loc in locs:
                span = pred_ground[f'{vid}_{qid}']
                # we need to multiply video duration if Gaussian
                if gs: span = np.round(np.asarray(span)*anno['duration'], 1)
                tIoU, tIoP = get_tIoU(loc, span)
                if tIoU > max_tIoU:
                    max_tIoU = tIoU
                if tIoP > max_tIoP:
                    max_tIoP = tIoP
            if max_tIoP >= 0.3:
                crtp3 += 1
                if  max_tIoP >= 0.5:
                    crtp5 += 1
                    kid = f'{vid}_{qid}'
                    
                    if pred_qa:
                        if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                            cqt+= 1
                            # print(kid)

            if max_tIoU >= 0.3:
                crt3 += 1
                if max_tIoU >= 0.5:
                    crt5 += 1
                    # if pred_qa:
                    #     if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                    #         print(kid)

            cnt += 1
            mIoU += max_tIoU
            mIoP += max_tIoP
    
    mIoU = mIoU /cnt * 100
    mIoP = mIoP/cnt * 100
    print('Acc&GQA mIoP TIoP@0.3 TIoP@0.5 mIoU TIoU@0.3 TIoU@0.5 ')
    print('{:.1f} \t {:.1f}\t {:.1f}\t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}'.format(cqt*1.0/cnt*100, mIoP,
          crtp3*1.0/cnt*100, crtp5*1.0/cnt*100, 
          mIoU, crt3*1.0/cnt*100, crt5*1.0/cnt*100))
    
def combine(pred1, pred2, gt):
    """
    pred1: ground segment by gaussian mask
    pred2: ground segment by post-hoc attention
    gt: to get NExT-GQA subset
    """
    def _cb_seg(seg1, seg2, way='uni'):
        # print(seg1, seg2)
        if way == 'uni':
            ts = [seg1[0], seg1[1], seg2[0], seg2[1]]
            ts = sorted(ts)
            new_seg = [ts[0], ts[-1]]
        elif way == 'itsc':
            start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
            end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
            if not (start <= end):
                new_seg = seg2.tolist() #trust more on attention
            else:
                new_seg = [start, end]
        return new_seg
    
    cb_ground = {}
    for vqid, seg in pred1.items():
        vid, qid = vqid.split('_')
        if not (vid in gt and qid in gt[vid]['location']):
            continue 
        duration = gt[vid]['duration']
        seg = np.round(np.asarray(seg)*duration, 1)
        seg_att = np.asarray(pred2[vqid])
        new_seg  = _cb_seg(seg, seg_att, way='itsc')
        cb_ground[vqid] = new_seg
    
    # save_to()
    return cb_ground


def main(res_dir, filename, gs=False):

    data_dir = '../../datasets/nextgqa/'
    
    dset = filename.split('_')[0]
    gt_file = osp.join(data_dir, f'gsub_{dset}.json')
    pred_file = osp.join(res_dir, filename)
    qa_file = osp.join(res_dir, f'{dset}-res.json')
    # sub_file = osp.join('./aba/TempVQA/gdqa/gdqa_sub.json')
    # sub = load_file(sub_file)
    gt_ground = load_file(gt_file)
    pred_ground = load_file(pred_file)
    pred_qa = load_file(qa_file)
    
    eval_ground(gt_ground, pred_ground, pred_qa, subset=None, gs=gs)

    if 1:
        print('=============post-hoc ground==============')
        pred_file = osp.join(res_dir, f'{dset}_ground_ada.json')
        pred_ground_att = load_file(pred_file)
        eval_ground(gt_ground, pred_ground_att, pred_qa, gs=False)
        
        print('=======merge post-hoc and gauss mask======')
        cb_ground = combine(pred_ground, pred_ground_att, gt_ground)
        cb_ground_file = osp.join(res_dir, f'{dset}_ground_cb.json')
        eval_ground(gt_ground, cb_ground, pred_qa, gs=False)
        save_to(cb_ground_file, cb_ground)


if __name__ == "__main__":
    res_dir = '../../../data/gmodels/NG+/FrozenGQA/'
    # res_dir = '../../../data/gmodels/NG+/TempCLIP/'
    filename = 'test_ground_gs.json'
    main(res_dir, filename, gs=True)
