import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens, calculate_IoU_batch
import os.path as osp
import json
import numpy as np
import pickle as pkl
# from fvcore.nn import FlopCountAnalysis

def eval(model, data_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    results = {}
    ground_res = {}
    ground_gs = {}
    ground_mask = {}
    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        
        for i, batch in enumerate(data_loader):
            answer_id, answer, video_frames, question, question_id = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_frames"].cuda(),
                batch["question"].cuda(),
                batch['question_id'],    
            )
           
            video_len = batch["video_len"]
           
            if args.lan in ['DistilBERET','BERT', 'DeBERTa']:
                pad_id = 0
            elif args.lan == 'RoBERTa':
                pad_id = 1
            question_mask = (question!=pad_id).float() 
            answer_mask = (answer!=pad_id).float() 
            video_mask = get_mask(video_len, video_frames.size(1)).cuda()
            count += answer_id.size(0)
            bsize = answer_id.shape[0]
            
            #############Model FLOPs##########
            # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
            # flops = FlopCountAnalysis(model, inputs)
            # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
            # break
            ###################################
            fusion_proj, answer_proj, kargs = model(
                video_frames,
                video_mask,
                question,
                question_mask,
                answer,
                stage='GQA'
            )

            # predicts = fusion_proj.squeeze()
            # fatt = fatt.squeeze().cpu().numpy()
            
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

            # predicts = predicts.view(bsize, 8, -1).max(dim=1)[0] #slightly wrose than mean
            if args.baseline in ['NG+', 'NG']:
                prediction = predicts.view(bsize, args.prop_num, args.mc)
                
                # prop_scores = torch.from_numpy(np.zeros((bsize, args.prop_num))).cuda()
                # for bs, aid in enumerate(answer_id):
                #     prop_scores[bs] = prediction[bs, :, aid] #.cpu().numpy()

                index = torch.argmax(predicts, dim=-1)
                prop_scores = torch.from_numpy(np.zeros(index.shape)).cuda()
                for bp_id, max_op in enumerate(index):
                    prop_scores[bp_id] = predicts[bp_id,max_op] # predicts[torch.argmax(predicts, dim=-1)].view(bsize, -1)
                
                predicts = prediction.mean(dim=1)
                prop_scores = prop_scores.view(bsize, args.prop_num)
                
                idx = (-prop_scores).argsort(dim=-1)

                # print(idx.shape)
                att = kargs['fatt'].view(bsize, args.prop_num, -1)
                att = att.gather(1, idx.unsqueeze(-1).expand(-1, -1, att.size(-1)))
                gc = kargs['gcenter'].view(bsize, args.prop_num).gather(index=idx, dim=-1)
                gw = kargs['gwidth'].view(bsize, args.prop_num).gather(index=idx, dim=-1)
                gmask = kargs['gweight'].view(bsize, args.prop_num, args.max_feats).cpu().numpy() #.gather(index=idx, dim=1)
                gamma = args.gamma
                props = torch.stack([torch.clamp(gc-gamma*gw/args.sigma, min=0), torch.clamp(gc+gamma*gw/args.sigma, max=1)], dim=-1)
                props = props.cpu().numpy()
                
                if args.vote:
                    if args.vote == 1:
                        #vote for the proposal with maximal overlap with others
                        c = np.ones((bsize, args.prop_num))
                        votes = np.zeros((bsize, args.prop_num))
                        for i in range(args.prop_num):
                            for j in range(args.prop_num):
                                iou = calculate_IoU_batch((props[:, i, 0], props[:, i, 1]), (props[:, j, 0], props[:, j, 1]))
                                iou = iou * c[:, j]
                                votes[:, i] = votes[:, i] + iou
                        idx = np.argmax(votes, axis=1)
                        prop = props[np.arange(bsize), idx]
                        att = att[torch.arange(bsize), idx, :]
                    elif args.vote == 2:
                        assert args.vote == 2, 'not implemented yet'
                        #vote for the intersection of multiple proposals
                else:
                    #directly choose the temporal proposal with highest confidence
                    prop = props[:, 0].squeeze()
                    att = att[:,0,:]

            predicted = torch.max(predicts, dim=1).indices.cpu()
            metrics["acc"] += (predicted == answer_id).sum().item()
            
            if args.baseline == 'posthoc':
                att = kargs['fatt']

            att = att.squeeze().cpu().numpy()
            
            for bs, qid in enumerate(question_id):
                results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
                if args.baseline == 'posthoc':
                    ground_res[qid] = att[bs].tolist()
                else:
                    ground_gs[qid] = prop[bs].tolist()
                    ground_mask[qid] = gmask[bs]
                    ground_res[qid] = att[bs].tolist()
                

    step = "val" if not test else "test"
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break
    
    if args.baseline == 'NG+':
        with open(osp.join(args.save_dir,'gauss_mask.pkl'), 'wb') as fp:
            pkl.dump(ground_mask, fp)
    
    return metrics["acc"] / count, results, [ground_res, ground_gs]


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_vg_loss, running_div_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter()
    )
    for i, batch in enumerate(train_loader):
        answer_id, answer, video_frames, question = (
            batch["answer_id"],
            batch["answer"],
            batch["video_frames"].cuda(),
            batch["question"].cuda(),
        )
       
        qsns_id, qsns_token_ids, qsns_seq_len = (
            batch['qsns_id'],
            batch['qsns_token_ids'],
            batch['qsns_seq_len']
        )
        
        video_len = batch["video_len"]
        
        if args.lan in ['DistilBERET','BERT', 'DeBERTa']:
            pad_id = 0
        elif args.lan == 'RoBERTa':
            pad_id = 1
        question_mask = (question!=pad_id).float().cuda() 
        answer_mask = (answer!=pad_id).float().cuda()

        video_mask = (
            get_mask(video_len, args.max_feats).cuda() if args.max_feats > 0 else None
        )
        
        N = answer_id.size(0)
        
        if args.baseline in['NG+', 'posthoc', 'NG']:
            #find the video moments that are relevant to the question. 
            # qsns_mask = (qsns_token_ids != pad_id).float().cuda()
            if args.vg_loss:
                vt_proj, txt_proj, args_vg = model(
                    video_frames,
                    video_mask,
                    question,
                    question_mask,
                    answer=qsns_token_ids,
                    answer_id = qsns_id,
                    stage='GD'
                )
                vt_proj = vt_proj.unsqueeze(2)
                vq_predicts = torch.bmm(txt_proj, vt_proj).squeeze()
                vq_predicts = vq_predicts.view(N, args.prop_num, -1).mean(dim=1)
                vg_loss = criterion(vq_predicts, qsns_id.cuda())
            
            
            predicted = torch.max(vq_predicts, dim=1).indices.cpu() 
            running_acc.update((predicted == qsns_id).sum().item() / N, N)

            
            if args.prop_num > 1 and args.div_loss:
                    gauss_weight = args_vg['gweight'].view(N, args.prop_num, -1)
                    # gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
                    target = torch.eye(args.prop_num).unsqueeze(0).cuda() * args.lamb
                    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
                    div_loss_tmp = torch.norm(target - source, dim=(1, 2))**2
                    div_loss = div_loss_tmp.mean()
            
        loss = vg_loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()
        
        # running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.vg_loss: running_vg_loss.update(vg_loss.detach().cpu().item(), N)
        if args.div_loss:
            running_div_loss.update(div_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            
            if not args.vg_loss and args.div_loss:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Lvqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}, Div Loss: {running_div_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, vg loss: "
                    f"{running_vg_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                )
           
            if args.vg_loss: running_vg_loss.reset()
            if args.div_loss: running_div_loss.reset()
