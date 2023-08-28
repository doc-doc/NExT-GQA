import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import logging

from transformers import get_cosine_schedule_with_warmup

from args import get_args

from model.TempVLM import VQA
from loss import LogSoftmax
from util import compute_a2v, save_to
from dataloader.vqa_loader import get_videoqa_loaders
from train.trainval_gdqa import train, eval
# uncomment in stage-1 pretrain
from train.trainval_warm import train, eval


def main(args):
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)

    
    if args.lan == 'BERT':
        from transformers import BertTokenizer #DistilBertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lan == 'RoBERTa':
        from transformers import RobertaTokenizerFast,RobertaTokenizer
        bert_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.lan == 'DeBERTa':
        from transformers import DebertaTokenizer
        bert_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    
    # special_tokens_dict = {'additional_special_tokens': ['[TSW]']}
    # bert_tokenizer.add_special_tokens(special_tokens_dict)
    
    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=bert_tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    # Model
    model = VQA(
        bert_tokenizer = bert_tokenizer,
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        vocab_size = bert_tokenizer.vocab_size,
        baseline=args.baseline,
        feat_type=args.feat_type,
        lan=args.lan,
        prop_num=args.prop_num,
        sigma=args.sigma
    )
    model.cuda()
    logging.info("Using {} GPUs".format(torch.cuda.device_count()))

    # Load pretrain path
    model = nn.DataParallel(model)
    
    if args.pretrain_path != "":
        model.load_state_dict(torch.load(args.pretrain_path))
        # model.load_state_dict(load_model_by_key(model, args.pretrain_path))
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
    logging.info(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_videoqa_loaders(args, args.features_path, a2id, bert_tokenizer, test_mode = args.test)

    if args.test != 'train':
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))
    else:
        logging.info("number of train instances: {}".format(len(train_loader.dataset)))
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

   
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(
        params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion.cuda()

    # Training
    if args.test == 'train':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, len(train_loader) * args.epochs
        )
        logging.info(
            f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
        )
        if args.pretrain_path != "":
            val_acc, results, _ = eval(model, val_loader, a2v, args, test=False)  # zero-shot VideoQA
            save_path = osp.join(args.save_dir, 'val-res0.json')
            save_to (save_path, results)
        best_val_acc = 0 if args.pretrain_path == "" else val_acc
        best_epoch = 0
        for epoch in range(args.epochs):
            train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, bert_tokenizer)
            val_acc, results, _ = eval(model, val_loader, a2v, args, test=False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
                save_path = osp.join(args.save_dir, 'val-res.json')
                save_to (save_path, results)
            if args.save_ep: 
                ep_file = os.path.join(args.save_dir, f"e{epoch}.pth")
                torch.save(model.state_dict(), ep_file)
                logging.info('Save to '+ep_file)
        logging.info(f"Best val model at epoch {best_epoch + 1}")
    else:   
    # Evaluate
        test_acc, results, ground = eval(model, test_loader, a2v, args, test=True)
        save_path = osp.join(args.save_dir, f'{args.test}-res.json')
        save_to(save_path, results)
        
        ground_att, ground_gs = ground[0], ground[1]
        if args.baseline in ['NG', 'NG+']:
            gd_path = osp.join(args.save_dir, f'{args.test}_ground_gs.json')
            save_to(gd_path, ground_gs)
            att_path = osp.join(args.save_dir, f'{args.test}_ground_att.json')
            save_to(att_path, ground_att)
            from tools import getspan
            getspan.main(args.save_dir, args.test)
            import eval_ground
            eval_ground.main(args.save_dir, osp.basename(gd_path), True)
        if args.baseline == 'posthoc':
            att_path = osp.join(args.save_dir, f'{args.test}_ground_att.json')
            save_to(att_path, ground_att)
            from tools import getspan
            getspan.main(args.save_dir, args.test)
            import eval_ground
            eval_ground.main(args.save_dir, f'{args.test}_ground_ada.json')
        

if __name__ == "__main__":
    # set random seeds
    args = get_args()
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
        
    main(args)
