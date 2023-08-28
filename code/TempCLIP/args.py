import argparse
import os

from global_parameters import (
    DEFAULT_DATASET_DIR,
    DEFAULT_CKPT_DIR,
    TRANSFORMERS_PATH,
    dataset2folder,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="nextqa",
        choices=[
            "nextqa",
            "nextgqa",
        ],
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        choices=["", "1", "10", "20", "50"],
        help="use a subset of the generated dataset",
    )

    # Model
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["posthoc", "qa", "oeqa", "NG", "NG+"],
        help="qa baseline does not use the video, video baseline does not use the question",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of layers in the multi-modal transformer",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads in the multi-modal transformer",
    )
    parser.add_argument(
        "--embd_dim",
        type=int,
        default=512,
        help="multi-modal transformer and final embedding dimension",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="multi-modal transformer feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate in the multi-modal transformer",
    )
    parser.add_argument(
        "--sentence_dim",
        type=int,
        default=2048,
        help="sentence dimension for the differentiable bag-of-words embedding the answers",
    )
    parser.add_argument(
        "--qmax_words",
        type=int,
        default=20,
        help="maximum number of words in the question",
    )
    parser.add_argument(
        "--amax_words",
        type=int,
        default=10,
        help="maximum number of words in the answer",
    )
    parser.add_argument(
        "--max_feats",
        type=int,
        default=20,
        help="maximum number of video features considered",
    )

    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="folder where the datasets folders are stored",
    )
    parser.add_argument(
        "--checkpoint_predir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="folder to store checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="subfolder to store checkpoint"
    )
    parser.add_argument(
        "--pretrain_path", type=str, default="", help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=TRANSFORMERS_PATH,
        help="path to transformer models checkpoints",
    )

    # Train
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_val", type=int, default=2048)
    parser.add_argument(
        "--n_pair",
        type=int,
        default=32,
        help="number of clips per video to consider to train on HowToVQA69M",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_ep", action="store_true", help="whether to save checkpoit every epoch")
    parser.add_argument(
        "--test", type=str, default='test', help="[test, val]"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--clip",
        type=float,
        default=12,
        help="gradient clipping",
    )

    # Print
    parser.add_argument(
        "--freq_display", type=int, default=3, help="number of train prints per epoch"
    )
    parser.add_argument(
        "--num_thread_reader", type=int, default=16, help="number of workers"
    )

    # Masked Language Modeling and Cross-Modal Matching parameters
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--n_negs", type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--min_time", type=int, default=10)
    parser.add_argument("--min_words", type=int, default=10)

    # Demo parameters
    parser.add_argument(
        "--question_example", type=str, default="", help="demo question text"
    )
    parser.add_argument("--video_example", type=str, default="", help="demo video path")
    parser.add_argument("--port", type=int, default=8899, help="demo port")
    parser.add_argument(
        "--pretrain_path2", type=str, default="", help="second demo model"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./gmodels/", help="path to save dir"
    )
    parser.add_argument(
        "--mc", type=int, default=5, help="number of multiple choices"
    )
    parser.add_argument(
        "--feat_type", type=str, default='CLIP', choices=['Swin', 'CLIP', 'CLIPL', 'BLIP', 'BLIP2']
    )
    parser.add_argument(
        "--vg_loss", type=float, default=0, help="trade offf with video grounding loss"
    )
    parser.add_argument(
        "--lan", type=str, default='RoBERTa', choices=['DistilBERT', 'BERT', 'RoBERTa', 'DeBERTa']
    )
    parser.add_argument(
        "--prop_num", type=int, default=1, help="number of temporal propsoal num" 
    )
    parser.add_argument(
        "--sigma", type=float, default=9, help="control the wigth of Gaussian distribution" 
    )
    parser.add_argument(
        "--div_loss", type=float, default=1, help="diversity loss on Gaussian masks"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.15, help="control the overlap extent of different proposal, 0: no overlap, 1 no diversity"
    )
    parser.add_argument(
        "--vote", type=int, default=0, help="determine the best temporal proposal during inference"
    )
    parser.add_argument(
        "--gamma", type=float, default=1, help="Gaussian confidence interval"
    )

    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.bert_path

    # feature dimension
    args.feature_dim = args.ff_dim  # S3D:1024 app_mot:4096 #2048 RoI
    args.word_dim = 768  # DistilBERT

    # Map from dataset name to folder name

    load_path = os.path.join(args.dataset_dir, args.dataset)
    args.load_path = load_path

    args.features_path = f'../../../data/{args.dataset}/'
    args.train_csv_path = os.path.join(load_path, "train.csv")
    if args.dataset == 'tgifqa':
        args.val_csv_path = os.path.join(load_path, "test.csv")
    else:
        args.val_csv_path = os.path.join(load_path, "val.csv")
    args.test_csv_path = os.path.join(load_path, "test.csv")
    args.vocab_path = os.path.join(load_path, "vocab.json")
    

    return args
