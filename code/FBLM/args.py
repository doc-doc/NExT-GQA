import argparse
import os

PRESAVE_DIR = ""
MODEL_DIR = ""
DATA_DIR = "../../datasets/"
SSD_DIR = ""
name2folder = {
    "nextqa": 'nextqa',
    "nextgqa": 'nextgqa'
}


def get_args_parser():
    parser = argparse.ArgumentParser("Set FrozenBiLM", add_help=False)

    # Dataset specific
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="list of datasets to combine for training",
        required=True,
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="list of datasets to combine for eval",
        required=True,
    )
   
    parser.add_argument(
        "--nextqa_features_path",
        default=os.path.join(DATA_DIR, name2folder["nextqa"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--nextqa_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextqa"], "train.csv"),
    )
    parser.add_argument(
        "--nextqa_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextqa"], "val.csv"),
    )
    parser.add_argument(
        "--nextqa_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextqa"], "test.csv"),
    )
    parser.add_argument(
        "--nextgqa_features_path",
        default=os.path.join(DATA_DIR, name2folder["nextqa"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--nextgqa_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextgqa"], "train.csv"),
    )
    parser.add_argument(
        "--nextgqa_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextgqa"], "val.csv"),
    )
    parser.add_argument(
        "--nextgqa_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["nextgqa"], "test.csv"),
    )

    # Training hyper-parameters
    parser.add_argument(
        "--mlm_prob",
        type=float,
        default=0.15,
        help="masking probability for the MLM objective",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--beta2", default=0.95, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size used for training"
    )
    parser.add_argument(
        "--batch_size_val",
        default=32,
        type=int,
        help="batch size used for eval",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--lr_drop",
        default=10,
        type=int,
        help="number of epochs after which the learning rate is reduced when not using linear decay",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--schedule",
        default="",
        choices=["", "linear_with_warmup"],
        help="learning rate decay schedule, default is constant",
    )
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.1,
        type=float,
        help="fraction of number of steps used for warmup when using linear schedule",
    )
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=400,
        help="print log every print_freq iterations",
    )

    # Model parameters
    parser.add_argument(
        "--ft_lm",
        dest="freeze_lm",
        action="store_false",
        help="whether to finetune the weights of the language model",
    )
    parser.add_argument(
        "--model_name",
        default="deberta-v2-xlarge",
        choices=(
            "bert-base-uncased",
            "bert-large-uncased",
            "deberta-v2-xlarge",
            "gpt-neo-1p3b",
            "gpt-j-6b",
            "gpt-neo-2p7b",
        ),
    )
    parser.add_argument(
        "--ds_factor_attn",
        type=int,
        default=0,
        help="downsampling factor for adapter attn",
    )
    parser.add_argument(
        "--ds_factor_ff",
        type=int,
        default=0,
        help="downsampling factor for adapter ff",
    )
    parser.add_argument(
        "--freeze_ln",
        dest="ft_ln",
        action="store_false",
        help="whether or not to freeze layer norm parameters",
    )
    parser.add_argument(
        "--ft_mlm",
        dest="freeze_mlm",
        action="store_false",
        help="whether or not to finetune the mlm head parameters",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout to use in the adapter"
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="whether to train the LM with or without language init",
    )
    parser.add_argument(
        "--n_ans",
        type=int,
        default=0,
        help="number of answers in the answer embedding module, it is automatically set",
    )
    parser.add_argument(
        "--ft_last",
        dest="freeze_last",
        action="store_false",
        help="whether to finetune answer embedding module or not",
    )

    # Run specific
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to run evaluation on val or test set",
    )
    parser.add_argument(
        "--save_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--presave_dir",
        default=PRESAVE_DIR,
        help="the actual save_dir is an union of presave_dir and save_dir",
    )
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--load",
        default="",
        help="path to load checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training if loading checkpoint",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="only run evaluation")
    parser.add_argument(
        "--num_workers", default=3, type=int, help="number of workers for dataloader"
    )

    # Distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Video and Text parameters
    parser.add_argument(
        "--max_feats",
        type=int,
        default=10,
        help="maximum number of video features considered, one per frame",
    )
    parser.add_argument(
        "--features_dim",
        type=int,
        default=768,
        help="dimension of the visual embedding space",
    )
    parser.add_argument(
        "--no_video",
        dest="use_video",
        action="store_false",
        help="disables usage of video",
    )
    parser.add_argument(
        "--no_context",
        dest="use_context",
        action="store_false",
        help="disables usage of speech",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="maximum number of tokens in the input text prompt",
    )
    parser.add_argument(
        "--max_atokens",
        type=int,
        default=5,
        help="maximum number of tokens in the answer",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="task induction before question for videoqa",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="suffix after the answer mask for videoqa",
    )

    # Demo
    parser.add_argument(
        "--question_example",
        default="",
        type=str,
        help="question example for demo",
    )
    parser.add_argument(
        "--video_example",
        default="",
        type=str,
        help="path to a video example for demo",
    )
    parser.add_argument(
        "--feat_type",
        default="CLIP",
        type=str,
        help="",
    )

    # Gaussian Mask
    parser.add_argument(
        "--gamma",
        default=0.8,
        type=float,
        help="control the confidence interval",
    )
    parser.add_argument(
        "--sigma",
        default=9,
        type=float,
        help="control the Gaussian width",
    )
    
    parser.add_argument(
        "--baseline",
        default='naive',
        type=str,
        help="choose from ['posthoc', 'naive', 'gdqa']",
    )
    
    parser.add_argument(
        "--vg_loss",
        default=0,
        type=float,
        help="use video question grounding loss",
    )

    return parser
