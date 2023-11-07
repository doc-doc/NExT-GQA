GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node 4 --use_env mc_dist.py \
    --combine_datasets nextgqa \
    --combine_datasets_val nextgqa \
    --save_dir='../../../data/gmodels/nextgqa/NG/FBLM/' \
    --lr=1e-5 \
    --schedule=linear_with_warmup \
    --nextgqa_features_path='../../../data/nextgqa/' \
    --ds_factor_ff=8 \
    --ds_factor_attn=8 \
    --feat_type="CLIPL" \
    --max_feats=32 \
    --baseline='NG' \
    --suffix="." \
    --sigma=9 \
    --batch_size=6 \
    --batch_size_val=64 \
    --max_tokens=100 \
    --epochs=20 \
    --print_freq=2000 \
    --vg_loss=0 \
    --load='../../../data/pretrain_models/FBLM/frozenbilm.pth' \
    #--load='../data/gmodels/nextgqa/gdqa/FBLM/CLIPL/AG01rep03/best_model.pth' \
    
   
    
    
    
    
