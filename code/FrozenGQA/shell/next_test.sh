GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python mc_dist.py \
	--test \
	--combine_datasets nextgqa \
	--combine_datasets_val nextgqa \
	--nextgqa_features_path='../../../data/nextgqa/' \
	--save_dir='../../../data/gmodels/NG+/FrozenGQA/' \
	--features_dim=768 \
	--feat_type="CLIPL" \
	--max_feats=32 \
	--ds_factor_ff=8 \
	--ds_factor_attn=8 \
	--baseline='NG+' \
	--suffix="." \
	--batch_size_val=64 \
	--max_tokens=100 \
	--num_worker=4 \
	--gamma=1.0 \
	--load='../../../data/gmodels/NG+/FrozenGQA/best_model.pth' \
