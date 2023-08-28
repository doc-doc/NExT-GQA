GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python extract_feature_vid.py \
	--model_type 'CLIPL' \
	--mode 'test'