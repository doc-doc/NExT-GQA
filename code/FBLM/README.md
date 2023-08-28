## Introduction
This repo holds <a href="https://github.com/antoyang/FrozenBiLM">FrozenBiLM</a> baseline for <a href="https://github.com/doc-doc/NExT-GQA">NExT-GQA</a>.
</details>

<div align="center">
  <img width="80%" alt="Visually Grounded VideoQA" src="./misc/NExT-GQA.png">
</div>

## Preparation
FrozenBiLM relies on pretrained <a href="https://huggingface.co/microsoft/deberta-v2-xlarge">DeBerta-V2-XL</a>, you need to download it from hugging face via:
```
>cd workspace/data/pretrain_models
>git clone https://huggingface.co/microsoft/deberta-v2-xlarge
```
Additionally, you need to download the <a href="https://drive.google.com/file/d/1-_mUTxSjQj-NZ-le0-mDUftaikB_2nsU/view">cross-modal pretrained weights </a> from FrozenBiLM, and put it into ```workspace/data/pretrain_models/```.

Finally, please download our fine-tuned <a href="https://drive.google.com/file/d/1OOIVRN7dxd_2P0TfMR4bQSS6vyS34_hh/view?usp=drive_link">checkpoint</a> and extract it into ```workspace/data/gmodels/NG+/FBLM/```. 
## Inference
```
./shell/next_test.sh 0
```
## Evaluation
please refer to TempCLIP

## Train
```
./shell/nextqa_train.sh 0
```
It will train the model and save to the folder 'workspace/data/gmodels/'
## Result Visualization (NExT-GQA)
<div align="center">
  <img width="100%" alt="NExT-GQA for visually-grounded VideoQA" src="./misc/app-res.png">
</div>
## Citation 
```
@inproceedings{xiao2023nextgqa,
  title={Can I Trust Your Answer? Towards Visually Grounded Video Question Answering},
  author={Xiao, Junbin and Angela, Yao and Li, Yicong and Chua, Tat-Seng},
  booktitle={arXiv},
  pages={preprint},
  year={2023},
}
@inproceedings{yang2022frozenbilm,
  title = {Zero-Shot Video Question Answering via Frozen Bidirectional Language Models},
  author = {Yang, Antoine and Miech, Antoine and Sivic, Josef and Laptev, Ivan and Schmid, Cordelia},
  booktitle={Advances in Neural Information Processing Systems}
  year = {2022},
}
```
