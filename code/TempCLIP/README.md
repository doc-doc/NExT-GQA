# NExT-GQA: Visually Grounded Video Question Answering
<details open>
<summary> <b>Abstract</b> </summary>
We study visually grounded VideoQA by forcing vision-language models (VLMs) to answer questions and simultaneously ground the relevant video moments as visual evidence.
</details>

<div align="center">
  <img width="80%" alt="Visually Grounded VideoQA" src="./misc/NExT-GQA.png">
</div>

## Environment
Assume you have installed Anaconda, please do the following to setup the environment:
```
>conda create -n videoqa python==3.8
>conda activate videoqa
>conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia 
>git clone https://github.com/doc-doc/NExT-GQA.git
>pip install -r requirements.txt
```
## Preparation
Please create a data folder outside this repo, so you have two folders in your workspace 'workspace/data/' and 'workspace/NExT-GQA/'. 

Please download the related <a href="https://drive.google.com/file/d/101W4r6ibXJE2IOr6MINbNIMC3MFiN-us/view?usp=drive_link">video feature</a> or <a href="https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view">raw videos</a>. Extract the feature into ```workspace/data/nextqa/``` and <a href="https://drive.google.com/file/d/1OOIVRN7dxd_2P0TfMR4bQSS6vyS34_hh/view?usp=drive_link">checkpoint</a> files into ```workspace/data/gmodels/NG+/```. If you download the raw videos, you need to decode each video at 6fps and then extract the frame feature of CLIP via the script ```tools/extract_feat.sh```.

## Inference
```
./shell/next_test_dual.sh 0
```
## Evaluation
QA
```
python eval_next.py --folder TempCLIP --mode test
```
GroundedQA 
```
python eval_ground.py
``` 
## Result Visualization (NExT-GQA)
<div align="center">
  <img width="70%" alt="NExT-GQA for visually-grounded VideoQA" src="./misc/res.png">
</div>

## Train
We have provided all the scripts in the folder 'shells', you can start your training by specifying the GPU IDs behind the script. (If you have multiple GPUs, you can separate them with comma: ./shell/nextqa_train.sh 0,1)
```
./shell/nextqa_train_dual.sh 0
```
It will train the model and save to the folder 'workspace/data/gmodels/'

## TODO
[ ] Release FrozenGQA.

## Citation 
```
@inproceedings{xiao2023nextgqa,
  title={Can I Trust Your Answer? Towards Visually Grounded Video Question Answering},
  author={Xiao, Junbin and Angela, Yao and Li, Yicong and Chua, Tat-Seng},
  booktitle={arXiv},
  pages={preprint},
  year={2023},
}
```
