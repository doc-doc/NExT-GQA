# Can I Trust Your Answer? Towards Visually Grounded Video Question Answering
<details open>
<summary> <b>Introduction</b> </summary>
We study visually grounded VideoQA by forcing vision-language models (VLMs) to answer questions and simultaneously ground the relevant video moments as visual evidence. We constructed the NExT-GQA dataset by extending the QA annotations in NExT-QA with corresponding time-span annotations. With NExT-GQA, we examine a variety of VLMs and find that they excel at QA but are weak in grounding the visual evidence. As a remedy, we further explore and suggest a Gaussian mask learning strategy and demonstrate its effectiveness across different backbones for both video grounding and QA.
</details>

<div align="center">
  <img width="100%" alt="Visually Grounded VideoQA" src="./misc/NExT-GQA.png">
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

Please download the related <a href="https://drive.google.com/file/d/101W4r6ibXJE2IOr6MINbNIMC3MFiN-us/view?usp=drive_link">video feature</a> or <a href="https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view">raw videos</a>. Extract the feature into ```workspace/data/nextqa/CLIPL/```. If you download the raw videos, you need to decode each video at 6fps and then extract the frame feature of CLIP via the script provided in ```code/TempCLIP/tools/extract_feat.sh```.

Please follow the instructions in ```code``` for train and test the respective models.

## Result Visualization (NExT-GQA)
<div align="center">
  <img width="100%" alt="NExT-GQA for visually-grounded VideoQA" src="./misc/res.png">
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
```
