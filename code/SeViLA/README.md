# Self-Chained Image-Language Model for Video Localization and Question Answering
Please follow the <a href="https://github.com/Yui010206/SeViLA">official code</a> to reproduce the results on NExT-GQA. Afterwards, you can use the tools provided here to convert the predicted frame ids into timestamps for evaluation. To run the code, you need to copy ```tools``` into ```sevila``` and change the file paths accordingly.
```
python fid2seconds.py
python split_qa_los.py
```
