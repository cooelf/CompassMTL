# Task Compass: Scaling Multi-task Pre-training with Task Prefix

![](https://github.com/cooelf/CompassMTL/blob/main/fig/overview.png)

## Environment

- numpy
- torch
- transformers==4.17.0
- wandb
- sentencepiece
- sklearn
- datasets

## Data

Download data from [datasets](https://drive.google.com/file/d/17IbQLjf140ZCl0y--VvDd9E-AQuPrYiw/view?usp=sharing)

## Instructions

Training:

```bash
bash run_train.sh
```

evaluate:

```bash
bash run_evaluate.sh
```

## Commonsense Reasoning Models (ANLI and HellaSwag)

We provide the models and outputs for the ANLI and HellaSwag Commonsense Reasoning tasks:

Our sinlge models for ANLI and HellaSwag are available at [reasoning_models](https://drive.google.com/file/d/10nrvDN4-pR8rhBPyHIYw6QqKOw8fMQJx/view?usp=sharing)

The outputs can be found at ```model_outputs```.

## Reference

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{zhang2022task,
  title={Task Compass: Scaling Multi-task Pre-training with Task Prefix},
  author={Zhang, Zhuosheng and Wang, Shuohang and Xu, Yichong and Fang, Yuwei and Yu, Wenhao and Liu, Yang and Zhao, Hai and Zhu, Chenguang and Zeng, Michael},
  booktitle={arXiv preprint arXiv:2210.06277},
  year={2022}
}
```