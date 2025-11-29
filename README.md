# [ICCV 2025] Closed-Loop Transfer for Weakly-supervised Affordance Grounding

by Jiajin Tang*, Zhengxuan Wei*, Ge Zheng, Sibei Yang†

*Equal contribution; †Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-2510.17384-b31b1b.svg)](https://arxiv.org/abs/2510.17384)

## Abstract

Humans can perform previously unexperienced interactions with novel objects simply by observing others engage with them. Weakly-supervised affordance grounding mimics this process by learning to locate object regions that enable actions on egocentric images, using exocentric interaction images with image-level annotations. However, extracting affordance knowledge solely from exocentric images and transferring it one-way to egocentric images limits the applicability of previous works in complex interaction scenarios. Instead, this study introduces LoopTrans, a novel closed-loop framework that not only transfers knowledge from exocentric to egocentric but also transfers back to enhance exocentric knowledge extraction. Within LoopTrans, several innovative mechanisms are introduced, including unified cross-modal localization and denoising knowledge distillation, to bridge domain gaps between object-centered egocentric and interaction-centered exocentric images while enhancing knowledge transfer. Experiments show that LoopTrans achieves consistent improvements across all metrics on image and video benchmarks, even handling challenging scenarios where object interaction regions are fully occluded by the human body. 

## Framework
<p align="center">
  <img src="assets/framework.png" width="700"/>
</p>

## Prerequisites

### 0. Clone this repo

```
git clone git clone https://github.com/SooLab/LoopTrans.git
cd LoopTrans
```

### 1. Requirements

Please create a conda environment and install the required packages:
```
conda create -n looptrans python=3.7 -y
conda activate looptrans
pip install -r requirements.txt
```

### 2. Dataset

Download the AGD20K dataset
from [ [Google Drive](https://drive.google.com/file/d/1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1IRfho7xDAT0oJi5_mvP1sg) (g23n) ]
.

### 3. Cluster Module

We use [ACSeg](https://arxiv.org/abs/2210.05944) as our clustering module. We have pre-trained two modules via unsupervised learning on the AGD20K "Seen" and "Unseen" training sets, respectively. You can download these models from [Google Drive](https://drive.google.com/drive/folders/1w8Y4KI5AXyoXu469RFKjnn-TGAx1HbLC?usp=drive_link) and place the files in the `./ckpts` directory.


## Training

Run following commands to start training:

```
python train.py --data_root <PATH_TO_DATA>
```

## Testing

Run following commands to start testing:

```
python test.py --data_root <PATH_TO_DATA> --model_file <PATH_TO_MODEL>
```

## Citation

```
@inproceedings{tang2025closed,
  title={Closed-Loop Transfer for Weakly-supervised Affordance Grounding},
  author={Tang, Jiajin and Wei, Zhengxuan and Zheng, Ge and Yang, Sibei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9530--9539},
  year={2025}
}
```

## Anckowledgement

This repo is based on [LOCATE](https://github.com/Reagan1311/LOCATE)
, [Cross-View-AG](https://github.com/lhc1224/Cross-View-AG). Thanks for their great work!
