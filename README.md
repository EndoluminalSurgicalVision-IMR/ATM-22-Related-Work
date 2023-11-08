# Airway Tree Modeling (ATM'22) Benchmark
[![Framework: Python](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
<div align=center><img src="figs/bannar.png"></div>

**Official repository for MICCAI 2022 Challenge:** [**_Multi-site, Multi-Domain Airway Tree Modeling (ATM’22)_**](https://atm22.grand-challenge.org/homepage/).
> Minghui Zhang, Yangqian Wu, Hanxiao Zhang, Yulei Qin, Hao Zheng, Weihao Yu, Jiayuan Sun, Guang-Zhong Yang, Yun Gu.
>> ATM'22 organization team: Institute of Medical Robotics, Shanghai Jiao Tong University & Department of Respiratory and Critical Care Medicine, Shanghai Chest Hospital

**Highlight: The benchmark manuscritpt: [Multi-site, Multi-domain Airway Tree Modeling](https://www.sciencedirect.com/science/article/abs/pii/S1361841523002177)** has been accepted for publication in Medical Image Analysis. 
If ATM'22 challenge, dataset or this repo is helpful to your scientific research, please cite the [**paper**](https://www.sciencedirect.com/science/article/abs/pii/S1361841523002177):
```
@article{zhang2023multi,
  title={Multi-site, Multi-domain Airway Tree Modeling},
  author={Zhang, Minghui and Wu, Yangqian and Zhang, Hanxiao and Qin, Yulei and Zheng, Hao and Tang, Wen and Arnold, Corey and Pei, Chenhao and Yu, Pengxin and Nan, Yang and others},
  journal={Medical Image Analysis},
  volume={90},
  pages={102957},
  year={2023},
  publisher={Elsevier}
}
```

## Content
1. [ATM'22 Challenge Collection](#ATM'22-Challenge-Collection)
    - [Registration](#Registration)
    - [Baseline and Docker Tutorial](#Baseline-and-Docker-Tutorial)
    - [Evaluation](#Evaluation)
2. [Related Works](#Related-Works)
3. [Citation and Dataset Rule](#Citation-and-Dataset-Rule)

## ATM'22 Challenge Collection
This challenge is open-call (challenge opens for new submissions after MICCAI 2022 deadline). The online evaluation for individual algorithms is still working. 

### Registration
The registration information, and detailed information could refer to [**Registration Page**](https://atm22.grand-challenge.org/registration/). 

**NOTE:** All participants must send the signed data agreement to IMR-ATM22@outlook.com for successful registration, as required in [**Registration Page**](https://atm22.grand-challenge.org/registration/). 

### Baseline and Docker Tutorial
We provide a baseline model and a detailed docker tutorial, please refer to: [**Baseline and Docker Example**](https://github.com/Puzzled-Hui/ATM-22-Related-Work/tree/main/baseline-and-docker-example) for detailed instructions.

### Evaluation
The evaluation code is provided in [**Evaluation**](https://github.com/Puzzled-Hui/ATM-22-Related-Work/tree/main/evaluation).

## Related Works
We collected the papers related to pulmonary airway segmentation and bronchoscopy navigation:

Please refer to [**Related Works**](https://github.com/Puzzled-Hui/ATM-22-Related-Work/tree/main/related_works) for detailed information.


## Citation and Dataset Rule
If you find this repo's papers, codes, and ATM'22 challenge data are helpful to your research, and if you use our dataset provided by ATM'22 for your scientific research, 
**please cite the following works:**

```
@inproceedings{zhang2022cfda,
  title={CFDA: Collaborative Feature Disentanglement and Augmentation for Pulmonary Airway Tree Modeling of COVID-19 CTs},
  author={Zhang, Minghui and Zhang, Hanxiao and Yang, Guang-Zhong and Gu, Yun},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part I},
  pages={506--516},
  year={2022},
  organization={Springer}
}

@article{zheng2021alleviating,
  title={Alleviating class-wise gradient imbalance for pulmonary airway segmentation},
  author={Zheng, Hao and Qin, Yulei and Gu, Yun and Xie, Fangfang and Yang, Jie and Sun, Jiayuan and Yang, Guang-Zhong},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={9},
  pages={2452--2462},
  year={2021},
  publisher={IEEE}
}

@inproceedings{yu2022break,
  title={BREAK: Bronchi Reconstruction by gEodesic transformation And sKeleton embedding},
  author={Yu, Weihao and Zheng, Hao and Zhang, Minghui and Zhang, Hanxiao and Sun, Jiayuan and Yang, Jie},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}

@inproceedings{qin2019airwaynet,
  title={Airwaynet: a voxel-connectivity aware approach for accurate airway segmentation using convolutional neural networks},
  author={Qin, Yulei and Chen, Mingjian and Zheng, Hao and Gu, Yun and Shen, Mali and Yang, Jie and Huang, Xiaolin and Zhu, Yue-Min and Yang, Guang-Zhong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={212--220},
  year={2019},
  organization={Springer}
}
```

Alternatively, you could also cite our challenge benchmark manuscript:
```
@article{zhang2023multi,
  title={Multi-site, Multi-domain Airway Tree Modeling},
  author={Zhang, Minghui and Wu, Yangqian and Zhang, Hanxiao and Qin, Yulei and Zheng, Hao and Tang, Wen and Arnold, Corey and Pei, Chenhao and Yu, Pengxin and Nan, Yang and others},
  journal={Medical Image Analysis},
  volume={90},
  pages={102957},
  year={2023},
  publisher={Elsevier}
}
```