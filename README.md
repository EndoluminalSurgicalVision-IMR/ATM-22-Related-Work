# Airway Tree Modeling (ATM'22) Benchmark
[![Framework: Python](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
<div align=center><img src="figs/bannar.png"></div>

**News [2024/08/10]:** We have updated the Long-Term Test-Phase Results for the hidden test set (150 CT scans) after MICCAI 2022. The **Docker or API submission** via e-mail(**IMR-ATM22@outlook.com**) for the test phase is welcome, and we will update the results timely.
Please refer to  [**Long-Term-Test-Phase-Results**](https://github.com/Puzzled-Hui/ATM-22-Related-Work/tree/main/Long-Term-Test-Phase-Results) for detailed information!

**News [2024/07/31]:** We have integrated the Binary Airway Segmentation (BAS) dataset in this repository! Please refer to  [**BAS Dataset**](https://github.com/Puzzled-Hui/ATM-22-Related-Work/tree/main/BAS-Dataset) for detailed information!

**Official repository for MICCAI 2022 Challenge:** [**_Multi-site, Multi-Domain Airway Tree Modeling (ATM’22)_**](https://atm22.grand-challenge.org/homepage/).
> Minghui Zhang, Yangqian Wu, Hanxiao Zhang, Yulei Qin, Hao Zheng, Weihao Yu, Jiayuan Sun, Guang-Zhong Yang, Yun Gu.
>> ATM'22 organization team: Institute of Medical Robotics, Shanghai Jiao Tong University & Department of Respiratory and Critical Care Medicine, Shanghai Chest Hospital

**Highlight: The benchmark manuscritpt: [Multi-site, Multi-domain Airway Tree Modeling](https://www.sciencedirect.com/science/article/abs/pii/S1361841523002177)** has been accepted for publication in **Medical Image Analysis**. 
If ATM'22 challenge, dataset or this repo is helpful to your scientific research, please cite our [**benchmark paper**](https://www.sciencedirect.com/science/article/abs/pii/S1361841523002177):
```bibtex
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
2. [Citation and Dataset Rule](#📝-Citation-and-Dataset-Rule)
3. [Related Works](#Related-Works)

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

<!-- |2|[]()|[]()|2|[]()| -->
| Date | Author | Title |     Conf/Jour      | Code |
|:----:| :---: | :---: |:------------------:| :---: |
| 2023 |[ATM'22 Organizers and Participants]()|[Multi-site, Multi-domain Airway Tree Modeling](https://arxiv.org/abs/2303.05745)|MedIA|[Official](https://github.com/Puzzled-Hui/ATM-22-Related-Work)|
| 2025 |[Dexu Wang]()|[Airway segmentation using Uncertainty-based Double Attention Detail Supplement Network](https://www.sciencedirect.com/science/article/pii/S1746809425001594)| BSPC | [——]()  |
| 2024 |[Chenyu Li]()|[Airway Labeling Meets Clinical Applications: Reflecting Topology Consistency and Outliers via Learnable Attentions](https://arxiv.org/abs/2410.23854)| IPCAI 2025 | [Official](https://github.com/EndoluminalSurgicalVision-IMR/Reflecting-Topology-Consistency-and-Abnormality-via-Learnable-Attentions)  |
| 2024 |[Minghui Zhang](https://scholar.google.com/citations?hl=zh-CN&user=CepcxZcAAAAJ)|[Implicit Representation Embraces Challenging Attributes of Pulmonary Airway Tree Structures](https://papers.miccai.org/miccai-2024/paper/0885_paper.pdf)| MICCAI | [Official](https://github.com/EndoluminalSurgicalVision-IMR/DGCI) |
| 2024 |[Bingyu Yang]()|[Multi-Stage Airway Segmentation in Lung CT Based on Multi-scale Nested Residual UNet](https://arxiv.org/abs/2410.18456)| Arxiv | [——]() |
| 2024 |[Kangxian Xie]()|[Efficient Anatomical Labeling of Pulmonary Tree Structures via Deep Point-Graph Representation-based Implicit Fields](https://www.sciencedirect.com/science/article/pii/S1361841524002925)| MedIA | [Official](https://github.com/M3DV/pulmonary-tree-labeling) |
| 2024 |[Ruiyun Zhu]()|[Semi-supervised Tubular Structure Segmentation with Cross Geometry and Hausdorff Distance Consistency](https://papers.miccai.org/miccai-2024/paper/0651_paper.pdf)| MICCAI | [——]() |
| 2024 |[Xuan Yang]()|[Airway Segmentation Based on Topological Structure Enhancement Using Multi-task Learning](https://papers.miccai.org/miccai-2024/paper/3005_paper.pdf)| MICCAI | [Official](https://github.com/xyang-11/airway_seg) |
| 2024 |[Yang Nan](https://scholar.google.com.hk/citations?user=bfIiCesAAAAJ&hl=zh-CN)|[Hunting imaging biomarkers in pulmonary fibrosis: Benchmarks of the AIIB23 challenge](https://www.sciencedirect.com/science/article/pii/S1361841524001786)| MedIA | [——]() |
| 2024 |[Puyang Wang]()|[Accurate Airway Tree Segmentation in CT Scans via Anatomy-aware Multi-class Segmentation and Topology-guided Iterative Learning](https://ieeexplore.ieee.org/abstract/document/10574168)|IEEE TMI|[——]()|
| 2024 |[Ali Keshavarzi](https://scholar.google.com/citations?user=GBaZhSQAAAAJ&hl=zh-CN&oi=hraa)|[Few-Shot Airway-Tree Modeling using Data-Driven Sparse Priors](https://arxiv.org/abs/2303.05745)|ISBI|[——](https://arxiv.org/abs/2407.04507)|
| 2023 |[Minghui Zhang](https://scholar.google.com/citations?hl=zh-CN&user=CepcxZcAAAAJ)|[Towards Connectivity-Aware Pulmonary Airway Segmentation](https://ieeexplore.ieee.org/abstract/document/10283811)|       IEEE JBHI       |[Official](https://github.com/EndoluminalSurgicalVision-IMR/Connectivity-Aware-Airway-Segmentaion)|
| 2023 |[Diedre S. Carmo](https://scholar.google.com/citations?user=YjA3hdoAAAAJ&hl=pt-BR)|[MEDPSeg: End-to-end segmentation of pulmonary structures and lesions in computed tomography](https://arxiv.org/pdf/2312.02365.pdf)|  Arxiv   |[Official](https://github.com/miclab-unicamp/medpseg)|
| 2023 |[Yan Hu]()|[Large-Kernel Attention Network with Distance Regression and Topological Self-correction for Airway Segmentation](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_10)|       AJCAI       |[——]()|
| 2023 |[Ron Alterovitz](https://scholar.google.com/citations?user=-XEZA0UAAAAJ&hl=zh-CN&oi=sra)|[Landmark Based Bronchoscope Localization for Needle Insertion Under Respiratory Deformation](https://robotics.cs.unc.edu/publications/Fried2023_IROS.pdf)|       IROS       |[——]()|
| 2023 |[Karen-Helene Støverud](https://scholar.google.com/citations?hl=zh-CN&user=6LBkICsAAAAJ&view_op=list_works&sortby=pubdate)|[AeroPath: An airway segmentation benchmark dataset with challenging pathology](https://arxiv.org/abs/2311.01138)|       Arxiv      |[Official](https://github.com/raidionics/AeroPath)|
| 2023 |[Wehao Yu](https://scholar.google.com/citations?hl=zh-CN&user=fCzlLE4AAAAJ)|[AirwayFormer: Structure-Aware Boundary-Adaptive Transformers for Airway Anatomical Labeling](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_37)| MICCAI |[Official](https://github.com/EndoluminalSurgicalVision-IMR/AirwayFormer)|
| 2023 |[Difei Gu]()|[Semi-Supervised Pulmonary Airway Segmentation with Two-Stage Feature Specialization Mechanism](https://ieeexplore.ieee.org/abstract/document/10230329/)|ISBI|[——]()|
| 2023 |[Ziqiao Weng]()|[Topology Repairing of Disconnected Pulmonary Airways and Vessels: Baselines and a Dataset](https://arxiv.org/abs/2306.07089)|MICCAI|[——]()|
| 2023 |[Mingyue Zhao]()|[GDDS: Pulmonary Bronchioles Segmentation with Group Deep Dense Supervision](https://arxiv.org/pdf/2303.09212.pdf)|Arxiv|[——]()|
| 2023 |[Hanxiao Zhang]()|[Deep anatomy learning for lung airway and artery-vein segmentation with synthetic contrast-enhanced CT generation]()|IPCAI|[——]()|
| 2023 |[Yanan Wu](https://scholar.google.com/citations?user=phzPWXMAAAAJ&hl=zh-CN&oi=sra)|[Two-stage Contextual Transformer-based Convolutional Neural Network for Airway Extraction from CT Images](https://arxiv.org/abs/2212.07651)|Artificial Intelligence in Medicine|[——]()|
| 2022 |[Zeyu Tang](https://scholar.google.com/citations?user=aSSpwswAAAAJ&hl=zh-CN&oi=sra)|[Human Treelike Tubular Structure Segmentation: A Comprehensive Review and Future Perspectives](https://www.sciencedirect.com/science/article/pii/S0010482522009490)|CBM|[——]()|
| 2022 |[Shuai Chen](https://scholar.google.com/citations?user=aSSpwswAAAAJ&hl=zh-CN&oi=sra)|[Label Refinement Network from Synthetic Error Augmentation for Medical Image Segmentation](https://arxiv.org/abs/2209.06353)|IEEE TMI|[Official](https://github.com/ShuaiChenBIGR/Label-refinement-network)|
| 2023 |[Zeyu Tang](https://scholar.google.com.hk/citations?user=Hp5DOVIAAAAJ&hl=zh-CN&oi=sra)|[Adversarial Transformer for Repairing Human Airway Segmentation](https://arxiv.org/pdf/2210.12029.pdf)|IEEE JBHI|[——]()|
| 2022 |[Yang Nan](https://scholar.google.com.hk/citations?user=bfIiCesAAAAJ&hl=zh-CN)|[Fuzzy Attention Neural Network to Tackle Discontinuity in Airway Segmentation](https://arxiv.org/abs/2209.02048)| IEEE TNNLS | [Official](https://github.com/Nandayang/FANN-for-airway-segmentation) |
| 2022 |[Wehao Yu](https://scholar.google.com/citations?hl=zh-CN&user=fCzlLE4AAAAJ)|[TNN: Tree Neural Network for Airway Anatomical Labeling](https://ieeexplore.ieee.org/document/9878127)|      IEEE TMI      |[Official](https://github.com/haozheng-sjtu/airway-labeling)|
| 2022 |[Yun Gu](https://scholar.google.com/citations?user=0pX32mkAAAAJ&hl=zh-CN)|[Vision-Kinematics-Interaction for Robotic-Assisted Bronchoscopy Navigation](https://ieeexplore.ieee.org/document/9830773)|      IEEE TMI      |[——]()|
| 2022 |[Minghui Zhang](https://scholar.google.com/citations?hl=zh-CN&user=CepcxZcAAAAJ)|[CFDA: Collaborative Feature Disentanglement and Augmentation for Pulmonary Airway TreeModeling of COVID-19 CTs](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_48)|       MICCAI       |[Official](https://github.com/Puzzled-Hui/CFDA)|
| 2022 |[Haifan Gong](https://haifangong.github.io/)|[BronchusNet: Region and Structure Prior Embedded Representation Learning for Bronchus Segmentation and Classification](https://arxiv.org/abs/2205.06947)|       Arxiv        |[——]()|
| 2021 |[Wehao Yu](https://scholar.google.com/citations?hl=zh-CN&user=fCzlLE4AAAAJ)|[BREAK: Bronchi Reconstruction by gEodesic transformation And sKeleton embedding](https://ieeexplore.ieee.org/abstract/document/9761697/)|        ISBI        |[——]()|
| 2021 |[Yangqian Wu](https://www.researchgate.net/profile/Yangqian-Wu)|[LTSP: long-term slice propagation for accurate airway segmentation](https://link.springer.com/article/10.1007/s11548-022-02582-7)|       IJCARS       |[——]()|
| 2021 |[Minghui Zhang](https://scholar.google.com/citations?hl=zh-CN&user=CepcxZcAAAAJ)|[Fda: Feature decomposition and aggregation for robust airway segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87722-4_3)|    DART@MICCAI     |[——]()|
| 2021 |[Hao Zheng](https://scholar.google.com/citations?hl=zh-CN&user=LsJVCSoAAAAJ&view_op=list_works&sortby=pubdate)|[Refined Local-imbalance-based Weight for Airway Segmentation in CT](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_39)|       MICCAI       |[Official](https://github.com/haozheng-sjtu/Local-imbalance-based-Weight)|
| 2021 |[Hao Zheng](https://scholar.google.com/citations?hl=zh-CN&user=LsJVCSoAAAAJ&view_op=list_works&sortby=pubdate)|[Alleviating class-wise gradient imbalance for pulmonary airway segmentation](https://ieeexplore.ieee.org/abstract/document/9427208/)|      IEEE TMI      |[Official](https://github.com/haozheng-sjtu/3d-airway-segmentation)|
| 2021 |[A. Garcia-Uceda Juarez](https://scholar.google.com/citations?user=5pLmIVYAAAAJ&hl=zh-CN&oi=sra)|[Automatic airway segmentation from Computed Tomography using robust and efficient 3-D convolutional neural networks](https://www.researchgate.net/profile/Raghavendra-Selvan-2/publication/350512163_Automatic_airway_segmentation_from_Computed_Tomography_using_robust_and_efficient_3-D_convolutional_neural_networks/links/60674588299bf1252e2432b1/Automatic-airway-segmentation-from-Computed-Tomography-using-robust-and-efficient-3-D-convolutional-neural-networks.pdf)| Scientific Reports |[Official](https://github.com/antonioguj/bronchinet)|
| 2020 |[Hanxiao Zhang]()|[Pathological airway segmentation with cascaded neural networks for bronchoscopic navigation](https://ieeexplore.ieee.org/abstract/document/9196756)|     IEEE ICRA      |[——]()|
| 2020 |[Yulei Qin](https://scholar.google.com/citations?user=vBnuTjwAAAAJ&hl=zh-CN&oi=sra)|[Learning Tubule-Sensitive CNNs for Pulmonary Airway and Artery-Vein Segmentation in CT](https://ieeexplore.ieee.org/abstract/document/9363945)|      IEEE TMI      |[Official](http://www.pami.sjtu.edu.cn/Show/56/146)|
| 2020 |[Raghavendra Selvan](https://raghavian.github.io/)|[Graph refinement based airway extraction using mean-field networks and graph neural networks](https://www.sciencedirect.com/science/article/pii/S1361841520301158)|        MedIA         |[Official](https://github.com/raghavian/graph_refinement)|
| 2019 |[Jihye Yun](https://sites.google.com/view/jihyeyunphd)|[Improvement of fully automated airway segmentation on volumetric computed tomographic images using a 2.5 dimensional convolutional neural net](https://www.sciencedirect.com/science/article/pii/S1361841518308508)|        MedIA         |[——]()|
| 2019 |[Chenglong Wang](https://scholar.google.com/citations?user=pLtUR5cAAAAJ&hl=zh-CN&oi=sra)|[Tubular structure segmentation using spatial fully connected network with radial distance loss for 3D medical images](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_39)|       MICCAI       |[——]()|
| 2019 |[A. Garcia-Uceda Juarez](https://scholar.google.com/citations?user=5pLmIVYAAAAJ&hl=zh-CN&oi=sra)|[A joint 3D UNet-graph neural network-based method for airway segmentation from chest CTs](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_67)|    MLMI@MICCAI     |[——]()|
| 2019 |[Yulei Qin](https://scholar.google.com/citations?user=vBnuTjwAAAAJ&hl=zh-CN&oi=sra)|[AirwayNet: A Voxel-Connectivity Aware Approach for Accurate Airway Segmentation Using Convolutional Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_24)|       MICCAI       |[——]()|
| 2017 |[Qier Meng](https://scholar.google.com/citations?user=mVPvS2AAAAAJ&hl=zh-CN&oi=sra)|[Tracking and segmentation of the airways in chest CT using a fully convolutional network](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_23)|       MICCAI       |[——]()|
| 2017 |[Jean-Paul Charbonnier](https://scholar.google.com/citations?user=K8Pz4m0AAAAJ&hl=zh-CN&oi=sra)|[Improving airway segmentation in computed tomography using leak detection with convolutional networks](https://www.sciencedirect.com/science/article/pii/S136184151630202X)|        MedIA         |[——]()|
| 2017 |[Dakai Jin](https://dakjin.github.io/)|[3D convolutional neural networks with graph refinement for airway segmentation using incomplete data labels](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_17)|    MLMI@MICCAI     |[——]()|
| 2015 |[Ziyue Xu](https://scholar.google.com/citations?hl=zh-CN&user=gmUta74AAAAJ)|[A hybrid method for airway segmentation and automated measurement of bronchial wall thickness on CT](https://www.sciencedirect.com/science/article/pii/S1361841515000705)|        MedIA         |[——]()|
| 2012 |[Pechin Lo](https://scholar.google.com/citations?user=p6V4-AUAAAAJ&hl=zh-CN&oi=sra)|[Extraction of airways from CT (EXACT'09)](https://ieeexplore.ieee.org/abstract/document/6249784)|      IEEE TMI      |[——]()|


## 📝 Citation and Dataset Rule
If you find this repo's papers, codes, and ATM'22 challenge data are helpful to your research, and if you use our dataset provided by ATM'22 for your scientific research, 
**please cite the following works:**

```bibtex
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
```bibtex
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

