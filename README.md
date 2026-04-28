This repo is the official implementation of the CVPR-2026 Findings paper: [Asymmetric Collaborative Distillation for Asymmetric Image Retrieval]().
The codebase is built upon [AIR-Distiller](https://github.com/SCY-X/D3still), with the main difference being that the gallery network is learnable, while the query network is kept frozen. The query network is initialized either with the pretrained weights released by AIR-Distiller or with weights trained based on its original framework.

NOTE: On the MSMT17 dataset, we fixed an issue where the downsampling stride in the final stage of ResNet-IBN was not set to 1. Based on the corrected architecture, we retrained the ResNet101-IBN model and updated the corresponding ACD results on MSMT17. The new results are shown in the table below.

| Method | mAP (%) | R1 (%) |
|:------:|:-------:|:------:|
| FitNet | 38.28 | 56.38 |
| FitNet + ACD | 40.63 (+2.35) | 59.20 (+2.82) |
| CC | 39.68 | 58.08 |
| CC + ACD | 40.63 (+0.95) | 59.20 (+1.12) |
| CSD | 39.31 | 57.39 |
| CSD + ACD | 40.72 (+1.41) | 58.98 (+1.59) |
| RAML | 39.76 | 58.29 |
| RAML + ACD | 41.20 (+1.44) | 59.77 (+1.48) |
| ROP | 37.25 | 56.81 |
| ROP + ACD | 39.10 (+1.85) | 58.26 (+1.45) |
| D3still | 40.90 | 60.17 |
| D3still + ACD | 42.43 (+1.53) | 61.85 (+1.68) |
| UGD | 41.98 | 61.22 |
| UGD + ACD | 41.99 (+0.01) | 61.63 (+0.41) |

### Introduction

ACD further improves the following distillation methods on Caltech-UCSD Birds 200 (CUB-200-2011), In-Shop Clothes Retrieval (In-Shop), Stanford Online Products (SOP), MSMT17 and VeRi-776:
|Method|Publication|YEAR|
|:---:|:---:|:---:|
|[FitNet](https://arxiv.org/abs/1412.6550) |ICLR|2015 |
|[CC](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) |ICCV| 2019|
|[CSD](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html) |CVPR|2023 |
|[RAML](https://openaccess.thecvf.com/content/WACV2023/html/Suma_Large-to-Small_Image_Resolution_Asymmetry_in_Deep_Metric_Learning_WACV_2023_paper.html)|WACV|2023|
|[ROP](https://openreview.net/forum?id=dYHYXZ3uGdQ)|ICLR|2023|
|[D3still](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html) |CVPR|2024|
|[UGD](https://www.sciencedirect.com/science/article/pii/S0893608025001820) |Neural Networks|2025|

### Installation

Environments:

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- ptflops 0.7.4
- cuda 11.8

Install the package:

```
sudo pip3 install -r requirements.txt
```

### Getting started

#### 0. download data
- The dataset has been prepared in the format we read at the link: https://pan.baidu.com/s/1-Ig5F4fMeABCergfTittYA?pwd=3c4c or https://drive.google.com/drive/folders/1CMXqNT3C25_Zy2gyKbC2NJipn90yc2pB?usp=sharing. Please download the data and untar it to `XXXX/data` via `unzip XXXX`. For example,  `unzip CUB_200_2011.zip`. Finally, the data file directory should be as follows:


  XXXX/data/  
    &nbsp; &nbsp; &nbsp; &nbsp; └── CUB_200_2011  
    &nbsp; &nbsp; &nbsp; &nbsp; └── InShop  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Stanford_Online_Products  
    &nbsp; &nbsp; &nbsp; &nbsp; └── MSMT17  
    &nbsp; &nbsp; &nbsp; &nbsp; └── VeRi776

#### 1. download teacher models
- Our teacher models are at https://pan.baidu.com/s/1-LrkEMfUR49iZ-KK-ojR2g?pwd=hmqn or https://drive.google.com/drive/folders/1ThCSwQEcGzON9Ju8GXifIZU2MAomcqBO?usp=sharing, please download the checkpoints to `ACD/download_teacher_ckpts`

#### 2. download student models
- Our student models are at https://pan.baidu.com/s/1RHlkgFOdBnthTNxVgeVGRQ?pwd=sn5v or https://drive.google.com/drive/folders/1kOx5OtUalIcs6JAiVG-jo8Jxns2MySPI?usp=sharing, please download the checkpoints to `ACD/download_student_ckpts`

#### 3. Path setting
- Please modify the following line in `ACD/tools/train.py`, `ACD/tools/test.py` and `ACD/tools/test_ours.py` :  
`sys.path.append(os.path.abspath("XXXXX/ACD"))`  
Replace `"XXXXX/ACD"` with the absolute path of your project to ensure correct module imports.

 **Example** (assuming the project path is `/home/user/ACD`):  
```python
import sys  
import os  
sys.path.append(os.path.abspath("/home/user/ACD"))
```
- Please set the `ROOT_DIR` path in the configuration file, i.e., XXX.yaml to the absolute path of the `data` folder.  
  
**Example** (assuming the data path is `/home/user/data`):  
```yaml
DATASETS:
  NAMES: "SOP"
  ROOT_DIR: "/home/user/data"
```


#### 4. Training 

 ```bash
  # For example, under the setting where ResNet-101 serves as the gallery network and ResNet-18 serves as the query network, ACD is introduced into the D3 method.
  python ACD/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
  ```

  - By default, the ImageNet pre-trained model will be used for training. The model will be automatically downloaded from the internet on the first run.  
  If you want to use a different pre-trained model, modify the `STUDENT_PRETRAIN_PATH` in the YAML configuration file.  


#### 5. Evaluation

 - During inference, you can first navigate to `ACD/utils/rank_cylib` and run the following commands to enable sorting with C language, which helps reduce inference time:  

```bash
python3 setup.py build_ext --inplace
rm -rf build
```

 ##### 5.1 Asymmetric Image Retrieval Evaluation

 ```bash
  # For example, under the setting where ResNet-101 serves as the gallery network and ResNet-18 serves as the query network, ACD is introduced into the D3 method.
  python ACD/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
 ```

  ##### 5.2 symmetric Image Retrieval Evaluation

To evaluate the performance of symmetric image retrieval, first set the distiller type in the YAML file as:
```yaml
DISTILLER:
  TYPE: "NONE"
```

```bash
# Example: evaluate the setting where ResNet-101 is used as the gallery network
# and ResNet-18 is used as the query network.
python ACD/tools/test_ours.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml
 ```

### Custom Distillation Method

1. create a python file at `ACD/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `ACD/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `ACD/config/defaults.py`

4. create a new config file and test it.



# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@InProceedings{ACD,
    author    = {Yi Xie, Huaidong Zhang, Xuandi Luo, Yan Zhou, Shengfeng He},
    title     = {Asymmetric Collaborative Distillation for Asymmetric Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
}
```

```BibTeX
@InProceedings{Xie_2024_CVPR,
    author    = {Xie, Yi and Lin, Yihong and Cai, Wenjie and Xu, Xuemiao and Zhang, Huaidong and Du, Yong and He, Shengfeng},
    title     = {D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17181-17190}
}
```

```BibTeX
@article{zhang2025unambiguous,
  title={Unambiguous granularity distillation for asymmetric image retrieval},
  author={Zhang, Hongrui and Xie, Yi and Zhang, Haoquan and Xu, Cheng and Luo, Xuandi and Chen, Donglei and Xu, Xuemiao and Zhang, Huaidong and Heng, Pheng Ann and He, Shengfeng},
  journal={Neural Networks},
  volume={187},
  pages={107303},
  year={2025},
  publisher={Elsevier}
}
```

# License

ACD is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement
- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
