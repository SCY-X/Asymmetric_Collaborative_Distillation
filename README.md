This repo is the official implementation of the CVPR-2026 Findings paper: [Asymmetric Collaborative Distillation for Asymmetric Image Retrieval]().
The codebase is built upon [AIR-Distiller](https://github.com/SCY-X/D3still), with the main difference being that the gallery network is learnable, while the query network is kept frozen. The query network is initialized either with the pretrained weights released by AIR-Distiller or with weights trained based on its original framework.


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

0. download data
- The dataset has been prepared in the format we read at the link: https://pan.baidu.com/s/1-Ig5F4fMeABCergfTittYA?pwd=3c4c or https://drive.google.com/drive/folders/1CMXqNT3C25_Zy2gyKbC2NJipn90yc2pB?usp=sharing. Please download the data and untar it to `XXXX/data` via `unzip XXXX`. For example,  `unzip CUB_200_2011.zip`. Finally, the data file directory should be as follows:


  XXXX/data/  
    &nbsp; &nbsp; &nbsp; &nbsp; └── CUB_200_2011  
    &nbsp; &nbsp; &nbsp; &nbsp; └── InShop  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Stanford_Online_Products  
    &nbsp; &nbsp; &nbsp; &nbsp; └── MSMT17  
    &nbsp; &nbsp; &nbsp; &nbsp; └── VeRi776

1. download teacher models
- Our teacher models are at https://pan.baidu.com/s/1X8urI8_bDfmdapSaNGYbtA?pwd=if2i or https://drive.google.com/drive/folders/1-S6r2nrcn6fQzBrnnEtLbivs4sZ028ZE?usp=drive_link, please download the checkpoints to `./download_teacher_ckpts`

2. download student models
- Our student models are at https://pan.baidu.com/s/1X8urI8_bDfmdapSaNGYbtA?pwd=if2i or https://drive.google.com/drive/folders/1-S6r2nrcn6fQzBrnnEtLbivs4sZ028ZE?usp=drive_link, please download the checkpoints to `./download_student_ckpts`

2. Path setting
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


3. Training 

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
  ```
 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our UGD method.
  python AIR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/UGD.yaml 
  ```

  - By default, the ImageNet pre-trained model will be used for training. The model will be automatically downloaded from the internet on the first run.  
  If you want to use a different pre-trained model, modify the `STUDENT_PRETRAIN_PATH` in the YAML configuration file.  


4. Evaluation

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
 ```

```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our UGD method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/UGD.yaml 
 ```

 - During inference, you can first navigate to `AIR_Distiller/utils/rank_cylib` and run the following commands to enable sorting with C language, which helps reduce inference time:  

```bash
python3 setup.py build_ext --inplace
rm -rf build
```

5. Testing with the exported student-only checkpoint

- We also provide exported student-only checkpoints for evaluation only.
- The checkpoints are available at https://pan.baidu.com/s/1g3Y0xbLaZL7TpFyvQAz3Ag?pwd=w8mb or https://drive.google.com/drive/folders/1JwLp
