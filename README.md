## SRFlow-DA: Super-Resolution Using Normalizing Flow with Deep Convolutional Block

[NTIRE 2021](https://data.vision.ee.ethz.ch/cvl/ntire21/) Learning the Super-Resolution Space [Challenge](https://github.com/andreas128/NTIRE21_Learning_SR_Space).

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Jo_SRFlow-DA_Super-Resolution_Using_Normalizing_Flow_With_Deep_Convolutional_Block_CVPRW_2021_paper.html)

- Challenge results of our SRFlow-DA model.

|Upscale|LR-PSNR|LPIPS|Diversity|
|------|---|---|---|
|X4|50.70 (1st)|0.121 (3rd)|23.091 (4th)|
|X8|50.86 (1st)|0.266 (3rd)|23.320 (4th)|

## Dependency
- Python 3.6 (anaconda, miniconda or pyenv is recommended)
- PyTorch 1.7
- Other dependencies in `requirements.txt`, 
   ```
   pip install -r requirements.txt
   ```
- Because the file (`requirements.txt`) contains the information of [abstract dependencies](https://caremad.io/posts/2013/07/setup-vs-requirement/), you can install other compatible versions referring to the file when you have a problem with the above command.
- Most of the code from the original SRFlow [repository](https://github.com/andreas128/SRFlow).


## First things to do
1. Clone this repo.
```
git clone https://github.com/yhjo09/SRFlow-DA
cd SRFlow-DA
```

2. Download datasets and baseline models.
```
sh ./prepare.sh
```

## Test
1. [Download](https://yonsei-my.sharepoint.com/:u:/g/personal/yh_jo_o365_yonsei_ac_kr/EcJEWvhzNipLiPV8_Yfy-pYBa5OMdIaUd4LWeefr-7LyaA?e=PvlKwM) SRFlow-DA models and unzip it.
```
unzip ./experiments.zip
```

2. Run. 
```
cd ./code
python test.py ./confs/SRFlow-DA_DF2K_4X.yml        # SRFlow-DA 4X SR
python test.py ./confs/SRFlow-DA_DF2K_8X.yml        # SRFlow-DA 8X SR
python test.py ./confs/SRFlow-DA-R_DF2K_4X.yml      # SRFlow-DA-R 4X SR
python test.py ./confs/SRFlow-DA-R_DF2K_8X.yml      # SRFlow-DA-R 8X SR
python test.py ./confs/SRFlow-DA-S_DF2K_4X.yml      # SRFlow-DA-S 4X SR
python test.py ./confs/SRFlow-DA-S_DF2K_8X.yml      # SRFlow-DA-S 8X SR
python test.py ./confs/SRFlow-DA-D_DF2K_4X.yml      # SRFlow-DA-D 4X SR
python test.py ./confs/SRFlow-DA-D_DF2K_8X.yml      # SRFlow-DA-D 8X SR
```
- If your GPU memory lacks, please try with prefix `CUDA_VISIBLE_DEVICES=-1` (CPU only).
- You may check `dataroot_LR` of the configuration file for the test.

3. Check your results in `./results`.


## Train
1. You may have to modify some variables (e.g. directories) in a config file `./confs/*.yml`.

2. Run.
```
cd ./code
python train.py -opt ./confs/SRFlow-DA_DF2K_4X.yml        # SRFlow-DA 4X SR
python train.py -opt ./confs/SRFlow-DA_DF2K_8X.yml        # SRFlow-DA 8X SR
python train.py -opt ./confs/SRFlow-DA-R_DF2K_4X.yml      # SRFlow-DA-R 4X SR
python train.py -opt ./confs/SRFlow-DA-R_DF2K_8X.yml      # SRFlow-DA-R 8X SR
python train.py -opt ./confs/SRFlow-DA-S_DF2K_4X.yml      # SRFlow-DA-S 4X SR
python train.py -opt ./confs/SRFlow-DA-S_DF2K_8X.yml      # SRFlow-DA-S 8X SR
python train.py -opt ./confs/SRFlow-DA-D_DF2K_4X.yml      # SRFlow-DA-D 4X SR
python train.py -opt ./confs/SRFlow-DA-D_DF2K_8X.yml      # SRFlow-DA-D 8X SR
```
- If your GPU memory lacks, please try with lower batch size or patch size.

3. Training logs, model parameters, and validation result images will be stored in `./experiments`.


## BibTeX
```
@InProceedings{jo2021srflowda,
   author = {Jo, Younghyun and Yang, Sejong and Kim, Seon Joo},
   title = {SRFlow-DA: Super-Resolution Using Normalizing Flow with Deep Convolutional Block},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
   month = {June},
   year = {2021}
}
```
