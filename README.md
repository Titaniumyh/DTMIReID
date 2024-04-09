![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)


## Requirements

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Datasets

```bash
mkdir data
```

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), 
Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── images ..
├── MSMT17
│   └── images ..
├── dukemtmcreid
│   └── images ..
├── Occluded_Duke
    └── images ..

```

### Prepare DeiT or ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth), [ViT-Small](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth), [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth)

## Training

We utilize 1  GPU for training.

```bash
python train.py --config_file configs/dukemtmc.yml
```

## Evaluation

```bash
python test.py --config_file configs/dukemtmc.yml TEST.WEIGHT '../logs/Duke_Deformable/transformer_120.pth'
```




