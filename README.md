## Code and Datasets from the paper "Self-supervised contrastive learning for volcanic unrest detection from InSAR data"

You can download the pretrained encoders [here](https://www.dropbox.com/s/qcieo92cdyqtjgp/models.zip?dl=0).

If you use this repo please consider citing our paper 

```
@ARTICLE{9517282,
  author={Bountos, Nikolaos Ioannis and Papoutsis, Ioannis and Michail, Dimitrios and Anantrasirichai, Nantheera},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Self-Supervised Contrastive Learning for Volcanic Unrest Detection}, 
  year={2021},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2021.3104506}}
  ```
  
  
  
  ### Loading Pre-Trained encoder example: ###
  
  ```
  backbone = torchvision.models.resnet50(pretrained=False)
  backbone.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), backbone.fc)
  backbone = torch.nn.parallel.DataParallel(backbone,device_ids=[0,1])
  backbone.load_state_dict(torch.load('ResNet50_Simclr_500_Epochs.pt'))
  backbone.module.fc = nn.Identity()
  backbone = backbone.module
  ```
