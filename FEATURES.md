# Representations used in the project

The table below presents the specifications of different the representation we use in this project. The last column refers to the name we use for the specific feature in config.

| Backbone  | Obj.        | Data          | `DATA.FEATURE`             |
| --------- | ----------- | ------------- | -------------------------- |
| ResNet-50 | Supervised  | ImageNet      | `imagenet_supervised`      |
| ResNet-50 | Supervised  | iNat2021      | `inat2021_supervised`      |
| ResNet-50 | Supervised  | iNat2021-mini | `inat2021_mini_supervised` |
| ResNet-50 | MoCo-v2     | ImageNet      | `imagenet_moco_v2`         |
| ResNet-50 | MoCo-v3     | ImageNet      | `mocov3_rn50`              |
| ResNet-50 | SwAV        | ImageNet      | `imagenet_swav`            |
| ResNet-50 | DINO        | ImageNet      | `dino_resnet50`            |
| ResNet-50 | BarlowTwins | ImageNet      | `imagenet_barlowtwins`     |
| ViT-B/16  | Supervised  | ImageNet      | `sup_vitb16_224`           |
| ViT-L/16  | Supervised  | ImageNet      | `sup_vitl16_224`           |
| ViT-S/16  | DINO        | ImageNet      | `dino_vits16`              |
| ViT-S/16  | MoCo-v3     | ImageNet      | `mocov3_vits`              |
| ViT-B/16  | DINO        | ImageNet      | `dino_vitb16`              |
| ViT-B/16  | MoCo-v3     | ImageNet      | `mocov3_vitb`              |
| Swin-T    | Supervised  | ImageNet      | `swint_imagenet`           |
| Swin-S    | Supervised  | ImageNet      | `swins_imagenet`           |
| Swin-B    | Supervised  | ImageNet      | `swinb_imagenet_224`       |
| Swin-T    | MoBY        | ImageNet      | `swint_imagenet_ssl`       |

