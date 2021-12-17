# Rethinking Nearest Neighbors for Visual Classification


## Environment settings

Check out `scripts/env_setup.sh`



## Setup data

Download the following fine-grained datasets and ImageNet.

- [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)
- [CUB200 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

In current version, you need to modify each data file under `knn/data/finetune/*.py`



## Experiments

Some key configs:

- DATA.FEATURE: specify which representation to use
- DATA.BATCH_SIZE: ViT-based backbone requires a smaller batchsize
- RUN_N_TIMES: ensure only run once in case duplicated submision 
- MODEL.TYPE: base or joint training
- OUTPUT_DIR: output dir of the final model and logs
- SOLVER.BASE_LR: learning rate for the experiment
- SOLVER.WEIGHT_DECAY: weight decay value for the experiment
- MODEL.KNN_LAMBDA: alpha in Eq 4

### Linear evaluation

See `script/run_linear.sh` and `script/run_newt.sh`

### End-to-end finetuning

See `script/run_finetune.sh` 

## License

This repo are released under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/KMnP/nn-revisit/blob/master/LICENSE) for additional details.

## Acknowledgement

We thank the researchers who propose NEWT for providing the features for the datasets. 



