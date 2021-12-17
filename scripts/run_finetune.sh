# DATA.FEATURE: specify which representation to use
# DATA.BATCH_SIZE: ViT-based backbone requires a smaller batchsize
# RUN_N_TIMES: ensure only run once in case duplicated submision 
# MODEL.TYPE: base or joint training
# OUTPUT_DIR: output dir of the final model and logs
# SOLVER.BASE_LR: learning rate for the experiment
# SOLVER.WEIGHT_DECAY: weight decay value for the experiment
# MODEL.KNN_LAMBDA: alpha in Eq 4


##############
# base
##############

python train.py \
    --config-file configs/finetune/cub.yaml \
    MODEL.TYPE "base" \
    DATA.BATCH_SIZE "384" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    SOLVER.BASE_LR "0.00375" \
    SOLVER.WEIGHT_DECAY "0.01" \
    OUTPUT_DIR <OUTPUT_PATH> 

# tune hyper-parameters
python tune.py \
    --train-type "finetune" \
    --config-file configs/finetune/cub.yaml \
    MODEL.TYPE "base" \
    DATA.BATCH_SIZE "384" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 


##############
# joint (update every epoch)
# use a different OUTPUT_DIR than above base
##############

python train.py \
    --config-file configs/finetune/cub.yaml \
    MODEL.TYPE "joint" \
    SOLVER.LOSS "knn_reg" \
    MODEL.KNN_LAMBDA "0.01" \
    DATA.BATCH_SIZE "384" \
    SOLVER.BASE_LR "0.00375" \
    SOLVER.WEIGHT_DECAY "0.01" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 

# tune hyper-parameters
python tune.py \
    --train-type "finetune" \
    --config-file configs/finetune/cub.yaml \
    MODEL.TYPE "joint" \
    SOLVER.LOSS "knn_reg" \
    MODEL.KNN_LAMBDA "0.01" \
    DATA.BATCH_SIZE "384" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 
