##############
# base
##############

python train.py \
    --config-file configs/linear/cub.yaml \
    MODEL.TYPE "base" \
    DATA.BATCH_SIZE "2048" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    SOLVER.BASE_LR "0.025" \
    SOLVER.WEIGHT_DECAY "0.01" \
    OUTPUT_DIR <OUTPUT_PATH> 

# automatically tune hyper-parameters
python tune.py \
    --train-type "linear" \
    --config-file configs/linear/cub.yaml \
    MODEL.TYPE "base" \
    DATA.BATCH_SIZE "2048" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 


##############
# joint
##############
# possibly a different OUTPUT_DIR than above base


python train.py \
    --config-file configs/linear/cub.yaml \
    MODEL.TYPE "joint" \
    SOLVER.LOSS "knn_reg" \
    MODEL.KNN_LAMBDA "0.01" \
    DATA.BATCH_SIZE "2048" \
    SOLVER.BASE_LR "0.025" \
    SOLVER.WEIGHT_DECAY "0.01" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 

# automatically tune hyper-parameters
python tune.py \
    --train-type "linear" \
    --config-file configs/linear/cub.yaml \
    MODEL.TYPE "joint" \
    SOLVER.LOSS "knn_reg" \
    MODEL.KNN_LAMBDA "0.01" \
    DATA.BATCH_SIZE "2048" \
    DATA.FEATURE "imagenet_supervised" \
    RUN_N_TIMES "1" \
    OUTPUT_DIR <OUTPUT_PATH> 
