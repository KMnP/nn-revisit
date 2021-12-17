# base
python train.py \
    --config-file configs/linear/newt.yaml \
    MODEL.TYPE "linear_base" DATA.NAME "ml_bio_raptor_utility_pole"\
    DATA.FEATURE "inat2021_supervised" \
    OUTPUT_DIR <OUTPUT_PATH> 


# joint
python train.py \
    --config-file configs/linear/newt.yaml \
    MODEL.TYPE "linear_joint" DATA.NAME "ml_bio_raptor_utility_pole"\
    SOLVER.LOSS "knn_reg" \
    MODEL.KNN_LAMBDA "0.01" \
    DATA.FEATURE "inat2021_supervised" \
    OUTPUT_DIR <OUTPUT_PATH> 

