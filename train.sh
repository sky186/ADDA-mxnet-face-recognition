#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_ENABLE_GPU_P2P=1
export MXNET_GPU_WORKER_NTHREADS=4
# MXNET_BACKWARD_DO_MIRROR=1
#--pretrained ./models/y1-arcface-emore/model \
#--pretrained-epoch "1" \


#DATA_DIR=../datasets/faces_glint #--dataset "emore"

#NETWORK=y1
#NETWORK=r50
NETWORK=sger50  #int the config have the name
# NETWORK=srmr50  
JOB=arcface
#LOSS=4

MODELDIR="/home/svt/mxnet_recognition/modify_model_output/gan_transfer_IR-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

CUDA_VISIBLE_DEVICES='2,3' python -u gan_transfer_train.py \
--network "$NETWORK" \
--models-root "$PREFIX" \
--per-batch-size 340 \
--pretrained ../model_low_finetune \
--pretrained-epoch "1" \
--loss arcface \
--lr 0.0001 \
--dataset emore \
2>&1 | tee "$LOGFILE"

###  ../modelres50-r50-arcface/model/r50-arcface-emore/modelfc7-0001.params
####./models/model-r50-am-lfw/model
###../modelres50-r50-arcface/model/r50-arcface-emore/modelfc7

