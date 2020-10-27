#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python main_rot.py --dataset multi --target clipart --num 3 --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=0 python main_old.py --dataset multi --source real --target clipart --num 3 --net resnet34 --save_check
