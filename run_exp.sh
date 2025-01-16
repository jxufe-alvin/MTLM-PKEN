#!/bin/bash
# 脚本名称: run_python.sh


# 运行Python程序
# 'self_harm', 'depression'
python train_mm_MTLM_PKEN.py --dropout_transformer 0.1 --dim_feedforward 256 --num_layers 2 --nhead 1 --dataset 'mmh4' --bs 6 --lr 0.000015 --dropout_ 0.1

# 'anxiety', 'ptsd'
python train_mm_MTLM_PKEN.py --dropout_transformer 0.1 --dim_feedforward 512 --num_layers 2 --nhead 2 --dataset 'mmh7' --bs 6 --lr 0.000015 --dropout_ 0.1

# 'self_harm', 'bipolar'
python train_mm_MTLM_PKEN.py --dropout_transformer 0.1 --dim_feedforward 128 --num_layers 2 --nhead 1 --dataset 'mmh8' --bs 6 --lr 0.000015 --dropout_ 0.1

# mmhX表示任务组合，具体请参考代码：dataloader.py