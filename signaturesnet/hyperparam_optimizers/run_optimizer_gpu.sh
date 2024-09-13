#!/bin/sh
#$ -N ae_optimizer
#$ -cwd
#$ -j y
#$ -t 1-1
#$ -q gpu_long
#$ -l gpu=1
#$ -l h_rt=100:00:00
#$ -l virtual_free=8G
#$ -o Cluster/aeclassifier_gpu.out

workon env_SigNet

python optimize_ae_classifier.py
