#!/bin/sh
#$ -N ae_classifier_2
#$ -cwd
#$ -j y
#$ -t 1-1
#$ -q gpu_long
#$ -l gpu=1
#$ -l h_rt=14:00:00
#$ -l virtual_free=8G
#$ -o aeclassifier_3.out

workon env_SigNet

python train_vae_classifier.py --config_file configs/vae_classifier/vc_config_2.yaml
