#!/bin/bash
cd /home/husrcf/Code/PPC
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
exec /home/husrcf/anaconda3/envs/AIAA/bin/python -u \
    scripts/training/train_esm_site.py \
    --config configs/train_contact_site_esm_predstruct_noctx_reg_b64_pf1_mmseq30.yaml
