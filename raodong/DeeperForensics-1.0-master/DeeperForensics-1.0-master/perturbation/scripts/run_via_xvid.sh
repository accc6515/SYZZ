#!/usr/bin/env bash

set -x

python add_distortion_to_video.py \
    --vid_in data/00109.mp4 \
    --vid_out results/output.mp4 \
    --type GNC \
    --level 5 \
    --meta_path metas/meta.txt \
    --via_xvid
