import cv2
import numpy as np
import skvideo
import torch
from python_speech_features import logfbank

skvideo.setFFmpegPath('/usr/bin')
import skvideo.io
from tqdm import tqdm
import os
import os.path as osp
import sys
from base64 import b64encode
import tempfile
from argparse import Namespace
import utils as avhubert_utils
from av_hubert.fairseq.fairseq import checkpoint_utils, options, tasks
import av_hubert.fairseq.fairseq.utils as fairseq_utils
from av_hubert.fairseq.fairseq.dataclass.configs import GenerationConfig
from glob import glob
from scipy.io import wavfile
import shutil
#from av_hubert.avhubert import utils as avhubert_utils
import soundfile as sf
import json
import torch.nn.functional as F
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from similarityPlot import plot_enhanced_similarity

def calc_cos_dist(feat1, feat2, vshift=15):
    feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)
    if len(feat1) != len(feat2):
        sample = np.linspace(0, len(feat1) - 1, len(feat2), dtype=int)
        feat1 = feat1[sample.tolist()]
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(torch.nn.functional.cosine_similarity(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]).cpu().numpy())
    return np.asarray(dists)


def extract_visual_feature(video_path, max_length):
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)
    ])

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    success, frame = video.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1)
        frame = torch.FloatTensor(frame)
        frames.append(frame)
        success, frame = video.read()
    video.release()

    if len(frames) > fps * max_length:
        frames = frames[:int(fps * max_length)]

    if len(frames) == 0:
        print(f"Error: No valid frames extracted from {video_path}")
        return None

    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.permute(3, 0, 1, 2).unsqueeze(dim=0).cuda(0)

    with torch.no_grad():
        feature, _ = model.extract_finetune(
            source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None
        )
        feature = feature.squeeze(dim=0)
    return feature


def stacker(feats, stack_order):
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats


def extract_audio_feature(audio_path):
    sample_rate, wav_data = wavfile.read(audio_path)
    if sample_rate != 16000 or len(wav_data.shape) != 1:
        print(f"Error: Invalid audio file {audio_path}")
        return None

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
    audio_feats = stacker(audio_feats, 4)
    audio_feats = torch.FloatTensor(audio_feats).cuda(0)

    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    audio_feats = audio_feats.transpose(0, 1).unsqueeze(dim=0)

    with torch.no_grad():
        feature, _, = model.extract_finetune(source={'video': None, 'audio': audio_feats}, padding_mask=None, output_layer=None)
        feature = feature.squeeze(dim=0)
    return feature


def evaluate_single_video(video_path, audio_path, max_length=50, threshold=0.3):

    visual_feature = extract_visual_feature(video_path, max_length)
    audio_feature = extract_audio_feature(audio_path)

    min_length = min(visual_feature.shape[0], audio_feature.shape[0])
    visual_feature = visual_feature[:min_length]
    audio_feature = audio_feature[:min_length]

    dist = calc_cos_dist(visual_feature.cpu(), audio_feature.cpu())
    similarity_score = dist.mean(axis=0)
    similarity_score = similarity_score.max()  # 取平均值
  # 或 similarity_score.mean()
    print(f"Similarity Score: {similarity_score:.4f}")
    if similarity_score >= threshold:
        print("real")
    else:
        print("fake")
        
    # 折线图
    best_offset_idx = dist.mean(axis=0).argmax()
    vshift = 15
    best_frame_offset = best_offset_idx - vshift
    print(f"best_offset：{best_frame_offset} ")
    plot_enhanced_similarity(dist, best_frame_offset, threshold)
    # 折线图
    
    best_dist = dist[:, best_offset_idx]
    cnt_low_threshold = (best_dist < threshold).sum()
    per_low_threshold = cnt_low_threshold / len(best_dist)
            
    print(f"per_low_threshold: {per_low_threshold}")

    return similarity_score, cnt_low_threshold


if __name__ == '__main__':
    print("Starting evaluation...")

    parser = argparse.ArgumentParser(description='Detect fake videos')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/large_vox_iter5.pt', help='Path to model checkpoint')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of video for processing')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for classifying fake videos')

    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    fairseq_utils.import_user_module(Namespace(user_dir="av_hubert/avhubert"))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]

    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")

    model.cuda()
    model.eval()

    # Run single video evaluation
    evaluate_single_video(args.video_path, args.audio_path, args.max_length, args.threshold)
