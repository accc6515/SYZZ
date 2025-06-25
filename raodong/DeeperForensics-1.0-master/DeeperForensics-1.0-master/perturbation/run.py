import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', required=True, help='扰动类型')
parser.add_argument('--level', required=True, help='扰动等级')
parser.add_argument('--vid_in', required=True, help='输入视频路径')
parser.add_argument('--vid_out', required=True, help='输出视频路径')
args = parser.parse_args()

distort_type = args.type
level = args.level
vid_in = args.vid_in
vid_out = args.vid_out

# 绝对路径到脚本
ADD_SCRIPT_PATH = "/root/Project/raodong/DeeperForensics-1.0-master/DeeperForensics-1.0-master/perturbation/add_distortion_to_video.py"

# ptb环境的python绝对路径，改成你实际路径
PTB_PYTHON = "/root/miniconda3/envs/ptb/bin/python"

# 构建命令行字符串，调用ptb环境的python
cmd = f'"{PTB_PYTHON}" "{ADD_SCRIPT_PATH}" --vid_in "{vid_in}" --vid_out "{vid_out}" --type {distort_type} --level {level}'
print(f'执行命令：{cmd}')
os.system(cmd)
print(f'完成 {distort_type} 扰动（Level {level}）处理。')
