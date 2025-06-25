# 项目环境搭建指南

## 一、准备工作

### 1. 克隆项目

在 `/root/` 路径下执行：

```bash
git clone https://github.com/accc6515/SYZZ.git
mv SYZZ/ Project/
cd Project/
```

### 2. 安装 PyTorch（选择合适版本）

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. 安装依赖项

```bash
pip install -r requirements.txt
```

### 4. 安装 Fairseq

```bash
cd av_hubert/fairseq
pip install -e .
```

### 5. 安装 FFmpeg

```bash
sudo apt update
sudo apt install ffmpeg
```

### 6. 拷贝人脸检测与特征提取模块

```bash
cd /root/Project
cp -r modification/retinaface preprocessing/face-alignment/face_alignment/detection
cp modification/landmark_extract.py preprocessing/face-alignment/
```

------

## 二、模型下载

```bash
cd /root/Project
mkdir -p checkpoints/
cd checkpoints/
```

### 1. 下载 Audio-Visual Speech Representation 预训练模型

```bash
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt
```

### 2. 下载 RetinaFace 模型

下载[模型](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) ，放在 `checkpoints/Resnet50_Final.pth`

### 3. 下载语音识别模型

```bash
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
```

------

## 三、鲁棒性评估环境

```bash
conda create -n ptb python=3.7 -y
conda init bash
source ~/.bashrc
conda activate ptb

cd ~/Project/raodong/DeeperForensics-1.0-master/DeeperForensics-1.0-master/perturbation
pip install -r requirements.txt

conda deactivate
```

------

## 四、下载前端

下载[前端压缩包](https://github.com/accc6515/SYZZ/releases/download/%E5%89%8D%E7%AB%AF%E5%8E%8B%E7%BC%A9%E5%8C%85/deepfake.zip)并解压

## 五、运行方式

### 1. 启动前端

进入前端解压后对应文件夹

```bash
npm run serve
```

### 2. 启动后端

```bash
cd /root/Project
python webBack/app.py
```

> ⚠️ **注意**：首次运行“视频解码”模块时会自动安装 OpenAI Whisper 模型，请耐心等待。

------

