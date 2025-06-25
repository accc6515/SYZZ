import os
import cv2
import librosa
import argparse

def generate_tsv(video_dir, audio_dir, output_tsv, output_wrd):
    """
    生成包含视频和音频元数据的TSV文件
    
    参数:
        video_dir: 视频文件目录
        audio_dir: 音频文件目录
        output_tsv: 输出TSV文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    
    with open(output_tsv, 'w') as f:
        # 写入根目录（TSV文件首行要求）
        f.write(f"{video_dir}\n")
        
        # 获取目录中所有MP4文件
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        for video_file in video_files:
            # 解析视频ID（假设文件名格式为"videos{ID}.mp4"）
            vid = os.path.splitext(video_file)[0].replace('videos', '')
            
            # 构建文件路径
            video_path = os.path.join(video_dir, video_file)
            audio_path = os.path.join(audio_dir, f"{vid}.wav")
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"警告：未找到匹配的音频文件 {audio_path}")
                continue
            
            try:
                # 获取视频帧数
                cap = cv2.VideoCapture(video_path)
                video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # 获取音频采样点数
                audio, sr = librosa.load(audio_path, sr=None, mono=False)
                audio_samples = audio.shape[-1]  # 兼容单声道/立体声
                
                # 写入TSV行
                f.write(f"{vid}\t{video_path}\t{audio_path}\t{video_frames}\t{video_frames}\n")
                print(f"已处理: ID={vid} | 视频帧数={video_frames} | 音频采样点={audio_samples}")
                
            except Exception as e:
                print(f"处理 {video_path} 时出错: {str(e)}")
    with open(output_wrd, 'w') as f:
        f.write("dummy\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成视频音频元数据TSV文件')
    parser.add_argument('--video_dir', required=True, help='视频文件目录路径')
    parser.add_argument('--audio_dir', required=True, help='音频文件目录路径')
    parser.add_argument('--output_tsv', required=True, help='输出TSV文件路径')
    parser.add_argument('--output_wrd', required=True, help='输出WRD文件路径')
    
    args = parser.parse_args()
    
    generate_tsv(
        video_dir=os.path.abspath(args.video_dir),
        audio_dir=os.path.abspath(args.audio_dir),
        output_tsv=os.path.abspath(args.output_tsv),
        output_wrd=os.path.abspath(args.output_wrd)
    )