# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
import uuid
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import json

import whisper
import re
import numpy as np
import base64

from models import db, DetectionRecord, VideoInfo
# ============ 环境设置 ============
PYTHON_BASE_ENV = "/root/miniconda3/bin/python"
PYTHON_PTB_ENV = "/root/miniconda3/envs/ptb/bin/python"

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///records.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all() # 创建数据库表

TEMP_DIR = "/root/Project/webBack/videos/temp" # 请确保此路径有效且可写
os.makedirs(TEMP_DIR, exist_ok=True)

def run_script(command_list, operation_name="脚本"): # 辅助函数，用于执行外部脚本
    print(f"执行 {operation_name}: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"{operation_name} 标准输出:\n{process.stdout}")
        if process.stderr:
            print(f"{operation_name} 标准错误:\n{process.stderr}")
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        error_message = f"{operation_name} 执行失败，返回码 {e.returncode}.\n"
        error_message += f"命令: {' '.join(e.cmd)}\n"
        if e.stdout: error_message += f"标准输出:\n{e.stdout}\n"
        if e.stderr: error_message += f"标准错误:\n{e.stderr}\n"
        print(error_message)
        raise
    except FileNotFoundError as e:
        print(f"错误: {operation_name} 的脚本或可执行文件未找到. 详情: {e}")
        raise

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "未提供视频文件"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "未选择文件或文件名为空"}), 400

    original_filename = video_file.filename
    current_pos = video_file.tell()
    video_file.seek(0, os.SEEK_END)
    filesize = video_file.tell()
    video_file.seek(current_pos)

    process_id = str(uuid.uuid4())
    temp_work_dir = os.path.join(TEMP_DIR, process_id)
    os.makedirs(temp_work_dir, exist_ok=True)
    video_filename_for_processing = f"video_{process_id}.mp4"
    uploaded_video_path = os.path.join(temp_work_dir, video_filename_for_processing)

    try:
        video_file.save(uploaded_video_path)
        print(f"视频已保存至: {uploaded_video_path}")

        video_info_record = VideoInfo(
            original_filename=original_filename,
            processed_filename=video_filename_for_processing,
            filesize_bytes=filesize,
            storage_path_identifier=uploaded_video_path,
        )
        db.session.add(video_info_record)
        db.session.commit()
        print(f"已为 {original_filename} 创建 VideoInfo 记录，ID 为 {video_info_record.id}")

        file_list_path = os.path.join(temp_work_dir, "file_list.txt")
        with open(file_list_path, "w", encoding='utf-8') as f: 
            f.write(video_filename_for_processing + "\n")

        landmark_dir = os.path.join(temp_work_dir, "landmark")
        output_video_dir = os.path.join(temp_work_dir, "outputvideo")

        landmark_script = "preprocessing/face-alignment/landmark_extract.py"
        align_script = "preprocessing/align_mouth.py"
        evaluate_script = "evaluate88.py"

        cmd1 = [PYTHON_BASE_ENV, landmark_script, "--video_root", temp_work_dir, "--file_list", file_list_path, "--out_dir", landmark_dir]
        cmd2 = [PYTHON_BASE_ENV, align_script, "--video_root", temp_work_dir, "--file_list", file_list_path, "--landmark_dir", landmark_dir, "--out_dir", output_video_dir]
        
        run_script(cmd1, "人脸关键点提取")
        run_script(cmd2, "嘴部对齐")

        processed_video_path = os.path.join(output_video_dir, video_filename_for_processing)
        output_audio_path = os.path.join(output_video_dir, f"video_{process_id}.wav")

        if not os.path.exists(processed_video_path):
            raise FileNotFoundError(f"对齐后的视频未找到: {processed_video_path}。嘴部对齐脚本可能出错。")
        if not os.path.exists(output_audio_path):
            print(f"警告: 音频文件 {output_audio_path} 未找到。评估脚本可能需要它或能处理其缺失的情况。")

        cmd3 = [PYTHON_BASE_ENV, evaluate_script, "--video_path", processed_video_path, "--audio_path", output_audio_path]
        eval_stdout, _ = run_script(cmd3, "评估脚本")
        
        similarity_score = None
        prediction = "Unknown" # 默认预测结果
        output_lines = eval_stdout.strip().split("\n")
        for line in output_lines:
            print(f"评估脚本输出行: {line}")
            if "Similarity Score:" in line: # 假设脚本输出的关键词是英文
                try:
                    score_str = line.split(":", 1)[1].strip()
                    similarity_score = float(score_str)
                except (ValueError, IndexError) as e:
                    print(f"无法从行 '{line}' 解析分数: {e}")
            # 提取 per_low_threshold
            elif "per_low_threshold:" in line:
                try:
                    per_str = line.split(":", 1)[1].strip()
                    per_low_threshold = float(per_str)
                except (ValueError, IndexError) as e:
                    print(f"无法从行 '{line}' 解析per_low_threshold: {e}")
            clean_line = line.strip().lower()
            if clean_line == "real": prediction = "real" # 内部值保持英文
            elif clean_line == "fake": prediction = "fake"
        if per_low_threshold > 0.25:
            prediction = "fake"

        detection_record = DetectionRecord(
            video_name=original_filename,
            similarity_score=similarity_score,
            prediction=prediction
        )
        db.session.add(detection_record)
        db.session.commit()
        print(f"已为 {original_filename} 创建 DetectionRecord，预测结果: {prediction}")
        
        with open("/root/Project/enhanced_similarity_plot.png", 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return jsonify({
            "score": similarity_score,
            "prediction": prediction, # 内部值
            "original_filename": original_filename,
            "filesize": filesize,
            "upload_timestamp": video_info_record.upload_timestamp.isoformat() + "Z",
            "image_base64": encoded_string
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"处理脚本执行失败，请检查服务器日志。失败操作: {e.cmd[1] if len(e.cmd)>1 else '未知脚本'}" }), 500
    except FileNotFoundError as e:
        print(f"文件或脚本未找到: {e}")
        return jsonify({"error": f"所需文件或脚本未找到: {e.filename}"}), 500
    except Exception as e:
        db.session.rollback()
        print(f"上传处理过程中发生意外错误: {str(e)}")
        # import traceback; print(traceback.format_exc()) # 调试时可取消注释以获取完整堆栈
        return jsonify({"error": "发生意外的服务器错误，请稍后再试。"}), 500
    finally:
        if os.path.exists(temp_work_dir):
            print(f"清理临时目录: {temp_work_dir}")
            shutil.rmtree(temp_work_dir, ignore_errors=True)

@app.route("/decode", methods=["POST"])
def decode():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    process_id = str(uuid.uuid4())
    temp_work_dir = os.path.join(TEMP_DIR, process_id)
    os.makedirs(temp_work_dir, exist_ok=True)

    video_filename = f"video_{process_id}.mp4"
    video_path = os.path.join(temp_work_dir, video_filename)
    video_file.save(video_path)

    file_list_path = os.path.join(temp_work_dir, "file_list.txt")
    with open(file_list_path, "w") as f:
        f.write(video_filename + "\n")

    landmark_dir = os.path.join(temp_work_dir, "landmark")
    output_video_dir = os.path.join(temp_work_dir, "outputvideo")

    cmd1 = [
        sys.executable, "preprocessing/face-alignment/landmark_extract.py",
        "--video_root", temp_work_dir,
        "--file_list", file_list_path,
        "--out_dir", landmark_dir
    ]
    cmd2 = [
        sys.executable, "preprocessing/align_mouth.py",
        "--video_root", temp_work_dir,
        "--file_list", file_list_path,
        "--landmark_dir", landmark_dir,
        "--out_dir", output_video_dir
    ]
    
    output_video_path = os.path.join(output_video_dir, video_filename)
    output_audio_path = os.path.join(output_video_dir, f"video_{process_id}.wav")
    
    cmd4 = [
        sys.executable, "/root/Project/av_hubert/avhubert/createTSV.py",
        "--video_dir", output_video_dir,
        "--audio_dir", output_video_dir,
        "--output_tsv", "/root/Project/av_hubert/avhubert/test_data/test.tsv",
        "--output_wrd", "/root/Project/av_hubert/avhubert/test_data/test.wrd",
    ]
    
    cmd5 = [
        sys.executable, "-B",
        "/root/Project/av_hubert/avhubert/infer_s2s.py",
        "--config-dir", "/root/Project/av_hubert/avhubert/conf/",
        "--config-name", "s2s_decode.yaml",
        "dataset.gen_subset=test",
        "common_eval.path=/root/Project/checkpoints/self_large_vox_433h.pt",
        f"common_eval.results_path=/root/Project/av_hubert/avhubert/results/{process_id}video",  # 添加进程ID避免路径冲突
        "override.modalities=['video']",
        "+override.data=/root/Project/av_hubert/avhubert/test_data",  # 创建专用数据目录
        "+override.label_dir=/root/Project/av_hubert/avhubert/test_data",  # 创建专用标签目录
        "common.user_dir=/root/Project/av_hubert/avhubert"
    ]
    
    # cmd6 = [
    #     sys.executable, "-B",
    #     "/root/Project/av_hubert/avhubert/infer_s2s.py",
    #     "--config-dir", "/root/Project/av_hubert/avhubert/conf/",
    #     "--config-name", "s2s_decode.yaml",
    #     "dataset.gen_subset=test",
    #     "common_eval.path=/root/autodl-tmp/models/large_noise_pt_noise_ft_433h.pt",
    #     f"common_eval.results_path=/root/Project/av_hubert/avhubert/results/{process_id}audio",  # 添加进程ID避免路径冲突
    #     "override.modalities=['audio']",
    #     "+override.data=/root/Project/av_hubert/avhubert/test_data",  # 创建专用数据目录
    #     "+override.label_dir=/root/Project/av_hubert/avhubert/test_data",  # 创建专用标签目录
    #     "common.user_dir=/root/Project/av_hubert/avhubert"
    # ]

    try:
        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)
        subprocess.run(cmd4, check=True)
        subprocess.run(cmd5, check=True)
        # subprocess.run(cmd6, check=True)

        # result_fn_audio = f"/root/Project/av_hubert/avhubert/results/{process_id}audio/hypo.json"
        # with open(result_fn_audio, 'r') as f:
        #     result_audio_dict = json.load(f)
        #     audio_hypo = result_audio_dict['hypo']
        
        model_path = "/root/autodl-tmp/models/"
        model = whisper.load_model("medium", download_root=model_path)
        # model = whisper.load_model("tiny")
        result_audio = model.transcribe(output_audio_path, language="en")
        audio_hypo = result_audio["text"]
        
        # 删除所有标点符号（保留字母、数字和空格）
        audio_hypo = re.sub(r"[^\w\s']", '', audio_hypo)
        # 将所有字母转换为小写
        audio_hypo = audio_hypo.lower()

        result_fn_video = f"/root/Project/av_hubert/avhubert/results/{process_id}video/hypo.json"
        with open(result_fn_video, 'r') as f:
            result_video_dict = json.load(f)
            video_hypo = result_video_dict['hypo']
            
        # 读取音频文件
        with open(output_audio_path, 'rb') as f:
            audio_bytes = f.read()
            
        # 截取
        cmd7 = [
            sys.executable, "/root/Project/split_video.py",
            "--video_path", video_path,
            "--save_path", "./combined_row.jpg",
            ]
        subprocess.run(cmd7, check=True)
        
        with open("/root/Project/combined_row.jpg", 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # 截取

        return jsonify({
            "audio_data": audio_bytes.hex(),
            "audio_hypo": audio_hypo,
            "video_hypo": video_hypo,
            "image_base64": encoded_string
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Processing failed: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_work_dir, ignore_errors=True)

@app.route("/records", methods=["GET"])
def get_records():
    try:
        records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()
        return jsonify([
            {
                "id": r.id,
                "video_name": r.video_name,
                "score": r.similarity_score,
                "prediction": r.prediction, # 内部值
                "timestamp": r.timestamp.isoformat() + "Z" if r.timestamp else None
            } for r in records
        ])
    except Exception as e:
        print(f"获取检测记录时出错: {str(e)}")
        return jsonify({"error": "无法获取检测记录。"}), 500

@app.route("/records/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    record = db.session.get(DetectionRecord, record_id)
    if record:
        try:
            db.session.delete(record)
            db.session.commit()
            return jsonify({"message": "记录删除成功"}), 200 # 中文消息
        except Exception as e:
            db.session.rollback()
            print(f"删除记录 {record_id} 时出错: {str(e)}")
            return jsonify({"error": f"删除记录失败: {str(e)}"}), 500
    else:
        return jsonify({"error": "未找到记录"}), 404 # 中文消息

@app.route("/videos", methods=["GET"])
def get_video_infos():
    try:
        video_infos = VideoInfo.query.order_by(VideoInfo.upload_timestamp.desc()).all()
        return jsonify([
            {
                "id": vi.id,
                "original_filename": vi.original_filename,
                "processed_filename": vi.processed_filename,
                "filesize_bytes": vi.filesize_bytes,
                "storage_path_identifier": vi.storage_path_identifier,
                "upload_timestamp": vi.upload_timestamp.isoformat() + "Z"
            } for vi in video_infos
        ])
    except Exception as e:
        print(f"获取视频信息时出错: {str(e)}")
        return jsonify({"error": "无法获取视频信息。"}), 500

from flask import Response
import mimetypes

@app.route('/api/perturb', methods=['POST'])
def perturb_video():
    if 'video' not in request.files:
        return jsonify({'error': '未上传视频'}), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({'error': '文件名为空'}), 400

    distortion_type = request.form.get('distortion_type')
    distortion_level = request.form.get('distortion_level')
    if not distortion_type or not distortion_level:
        return jsonify({'error': '扰动类型或等级未选择'}), 400

    session_id = str(uuid.uuid4())
    temp_work_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(temp_work_dir, exist_ok=True)

    try:
        original_filename = video_file.filename
        uploaded_path = os.path.join(temp_work_dir, f"input_{session_id}.mp4")
        perturbed_path = os.path.join(temp_work_dir, f"perturbed_{session_id}.mp4")

        video_file.save(uploaded_path)

        # 1. 添加扰动
        ptb_script = "/root/Project/raodong/DeeperForensics-1.0-master/DeeperForensics-1.0-master/perturbation/run.py"
        cmd_ptb = [
            PYTHON_PTB_ENV, ptb_script,
            "--type", distortion_type,
            "--level", str(distortion_level),
            "--vid_in", uploaded_path,
            "--vid_out", perturbed_path
        ]
        run_script(cmd_ptb, operation_name="扰动处理")

        if not os.path.exists(perturbed_path):
            return jsonify({'error': '扰动失败，输出视频不存在'}), 500

        # 2. 合成音频到扰动后视频
        final_video_path = os.path.join(temp_work_dir, f"final_{session_id}.mp4")
        add_audio_to_video(perturbed_path, uploaded_path, final_video_path)

        # 3. 处理视频并获取检测结果
        video_filename = os.path.basename(final_video_path)
        file_list_path = os.path.join(temp_work_dir, "file_list.txt")
        with open(file_list_path, "w", encoding='utf-8') as f:
            f.write(video_filename + "\n")

        landmark_dir = os.path.join(temp_work_dir, "landmark")
        output_dir = os.path.join(temp_work_dir, "outputvideo")

        # 3.1 提取关键点
        run_script([
            PYTHON_BASE_ENV, "preprocessing/face-alignment/landmark_extract.py",
            "--video_root", temp_work_dir,
            "--file_list", file_list_path,
            "--out_dir", landmark_dir
        ], "人脸关键点提取")

        # 3.2 嘴部对齐
        run_script([
            PYTHON_BASE_ENV, "preprocessing/align_mouth.py",
            "--video_root", temp_work_dir,
            "--file_list", file_list_path,
            "--landmark_dir", landmark_dir,
            "--out_dir", output_dir
        ], "嘴部对齐")

        # 3.3 评估检测
        processed_video_path = os.path.join(output_dir, video_filename)
        processed_audio_path = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}.wav")
        
        eval_stdout, _ = run_script([
            PYTHON_BASE_ENV, "evaluate88.py",
            "--video_path", processed_video_path,
            "--audio_path", processed_audio_path
        ], "评估脚本")

        # 解析评估结果
        similarity_score = None
        prediction = "Unknown"
        for line in eval_stdout.strip().splitlines():
            if "Similarity Score:" in line:
                try:
                    similarity_score = float(line.split(":")[1].strip())
                except:
                    pass
            # 提取 per_low_threshold
            elif "per_low_threshold:" in line:
                try:
                    per_str = line.split(":", 1)[1].strip()
                    per_low_threshold = float(per_str)
                except (ValueError, IndexError) as e:
                    print(f"无法从行 '{line}' 解析per_low_threshold: {e}")
            if line.strip().lower() == "real":
                prediction = "real"
            elif line.strip().lower() == "fake":
                prediction = "fake"
        if per_low_threshold > 0.25:
            prediction = "fake"

        # 保存检测记录到数据库    
        detection_record = DetectionRecord(
            video_name=original_filename + f" (perturbed type={distortion_type}, level={distortion_level})",
            similarity_score=similarity_score,
            prediction=prediction )
        db.session.add(detection_record)
        db.session.commit()

        # 关键修改：确保视频编码正确
        playable_video_path = os.path.join(temp_work_dir, f"playable_{session_id}.mp4")
        cmd_ffmpeg = [
            "ffmpeg", "-y",
            "-i", final_video_path,  # 最终合成的视频
            "-c:v", "libx264",      # 使用广泛支持的编码
            "-pix_fmt", "yuv420p",  # 确保像素格式兼容
            "-profile:v", "baseline", # 最大兼容性
            "-movflags", "+faststart", # 支持流式播放
            playable_video_path
        ]
        run_script(cmd_ffmpeg, "视频转码")

        # 4. 读取最终视频文件
        with open(playable_video_path, 'rb') as f:
            video_bytes = f.read()

        with open("/root/Project/enhanced_similarity_plot.png", 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 5. 返回JSON响应，包含视频数据和检测结果
        return jsonify({
            'video_data': video_bytes.hex(),  # 二进制转十六进制字符串
            'result': {
                'score': similarity_score,
                'prediction': prediction,
                'filename': original_filename,
                'distortion_type': distortion_type,
                'distortion_level': distortion_level,
                "image_base64": encoded_string
            }
        })

    except Exception as e:
        print(f"[ERROR] 扰动处理失败: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_work_dir):
            shutil.rmtree(temp_work_dir, ignore_errors=True)

def add_audio_to_video(video_path, audio_source_path, output_path):
    # 用原视频音频合成到扰动后视频
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        output_path
    ]
    subprocess.run(cmd, check=True)

@app.route("/")
def home():
    return "语音取证检测后端服务正在运行！" # 中文消息

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)