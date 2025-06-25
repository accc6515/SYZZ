import subprocess
import sys
import os

# 设置参数（替换为实际路径）

video_root = '/root/Project/webBack/videos/video'
file_list = '/root/Project/webBack/videos/test.txt'
out_dir = '/root/Project/webBack/videos/landmark'

# 构建脚本路径（确保路径正确）
script_path = os.path.join("preprocessing", "face-alignment", "landmark_extract.py")

# 构造命令列表
cmd = [
    sys.executable,  # 使用当前Python解释器
    script_path,
    "--video_root", video_root,
    "--file_list", file_list,
    "--out_dir", out_dir
]

try:
    # 执行命令并实时显示输出
    subprocess.run(cmd, check=True)
    print("\n脚本执行成功！")
except subprocess.CalledProcessError as e:
    print(f"\n执行出错，错误码：{e.returncode}")
except FileNotFoundError:
    print("错误：Python解释器或脚本路径不存在")
except Exception as e:
    print(f"未知错误：{str(e)}")


video_root = '/root/Project/webBack/videos/video'
file_list = '/root/Project/webBack/videos/test.txt'
landmark_dir = '/root/Project/webBack/videos/landmark'
out_dir = '/root/Project/webBack/videos/outputvideo'

# 构建脚本路径（确保路径正确）
script_path2 = os.path.join("preprocessing", "align_mouth.py")

# 构造命令列表
cmd2 = [
    sys.executable,  # 使用当前Python解释器
    script_path2,
    "--video_root", video_root,
    "--file_list", file_list,
    "--landmark_dir",landmark_dir,
    "--out_dir", out_dir
]

try:
    # 执行命令并实时显示输出
    subprocess.run(cmd2, check=True)
    print("\n脚本2执行成功！")
except subprocess.CalledProcessError as e:
    print(f"\n执行出错，错误码：{e.returncode}")
except FileNotFoundError:
    print("错误：Python解释器或脚本路径不存在")
except Exception as e:
    print(f"未知错误：{str(e)}")


video_path = '/root/Project/webBack/videos/outputvideo/00109_2_id00166_wavtolip.mp4'
audio_path = '/root/Project/webBack/videos/outputvideo/00109_2_id00166_wavtolip.wav'

# 构建脚本路径（确保路径正确）
script_path3 = os.path.join("evaluate88.py")

# 构造命令列表
cmd3 = [
    sys.executable,  # 使用当前Python解释器
    script_path3,
    "--video_path", video_path,
    "--audio_path", audio_path,
]

try:
    # 执行命令并实时显示输出
    subprocess.run(cmd3, check=True)
    print("\n脚本3执行成功！")
except subprocess.CalledProcessError as e:
    print(f"\n执行出错，错误码：{e.returncode}")
except FileNotFoundError:
    print("错误：Python解释器或脚本路径不存在")
except Exception as e:
    print(f"未知错误：{str(e)}")



# def run_script(script_path, args):
#     """执行脚本并返回输出"""
#     try:
#         result = subprocess.run(
#             [script_path] + args,
#             check=True,  # 自动检查命令是否成功
#             capture_output=True,
#             text=True,
#             timeout=600  # 设置执行超时（秒）
#         )
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         return f"脚本执行失败（错误码 {e.returncode}）:\n{e.stderr}"
#     except Exception as e:
#         return f"执行异常: {str(e)}"

# if __name__ == '__main__':
#     try:
#         # 第一步：运行人脸关键点提取
#         script1_args = [
#             '--video_root', '/root/Project/webBack/vedio/video',
#             '--file_list', '/root/Project/webBack/vedio/test.txt',
#             '--out_dir', '/root/Project/webBack/vedio/landmark'
#         ]
#         landmarks_output = run_script(
#             '/root/Project/preprocessing/face-alignment/landmark_extract.py',
#             script1_args
#         )

#         # 第二步：运行嘴巴对齐处理
#         script2_args = [
#             '--video_root', '/root/Project/webBack/vedio/video',
#             '--file_list', '/root/Project/webBack/vedio/test.txt',
#             '--landmarks_dir', '/root/Project/webBack/vedio/landmark',
#             '--out_dir', '/root/Project/webBack/vedio/outputvideo'
#         ]
#         print(landmarks_output.strip())
#         align_mouth_output = run_script(
#             '/root/Project/preprocessing/align_mouth.py',
#             script2_args
#         )
#         print(align_mouth_output.strip())
#         # 第三步 评估
#         script3_args = [
#             '--video_path', '/root/Project/webBack/vedio/outputvideo/00109_2_id00166_wavtolip.mp4',
#             '--audio_path', '/root/Project/webBack/vedio/outputvideo/00109_2_id00166_wavtolip.wav',
#         ]
#         final_output = run_script('/root/Project/evaluate88.py', script3_args)
#         print(final_output.strip())
#         # return jsonify({"result": final_output.strip()})
#     except Exception as e:
#         # return jsonify({"error": str(e)}), 500
#         print('Exception',500)