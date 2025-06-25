# import cv2
# import os
# import numpy as np

# def extract_and_concat(video_path, save_path, num_frames=10, resize_height=200):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"无法打开视频：{video_path}")
#         return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"视频总帧数：{total_frames}")

#     interval = total_frames // num_frames
#     images = []

#     for i in range(num_frames):
#         frame_index = i * interval
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#         ret, frame = cap.read()
#         if ret:
#             # 缩放图像到统一高度
#             h, w = frame.shape[:2]
#             scale = resize_height / h
#             new_w = int(w * scale)
#             resized_img = cv2.resize(frame, (new_w, resize_height))
#             images.append(resized_img)
#         else:
#             print(f"无法读取第 {frame_index} 帧")

#     cap.release()

#     if not images:
#         print("没有成功提取的帧，无法拼接")
#         return

#     final_image = cv2.hconcat(images)
#     cv2.imwrite(save_path, final_image)
#     print(f"保存拼接图像到：{save_path}")


import cv2
import os
import numpy as np
import argparse

def extract_and_concat(video_path, save_path, num_frames=10, resize_height=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数：{total_frames}")

    interval = total_frames // num_frames
    images = []

    for i in range(num_frames):
        frame_index = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            scale = resize_height / h
            new_w = int(w * scale)
            resized_img = cv2.resize(frame, (new_w, resize_height))
            images.append(resized_img)
        else:
            print(f"无法读取第 {frame_index} 帧")

    cap.release()

    if not images:
        print("没有成功提取的帧，无法拼接")
        return

    final_image = cv2.hconcat(images)
    cv2.imwrite(save_path, final_image)
    print(f"保存拼接图像到：{save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取帧并拼接为一张图像")
    parser.add_argument('--video_path', required=True, help='输入视频路径')
    parser.add_argument('--save_path', required=True, help='输出拼接图像保存路径')
    parser.add_argument('--num_frames', type=int, default=10, help='要提取的帧数，默认10')
    parser.add_argument('--resize_height', type=int, default=200, help='图像高度，默认200')

    args = parser.parse_args()
    extract_and_concat(
        video_path=args.video_path,
        save_path=args.save_path,
        num_frames=args.num_frames,
        resize_height=args.resize_height
    )
