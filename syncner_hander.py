#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import argparse
import subprocess
import json
import os
import tempfile
from moviepy.editor import VideoFileClip

from SyncNetInstance import *
import os
import shutil


# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description="SyncNet")

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
parser.add_argument('--batch_size', type=int, default=20, help='')
parser.add_argument('--vshift', type=int, default='15', help='')

parser.add_argument('--video_json_file_path', type=str, help='')

parser.add_argument('--video_json_file_key', type=str, default="", help='')

parser.add_argument('--video_dir', type=str, help='Directory containing video files')

parser.add_argument('--out_json_path', type=str, required=True, help='')

parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')
parser.add_argument('--reference', type=str, default="test", help='')


opt = parser.parse_args()

from datetime import datetime

# 获取当前时间
now = datetime.now()
# 格式化时间字符串
time_str = now.strftime("%H:%M:%S")
#print(time_str)

opt.reference = f"{opt.reference}_{time_str}"


# ==================== RUN EVALUATION ====================

s = SyncNetInstance()

s.loadParameters(opt.initial_model)
print("Model %s loaded." % opt.initial_model)


def resize_video(video_path, width=224, height=224):
    """Resize video to specified dimensions using moviepy."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
        clip = VideoFileClip(video_path)
        clip_resized = clip.resize(newsize=(width,height))
        clip_resized.write_videofile(temp_video_path, codec='libx264', audio_codec='aac')
        return temp_video_path


def evaluate_videos(video_list):
    mean_minval = 0.0

    mean_conf = 0.0

    index = 0
    
    for videofile in video_list:
        #filename = os.path.splitext(os.path.basename(videofile))[0]
        resized_videofile = resize_video(videofile)

        offset, minval, conf = s.evaluate(opt, videofile=resized_videofile)

        index +=1

        mean_minval +=minval
        mean_conf+=conf
        

        # 删除临时视频文件
        os.remove(resized_videofile)

    mean_minval = mean_minval / index
    mean_conf = mean_conf / index


    result =  {"LSE_C": mean_conf, "LSE_D": mean_minval}

    print(f"LSE_C: {mean_conf}, LSE_D: {mean_minval} ")

    return result


if opt.video_json_file_path:
    with open(opt.video_json_file_path, 'r') as file:
        data = json.load(file)
    #import pdb; pdb.set_trace()
    results = evaluate_videos([item.get(opt.video_json_file_key) for item in data])

elif opt.video_dir:
    video_files = sorted([os.path.join(opt.video_dir, f) for f in os.listdir(opt.video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))])
    #import pdb; pdb.set_trace()
    results = evaluate_videos(video_files)

os.makedirs(os.path.dirname(opt.out_json_path),exist_ok=True)


with open(opt.out_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

def delete_folder(folder_path):
    """递归删除文件夹及其内容"""
    try:
        shutil.rmtree(folder_path)
        print(f"文件夹 {folder_path} 及其内容已被删除。")
    except OSError as e:
        print(f"删除文件夹时出错: {e}")
delete_folder(f"/mnt/workspace/qinshiyang/syncnet_python/data/work/pytmp/{opt.reference}")
print(f"字典已保存为JSON文件: {opt.out_json_path}")