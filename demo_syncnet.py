#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
#parser.add_argument('--videofile', type=str, default="data/example.avi", help='');

parser.add_argument('--video_json_file_path', type=str, default="data/example.json", help='');

parser.add_argument('--out_json_path', type=str, default="data/out.json", help='');

parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');


opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

import json
import os

video_json_file_path = opt.video_json_file_path
out_json_path = opt.out_json_path


with open(video_json_file_path, 'r') as file:
    # 加载JSON文件中的数据到一个字典
    data = json.load(file)

# data = [{
#         "ref_image_path": ref_image_path,
#         "output_video": model_out_save_path,
#         "gt_video": audio_path,
#         "show_video":save_path
#         }]
results = {}

for item in data:

    videofile = item.get("output_video_resize_224")
    #videofile = item.get("gt_video_resize_224")
    filename = os.path.splitext(os.path.basename(videofile))[0]

    offset,minval,conf = s.evaluate(opt, videofile=videofile)
    #import pdb 
    #pdb.set_trace()  # 在这里设置断点
    #print(offset,minval,conf)
    results[filename] = {"LSE_C":conf,"LSE_D":minval}
    #break

with open(out_json_path, 'w') as json_file:
    # 将字典转换为JSON格式并写入文件
    # indent参数用于美化输出，使得JSON文件可读性更好
    json.dump(results, json_file, indent=4)

    print(f"字典已保存为JSON文件: {out_json_path}")

# fname = opt.videofile
# offset, conf, dist = s.evaluate(opt,videofile=fname)
# print(f"File: {fname}, Sync Error: {dist}, Confidence: {conf}, Offset: {offset}")
      
