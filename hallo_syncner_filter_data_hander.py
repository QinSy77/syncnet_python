#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("/mnt/workspace/qinshiyang/syncnet_python")


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




def resize_video(video_path, width=224, height=224):
    """Resize video to specified dimensions using moviepy."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
        clip = VideoFileClip(video_path)
        clip_resized = clip.resize(newsize=(width,height))
        clip_resized.write_videofile(temp_video_path, codec='libx264', audio_codec='aac')
        return temp_video_path

def get_video_path(args):
    if args.video_json_file_path:
        with open(args.video_json_file_path, 'r') as file:
            data = json.load(file)
        #import pdb; pdb.set_trace()
        return [item.get(args.video_json_file_key) for item in data]

    elif args.video_dir:
        video_files = sorted([os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))])
        #import pdb; pdb.set_trace()
        return video_files

def process_single_video(video_path,args):

    try:
        resized_videofile = resize_video(video_path)

        offset, lse_d, lse_c = model.evaluate(args, videofile=resized_videofile)

        print({"lse_d":lse_d , "lse_c":lse_c})

        if lse_d > 10 or lse_c < 1 :
            return False

        else:
            return True

        # 删除临时视频文件
        os.remove(resized_videofile)
    except:
        return False

def filter_data_form_json(args,file_path, output_file_path, hand_file_path, part, rank):
    extracted_data = []
    hand_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for idx, item in enumerate(data):
            if idx % part != rank:
                continue
            try:
                video_path = item["video_path"]

                if process_single_video(video_path,args) :
                    extracted_data.append(item)

                else:
                    print(f"不合格数据为：{video_path}")
                    hand_data.append(item)

            except (KeyError, TypeError) as e:
                print(f"在处理{item}时发生错误: {e}")

        with open(output_file_path, 'w', encoding='utf-8') as new_file:
            json.dump(extracted_data, new_file, ensure_ascii=False, indent=4)

        with open(hand_file_path, 'w', encoding='utf-8') as new_file:
            json.dump(hand_data, new_file, ensure_ascii=False, indent=4)

        print(f"合格数据已写入到 {output_file_path}", f"数量为：{len(extracted_data)}")
        print(f"不合格的数据已写入到 {hand_file_path}", f"数量为：{len(hand_data)}")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的JSON格式。")


def delete_folder(folder_path):
    """递归删除文件夹及其内容"""
    try:
        shutil.rmtree(folder_path)
        print(f"文件夹 {folder_path} 及其内容已被删除。")
    except OSError as e:
        print(f"删除文件夹时出错: {e}")


    

def main(args):
    from datetime import datetime

    # 获取当前时间
    now = datetime.now()
    # 格式化时间字符串
    time_str = now.strftime("%H:%M:%S")
    #print(time_str)

    args.reference = f"{args.reference}_{time_str}_{args.part}_{args.rank}"


    # ==================== RUN EVALUATION ====================
    global model

    model = SyncNetInstance()

    

    model.loadParameters(args.initial_model)
    print("Model %s loaded." % args.initial_model)

    flag = process_single_video(video_path,args)

    print(flag)

    #filter_data_form_json(args,args.json_file_path, args.output_file_path, args.hand_file_path, args.part, args.rank)

    
    #os.makedirs(os.path.dirname(args.out_json_path),exist_ok=True)


    #delete_folder(f"/mnt/workspace/qinshiyang/syncnet_python/data/work/pytmp/{args.reference}")




if __name__ == "__main__":


# ==================== LOAD PARAMS ====================


    parser = argparse.ArgumentParser(description="SyncNet")

    parser.add_argument('--initial_model', type=str, default="/mnt/workspace/qinshiyang/syncnet_python/data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default=20, help='')
    parser.add_argument('--vshift', type=int, default=15, help='')

    #parser.add_argument("--json_file_path", type=str, required=True, help="Path to the input JSON file.")
    #parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output JSON file without hand data.")
    #parser.add_argument("--hand_file_path", type=str, required=True, help="Path to the output JSON file with hand data.")

    parser.add_argument("--part", type=int, default=1, help="Partition number for distributed processing.")
    parser.add_argument("--rank", type=int, default=0, help="Rank for distributed processing.")



    parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='')
    parser.add_argument('--reference', type=str, default="test", help='')


    args = parser.parse_args()
    video_path = "/mnt/workspace/datasets/TalkingFaceProject/VFHQ/version_0715/outputs_100/Clip+-6ULbJet9SU+P0+C2+F8957-9129.mp4"
    
    main(args)
