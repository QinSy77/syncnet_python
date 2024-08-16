#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ
import torch

import numpy as np
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree

from moviepy.editor import VideoFileClip
import tempfile
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(device);

    def evaluate(self, opt, videofile):

        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========


        print("开始测试")

        resize_flag = False

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_videofile:
            # 加载视频文件
            video = VideoFileClip(videofile)

            # 检查视频的帧率和分辨率
            if video.fps == 25 and video.size == (224, 224):
                # 如果满足条件，不需要修改
                pass
            else:
                # 如果不满足条件，修改视频的分辨率和帧率
                # 使用 resize 方法改变分辨率，然后设置新的帧率
                video = video.resize(newsize=(224, 224)).set_fps(25)
                resize_flag = True

            # 写入新视频，确保在修改后写入（如果需要修改的话）
            video.write_videofile(temp_videofile.name, codec='libx264', audio_codec='aac')

            # 关闭视频剪辑，释放资源
            video.close()


        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            
            
            # 使用安全的路径创建临时的wav文件
            audio_file_path = temp_wav.name
            if resize_flag :

                # 构建ffmpeg命令
                command = (
                    "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}"
                    .format(temp_videofile.name, audio_file_path)
                )
            else:
                # 构建ffmpeg命令
                command = (
                    "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}"
                    .format(videofile, audio_file_path)
                )

                
                # 使用subprocess调用ffmpeg命令
            output = subprocess.call(command, shell=True, stdout=None)


            

        # ========== ==========
        # Load video 
        # ========== ==========
        if resize_flag :
            cap = cv2.VideoCapture(temp_videofile.name)
        else:
            cap = cv2.VideoCapture(videofile)

        if not cap.isOpened():
            print(f"Error: Could not open video ")
            return

        images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 如果没有帧可以读取，跳出循环
            
            # 将调整大小后的帧添加到列表中
            images.append(frame)


        cap.release()

        # 将图片列表转换为 np 数组
        im = np.stack(images, axis=0)  # (frame, height, width, channel)

        #import pdb; pdb.set_trace()

        im = np.expand_dims(im,axis=0)
        # 转换维度顺序以匹配 PyTorch 预期的格式 (batch, channel, time, height, width)
        im = np.transpose(im, (0, 4, 1, 2, 3))

        # 将 np 数组转换为 PyTorch 张量

        imtv = torch.from_numpy(im.astype(np.float32))

        #imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())


        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(audio_file_path)
        try:
            os.remove(audio_file_path)
            os.remove(temp_videofile)
        except:
            print("没有需要删除的文件")

        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])


        cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.from_numpy(cc.astype(np.float32))
        #cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.to(device));
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.to(device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = np.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = np.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = np.array([ dist.numpy() for dist in dists ])
        #return offset.numpy(), conf.numpy(), dists_npy
        return offset.numpy().item(),minval.numpy().item(),conf.numpy().item()

    


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
