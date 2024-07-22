cd /mnt/workspace/qinshiyang/syncnet_python
export CUDA_VISIBLE_DEVICES=6

python demo_syncnet.py --video_json_file_path /mnt/workspace/qinshiyang/AniPortrait_custom/utils/stage2_score/test_video_resize/hallo_fps30/results.json \
        --out_json_path /mnt/workspace/qinshiyang/AniPortrait_custom/utils/stage2_score/test_out_video/hallo_fps30_syncet_0706.json 