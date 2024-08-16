cd /mnt/workspace/qinshiyang/syncnet_python
export CUDA_VISIBLE_DEVICES=7
# source /mnt/workspace/qinshiyang/.bashrc

# conda activate /mnt/workspace/qinshiyang/miniconda3/envs/down_load_data




python -m syncner_hander \
        --video_json_file_path /mnt/workspace/qinshiyang/hallo/test_out_metric/liveportrait/vfhq_liveportrait.json \
        --video_json_file_key output_video_path  \
        --out_json_path /mnt/workspace/qinshiyang/hallo/hallo_metric/metric_results/vfhq_liveportrait_syncnet_0721.json 

# python -m syncner_hander \
#         --video_dir /mnt/workspace/qinshiyang/hallo/test_out_metric/aniportrait/cnc_videos/20240719/1547--seed_42-512x512_audio2vid_guanfang_guanfang \
#         --out_json_path /mnt/workspace/qinshiyang/hallo/hallo_metric/metric_results/cnc_aniportrait_syncnet.json