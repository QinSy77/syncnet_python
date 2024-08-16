export CUDA_VISIBLE_DEVICES=1
cd /mnt/workspace/qinshiyang/syncnet_python

# source /mnt/workspace/qinshiyang/.bashrc

# conda activate /mnt/workspace/qinshiyang/miniconda3/envs/down_load_data




# python -m syncner_hander \
#         --video_json_file_path /mnt/workspace/qinshiyang/AniPortrait_custom/utils/stage2_score/test_out_video/aniportrait_gf_diff_radio/kanghui_model_out_gf_a2v_diff_0710.json \
#         --video_json_file_key output_video  \
#         --out_json_path /mnt/workspace/qinshiyang/syncnet_python/tmp/test.json 

python -m syncner_hander \
        --video_dir /mnt/workspace/qinshiyang/hallo/test_out_metric/hallo_vfhq_4666item_0809/videos_ckpt150k_step39k \
        --out_json_path /mnt/workspace/qinshiyang/hallo/test_out_metric/hallo_vfhq_4666item_0809/videos_ckpt150k_step39k/hallo_4666item_syncnet_0812_ckpt150k_51k.json \
        #--reference hallo