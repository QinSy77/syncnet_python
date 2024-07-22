ffmpeg -i /mnt/workspace/qinshiyang/AniPortrait/utils/stage2_score/video/guangfan.mp4 -vf scale=224:224 /mnt/workspace/qinshiyang/syncnet_python/data/guangfan_256.mp4
ffmpeg -i /mnt/workspace/qinshiyang/AniPortrait/utils/stage2_score/video/guangfan.mp4 /mnt/workspace/qinshiyang/syncnet_python/data/guangfan.avi
ffmpeg -i /mnt/workspace/qinshiyang/syncnet_python/data/example.avi /mnt/workspace/qinshiyang/syncnet_python/data/example.mp4
ffmpeg -i /mnt/workspace/qinshiyang/hallo/tmp_cache/output_video/gufeng_libai.mp4 -vf scale=224:224 /mnt/workspace/qinshiyang/syncnet_python/data/gufeng_libai_224.mp4
ffmpeg -i /mnt/workspace/qinshiyang/syncnet_python/data/gufeng1_libai_224+wavlip.mp4 -vf scale=512:512 /mnt/workspace/qinshiyang/syncnet_python/data/gufeng1_libai_512+wavlip.mp4

ffmpeg -i /mnt/workspace/qinshiyang/hallo/tmp_cache/output_video/gufeng1_libai_25step.mp4 -vf scale=224:224 /mnt/workspace/qinshiyang/syncnet_python/data/gufeng1_libai_25step_224.mp4