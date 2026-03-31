#!/bin/bash

# cd /mnt/petrelfs/yejinhui/Projects/starVLA
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/share/project/fengyupu/github/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/share/project/fengyupu/conda_envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH}       # let LIBERO find the websocket tools from main repo

# # EGL渲染设置
# export MUJOCO_EGL_DEVICE_ID=0
# export EGL_DEVICE_ID=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# # 强制使用NVIDIA的EGL驱动
# export __NV_PRIME_RENDER_OFFLOAD=1
# export __GLX_VENDOR_LIBRARY_NAME=nvidia
# # 强制使用X11平台而不是DEVICE平台
# export EGL_PLATFORM=x11
# export MUJOCO_EGL_PLATFORM_ID=0
# 确保使用NVIDIA EGL
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
# export MUJOCO_GL=osmesa
export MUJOCO_GL=egl

# host="127.0.0.1"
host="172.24.87.146"
base_port=5001
unnorm_key="franka"
your_ckpt=./results/Checkpoints/1208_libero_all_QwenPI_qwen3/checkpoints/steps_50000_pytorch_model.pt
# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

task_suite_name=libero_goal
num_trials_per_task=50
# video_out_path="results/${task_suite_name}/${folder_name}"
video_out_path="results/${task_suite_name}/20260324_pi05_libero_18k"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path ${your_ckpt} \
  --args.host "$host" \
  --args.port $base_port \
  --args.task-suite-name "$task_suite_name" \
  --args.num-trials-per-task "$num_trials_per_task" \
  --args.video-out-path "$video_out_path"
