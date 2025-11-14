#!/bin/bash

# 创建日志目录
LOG_DIR=" "
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/[SFT]internLM3_8B_${TIMESTAMP}.log"

# 设置环境变量
# export ENABLE_AUDIO_OUTPUT=False
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# 设置随机端口号，避免端口冲突
export MASTER_PORT=$((10000 + RANDOM % 50000))

# 先打印启动信息
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Using port: $MASTER_PORT"

# 没有指定 model_type
# 启动训练并获取PID
nohup swift sft \
    --model ' ' \
    --dataset ' ' \
    --eval_steps 1000 \
    --train_type lora \
    --lora_rank 16 \
    --lora_dropout 0.01 \
    --lorap_lr_ratio 16\
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 6 \
    --target_modules all-linear \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --save_total_limit 20 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 1 \
    --max_length 6000 \
    --output_dir  \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --metric acc \
    --freeze_vit true \
    --model_author   \
    --model_name   \
    > "$LOG_FILE" 2>&1 &

# 获取PID并等待一下确保进程启动
TRAIN_PID=$!
sleep 2

# 检查进程是否还在运行
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training started successfully with PID $TRAIN_PID"
    echo "To view logs in real-time, use:"
    echo "tail -f $LOG_FILE"
    echo ""
    echo "To stop training, use:"
    echo "kill -9 $TRAIN_PID"
else
    echo "Failed to start training process"
    echo "Check log file for errors: $LOG_FILE"
fi