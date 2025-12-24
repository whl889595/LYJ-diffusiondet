# -*- coding: utf-8 -*-
import subprocess
import re
import time
import os

# ===================== 配置项（根据你的需求修改） =====================
CHECK_INTERVAL = 20  # 检查间隔，单位：秒
GPU_ID = 0  # 要监控的GPU编号
FREE_MEM_THRESHOLD = 20 * 1024  # 目标未用显存：20GB = 20480 MiB
TRAIN_COMMAND = (
    # 先设置显存优化参数，再执行训练（降低batch size避免显存不足）
    "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && "
    "nohup python train_net.py --num-gpus 1 --batch-size 2 "
    "--config-file configs/diffdet.coco.res50.yaml > training.log 2>&1 &"
)
# ====================================================================

def get_gpu_memory_info(gpu_id=0):
    """获取指定GPU的总显存、已用显存、未用显存（单位：MiB）"""
    try:
        # 执行nvidia-smi命令，获取总显存和已用显存
        cmd = [
            "nvidia-smi", 
            "--id={}".format(gpu_id), 
            "--query-gpu=memory.total,memory.used", 
            "--format=csv,nounits,noheader"
        ]
        result = subprocess.check_output(cmd, encoding="utf-8").strip()
        # 提取数字（总显存, 已用显存）
        mem_total, mem_used = map(int, re.findall(r"\d+", result))
        mem_free = mem_total - mem_used  # 计算未用显存
        return mem_total, mem_used, mem_free
    except Exception as e:
        print("获取GPU显存失败：{}".format(e))
        return None, None, None

def main():
    print("===== GPU显存监控脚本启动 =====")
    print("监控GPU {}，目标：未用显存 > {}GB（{}MiB）".format(
        GPU_ID, FREE_MEM_THRESHOLD/1024, FREE_MEM_THRESHOLD
    ))
    print("检查间隔：{}秒".format(CHECK_INTERVAL))
    print("满足条件后将执行命令：\n{}\n".format(TRAIN_COMMAND))

    while True:
        # 获取显存信息
        mem_total, mem_used, mem_free = get_gpu_memory_info(GPU_ID)
        if None in [mem_total, mem_used, mem_free]:
            time.sleep(CHECK_INTERVAL)
            continue
        
        # 打印当前状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("[{}] GPU {} 状态：".format(current_time, GPU_ID))
        print("  总显存：{} MiB（{:.2f} GB）".format(mem_total, mem_total/1024))
        print("  已用显存：{} MiB（{:.2f} GB）".format(mem_used, mem_used/1024))
        print("  未用显存：{} MiB（{:.2f} GB）".format(mem_free, mem_free/1024))

        # 判断：未用显存 > 20GB 时执行训练
        if mem_free > FREE_MEM_THRESHOLD:
            print("\n✅ 满足条件！未用显存大于{}GB，开始执行训练命令...".format(FREE_MEM_THRESHOLD/1024))
            os.system(TRAIN_COMMAND)
            print("✅ 训练命令已执行！日志文件：training.log")
            break
        else:
            print("❌ 未满足条件（未用显存≤{}GB），{}秒后再次检查...\n".format(
                FREE_MEM_THRESHOLD/1024, CHECK_INTERVAL
            ))
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()