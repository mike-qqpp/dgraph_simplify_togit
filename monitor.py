#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
n 2.7 实时监控 GPU 各项最大值并写入 txt
pip install nvidia-ml-py3
"""
from __future__ import print_function
import time
import signal
import sys
from datetime import datetime

try:
    import pynvml
except ImportError:
    sys.exit("请先安装 nvidia-ml-py3：pip install nvidia-ml-py3")

pynvml.nvmlInit()
dev_count = pynvml.nvmlDeviceGetCount()
log = open("gpu_max.txt", "a", 1)  # 行缓冲，追加（将 0 改为 1）

# 中断时 flush 并关闭
def stop(sig, frm):
    log.close()
    pynvml.nvmlShutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

# 初始化每卡最大值
max_temp = [0] * dev_count
max_power = [0.0] * dev_count
max_mem = [0] * dev_count
max_util = [0] * dev_count

# 表头
header = "time"
for i in range(dev_count):
    header += "\tGPU{i}_maxTemp\tGPU{i}_maxPower\tGPU{i}_maxMem\tGPU{i}_maxUtil".format(i=i)
log.write(header + "\n")

print("GPU 最大值监控已启动，每秒写入 gpu_max.txt ...（Ctrl+C 退出）")

interval = 0.05  # 秒
while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = now
    for i in range(dev_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).used // 1048576
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu

        # 更新最大值
        max_temp[i] = max(max_temp[i], temp)
        max_power[i] = max(max_power[i], power)
        max_mem[i] = max(max_mem[i], mem)
        max_util[i] = max(max_util[i], util)

        # 格式保留两位小数
        line += "\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            max_temp[i], max_power[i], max_mem[i], max_util[i])
    log.write(line + "\n")
    time.sleep(interval)
