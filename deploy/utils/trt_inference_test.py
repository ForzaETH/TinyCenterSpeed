#!/usr/bin/env python3

import tensorrt as trt
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

with open('/home/race_crew/catkin_ws/src/race_stack/perception/dataset/ml/trained_models/sample.engine', 'rb') as f:
    selialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(selialized_engine)



context = engine.create_execution_context()

input = torch.randn(1,3,256,256).cuda()
output = torch.zeros(1, 1, 256, 256).cuda()

context.set_input_shape('input.1', (1, 3, 256, 256))
context.set_tensor_address("input.1", input.data_ptr())
context.set_tensor_address("16", output.data_ptr())

st = time.perf_counter()
for i in range(100):
    start = time.perf_counter()
    input = torch.randn(1,3,256,256).cuda()
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    end = time.perf_counter()
    print("Time:", (end-start)*1000, "ms")
e = time.perf_counter()
print("Time:", (e-st)/i*1000, "ms")
print(output.shape)
print(output)