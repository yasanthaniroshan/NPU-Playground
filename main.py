from rknn.api import RKNN
import numpy as np
import os

rknn = RKNN()

# Config target platform
rknn.config(
    mean_values=[[0,0]],
    std_values=[[1,1]],
    quantized_dtype='w8a8',      # int8 weights & activations
    quantized_algorithm='mmse',   # reduces quantization error
    target_platform='rv1103',
    optimization_level=2
)

# Load ONNX
ret = rknn.load_tflite(model='model.tflite')
if ret != 0:
    print('Load model failed!')
    exit(ret)

ret = rknn.build(do_quantization=False)

if ret != 0:
    print('Build model failed!')
    exit(ret)    

ret = rknn.init_runtime(target='rv1103',perf_debug=True)
if ret != 0:
    print("Init runtime failed!")
    exit(ret)

file_names = os.listdir(os.path.join(os.getcwd(),"dataset"))
inputs = ["dataset/"+x for x in file_names]
print(inputs)
results = rknn.accuracy_analysis(inputs=inputs,target='rv1103')


# Export RKNN model
rknn.export_rknn('model.rknn')
print("RKNN model exported successfully!")
