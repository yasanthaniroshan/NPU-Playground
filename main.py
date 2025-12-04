from rknn.api import RKNN

rknn = RKNN()

# Config target platform
rknn.config(
    mean_values=[[0]],
    std_values=[[1]],
    quantized_dtype='w8a8',      # int8 weights & activations
    quantized_method='layer',     # 'layer' is enough for 1 channel
    quantized_algorithm='mmse',   # reduces quantization error
    target_platform='rv1103',
    optimization_level=2
)

# Load ONNX
ret = rknn.load_tflite(model='model.tflite')
if ret != 0:
    print('Load model failed!')
    exit(ret)

ret = rknn.build(do_quantization=False, dataset="recurrence_plots/data.txt")

if ret != 0:
    print('Build model failed!')
    exit(ret)    

# Export RKNN model
rknn.export_rknn('model.rknn')
print("RKNN model exported successfully!")
