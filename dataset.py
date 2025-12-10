import numpy as np
import os

os.makedirs("dataset", exist_ok=True)

N = 100

temperatures = np.round(np.random.uniform(26.7,40,N),2)
humidities = np.round(np.random.uniform(40,100,N),2)
temperatures_F = np.round(temperatures*9/5+32,2)

norm_temps = (temperatures_F - np.min(temperatures_F))/(np.max(temperatures_F) - np.min(temperatures_F))
norm_hums = (humidities - np.min(humidities))/(np.max(humidities) - np.min(humidities))

quantized_temp = norm_temps/0.00392157 -128
quantized_hums = norm_hums/0.00451887  - 83

i = 0

for (t,h) in zip(quantized_temp,quantized_hums):
    data = np.array([t,h],dtype=np.uint8)
    np.save(f"dataset/data_{i}.npy",data)
    i+=1

