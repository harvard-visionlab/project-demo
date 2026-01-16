from visionlab.datasets import StreamingDataset
import pandas as pd
import cv2
import litdata
import torch
import sys
import visionlab.project_demo as m
import inspect

print("✅ import successful")
print("module file:", m.__file__)
print("is from working tree? ->", "project_demo" in m.__file__)

print("torch:", torch.__version__)
print("torch cuda runtime tag:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print("device:", device)

print("litdata version:", litdata.__version__)

print("cv2 version:", cv2.__version__)

print("pandas version:", pd.__version__)

print("StreamingDataset:", StreamingDataset)
