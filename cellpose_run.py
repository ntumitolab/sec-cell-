import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.system("python -m cellpose")