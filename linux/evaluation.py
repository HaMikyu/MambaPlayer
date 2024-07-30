import lightning as L
import os
import torch
from helpers import get_newest_model
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import json
from dataset import MyDataset_Ultimate
from mamba_blocks import MambaTower,MambaBlock
from model import Model_Ultimate
from PIL import Image
from torch import where
from torchvision.transforms.functional import pil_to_tensor
from pickle import load
from random import randint
from helpers import get_type,output_to_keys
from numpy import array
from numpy import argmax
from torch import bfloat16,float16,float32,Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle



if __name__=="__main__":
    mp.set_sharing_strategy('file_descriptor')
    torch.set_float32_matmul_precision('medium')
    mp.set_start_method('spawn')

    with open('encoder_settings.json', 'r') as openfile:
        s = json.load(openfile)

    with open(s['dictionary'], 'rb') as file:
        dictionary = pickle.load(file)

    s=get_type(s)
    s["use_validation"] = True
    s['fast_training']=False
    md=MyDataset_Ultimate(s)
    model = get_newest_model(os.getcwd() + "/models").cuda().eval()
    print(model)

    for c in range(len(md)):
        inputs=md[c][0].unsqueeze(0).to('cuda')
        out=model(inputs)
        x=int(md[c][1][0])
        y=int(md[c][1][1])
        #x=int(s['space_x'][x])
        #y=int(s['space_y'][y])
        f=md[c][1][2]
        if f>0:
            f=1
        else:
            f=0
        print(f"{output_to_keys(out)} -> {x} {y} {f}")
