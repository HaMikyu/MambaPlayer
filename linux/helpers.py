import os
from glob import glob
from torch import bfloat16,float16,float32,where,Tensor,load,LongTensor
from torch.nn import BCEWithLogitsLoss,MSELoss,CrossEntropyLoss
from torchvision.transforms.functional import pil_to_tensor
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randint
import pickle
import numpy as np
import cv2
from numpy import argmax
import socket
def get_newest_epoch(link_do_modeli):
    epoch = 0
    if not os.path.isdir(link_do_modeli):
        os.mkdir(link_do_modeli)
    else:
        wyniki = glob(link_do_modeli + '/*')
        if wyniki:
            for c in range(len(wyniki)):
                x = wyniki[c].split('models/')[1][6:-4]
                wyniki[c] = int(x)
            epoch = max(wyniki)
    return epoch
def get_newest_model(cwd):
    epoch = get_newest_epoch(cwd)
    modelik = cwd+ f'/model-{epoch}.pth'
    model = load(modelik)
    print(f"MODEL LOADED: {f'model-{epoch}.pth'}")
    return model

def get_type(s):

    if s['dtype'] == 'torch.bfloat16':
        s['dtype'] = bfloat16
    elif s['dtype'] == 'torch.float16':
        s['dtype'] = float16
    elif s['dtype'] == 'torch.float32':
        s['dtype'] = float32

    return s
def combined_loss(tensor, targets,x_space,y_space,cross_x,cross_y,button_loss):

    x_tensor = tensor[:, :len(x_space)].cuda()
    y_tensor = tensor[:, len(x_space):len(x_space)+len(y_space)].cuda()
    buttons_tensor= tensor[:, len(x_space)+len(y_space):].cuda()


    x_targets = targets[:, :1].squeeze().type(LongTensor).cuda()
    y_targets  = targets[:, 1:2].squeeze().type(LongTensor).cuda()
    buttons_targets = targets[:, 2:].cuda()

    #print(x_tensor.shape, x_targets.shape)
    #print(y_tensor.shape, y_targets.shape)
    #print(buttons_tensor.shape, buttons_targets.shape)

    x_loss=cross_x(x_tensor,x_targets)
    y_loss=cross_y(y_tensor,y_targets)
    buttons_loss2=button_loss(buttons_tensor,buttons_targets)

    combined_loss = x_loss + y_loss+buttons_loss2

    return combined_loss
def prepare_image(img,s):
    if s['cropping'] is not False:
        width, height = img.size
        img = img.crop((width / 4, height / 4, 3 * (width / 4), 3 * (height / 4)))
    img = img.resize((s['image_size'], s['image_size'])).convert('RGB')
    img = pil_to_tensor(img)
    img = Tensor(img).type(s['dtype']).to(s['device'])
    return img
def output_to_dic(y):
    y=y.squeeze(0)
    y=y.tolist()

    c = 0

    with open('encoder_settings.json', 'r') as openfile:
        s = json.load(openfile)

    d={}

    if s['no_mouse']==False:
        keys=[]
        keys.append("X_char")
        for x in range(s['mouse_accuracy']):
            keys.append(f"X_{x+1}")
        keys.append("Y_char")
        for x in range(s['mouse_accuracy']):
            keys.append(f"Y_{x+1}")

        for key in keys:
            d[key]=y[c]
            c+=1

    with open(s['dictionary'], 'rb') as file:
        dictionary = pickle.load(file)

    for key in dictionary:
        d[key]=y[c]
        c+=1

    return d

def output_to_keys(y):

    with open('encoder_settings.json', 'r') as openfile:
        s = json.load(openfile)

    with open(s['dictionary'], 'rb') as file:
        dictionary = pickle.load(file)

    y = y.squeeze().tolist()
    mx = y[:len(s['space_x'])]
    my = y[len(s['space_x']):len(s['space_x']) + len(s['space_y'])]
    mb = y[len(s['space_x']) + len(s['space_y']):]

    mx = int(s['space_x'][argmax(mx)])
    my = int(s['space_y'][argmax(my)])
    buttons = []
    d = list(dictionary.keys())
    for c in range(len(mb)):
        if mb[c] > 0:
            buttons.append(d[c])

    return mx, my, buttons

def dic_to_keys(d):

    assert type(d)==type({4:'6'})

    with open('encoder_settings.json', 'r') as openfile:
        s = json.load(openfile)
    s = get_type(s)


    if s['no_mouse']==True:
        buttons=[]
        for x in d:
            if int(d[x])>=1:
                buttons.append(x)
        return buttons

    #print(list(d.keys())[0], list(d.keys())[s['mouse_accuracy'] + 1])

    x_char=1
    x_list=[]

    y_char=1
    y_list=[]

    x_char=d["X_char"]*(-1)
    y_char=d["Y_char"]*(-1)

    #print(x_char,y_char)

    x=""
    y=""

    for x1 in range(1,s['mouse_accuracy']+1):
        key=list(d.keys())[x1]
        x+=str(int(d[key]))

    for x1 in range(s['mouse_accuracy']+2,s['mouse_accuracy']*2+2):
        key=list(d.keys())[x1]

        y += str(int(d[key]))


    x=int(x,2)*x_char
    y=int(y,2)*y_char

    buttons=[]

    for x1 in range(s['mouse_accuracy']*2+2,len(d)):
        key=list(d.keys())[x1]
        #print(key)
        if int(d[key])>=1:
            buttons.append(key)

    return int(x),int(y),buttons
def send_list(l,server_address = ('192.168.2.5', 12345)):
    serialized_data = pickle.dumps(l)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(serialized_data, server_address)

def convert_from_image_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def takeClosest(num, collection):
    return min(collection, key=lambda x: abs(x - num))