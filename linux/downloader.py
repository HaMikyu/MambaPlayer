import time
import threading
import requests
from PIL import Image
from io import BytesIO
import numpy
import cv2
from helpers import get_newest_model
import os
from torch import set_float32_matmul_precision
from helpers import get_newest_model,prepare_image
import torch.multiprocessing as mp
import json
from dataset import MyDataset_Ultimate
from mamba_blocks import MambaTower,MambaBlock
from torch import where
from model import Model_Ultimate
from helpers import dic_to_keys,output_to_dic,send_list,convert_from_image_to_cv2,output_to_keys
class Downloader:

    def __init__(self):
        self.url = r"http://192.168.2.5:8000/camera/jpeg"
        self.counter = 0
        self.start = time.time()
        self.lock=threading.Lock()
        self.imgs=[]
        self.working=True

        with open('encoder_settings.json', 'r') as openfile:
            self.s = json.load(openfile)

        #self.condition = threading.Condition()

        for x in range(5):
            threading.Thread(target=self.thread,args=(x,)).start()

    def end(self):
        self.working=False

    def thread(self,x):
        while self.working:

            response = requests.get(self.url, stream=True)
            img = Image.open(BytesIO(response.content))
            if img.size !=(1280,720):
                print('xd')
                continue


            if len(self.imgs)<5:
                self.imgs.append(img)
            self.counter += 1
            #self.condition.notify_all()



    def get_image(self):
        #with self.condition:
            while len(self.imgs)<=0:
                time.sleep(0.01)
            return self.imgs.pop()


if __name__ == '__main__':

    mp.set_sharing_strategy('file_descriptor')
    set_float32_matmul_precision('medium')
    mp.set_start_method('spawn')

    with open('encoder_settings.json', 'r') as openfile:
        encoder_settings = json.load(openfile)

    model=Model_Ultimate(encoder_settings).to(encoder_settings['device'])
    try:
        model = get_newest_model(os.getcwd() + "/models")
    except:
        print("Fresh model!")

    downloader = Downloader()
    img=downloader.get_image()
    img=prepare_image(img,encoder_settings)
    if img.ndim==3:
        img=img.unsqueeze(0)
    y=model(img)
    print("Now running!")
    while True:
        img = downloader.get_image()
        cv2.imshow('Frame', convert_from_image_to_cv2(img))
        img = prepare_image(img, encoder_settings)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        if img.ndim == 3:
            img = img.unsqueeze(0)
        y = model(img)


        y=output_to_keys(y)
        #y = where(y > 0, 1, 0).to(encoder_settings['dtype'])
        #y = output_to_dic(y)
        #y = dic_to_keys(y)
        print(y)
        send_list(y)
    downloader.end()
