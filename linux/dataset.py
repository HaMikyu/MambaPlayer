from torch.utils.data import Dataset
from glob import glob
from pickle import load
from torch import Tensor
from tqdm import tqdm
import os
from torchvision.transforms.functional import pil_to_tensor
from pandas import read_excel
from ast import literal_eval
from numpy import clip
from PIL import Image
from helpers import get_type,takeClosest

class MyDataset_Ultimate(Dataset):

    def __init__(self, s):
        super().__init__()

        self.broken_keys=[]
        self.raw=False
        self.parts=1
        self.buttons = []
        self.photos = []

        with open(s['dictionary'], 'rb') as file:
            self.dictionary = load(file)

        if s["training_dataset"]==s["validation_dataset"]:
            s["fast_training"]='2'

        s=get_type(s)
        s['loc']=s["training_dataset"]
        self.s = s
        loc = s['loc'].replace("\\", "/")[2:]
        letter = s['loc'][0].lower()
        folders = glob(f"/mnt/{letter}{loc}/*")

        folders = [folder for folder in folders if len(glob(f"{folder}/*")) > 1]

        last_buttons=[]

        for folder in tqdm(folders):

            excel_file = f"{folder}/dane.xlsx"
            if not os.path.isfile(excel_file):
                continue

            try:
                a = read_excel(excel_file, dtype=object, header=None)
            except:
                print(f"file {excel_file} is broken! fix it yourself...")
                raise Exception

            l = a.values.tolist()
            l.pop(0)

            for row in l:

                if len(str(row[0])) < 5:
                    continue

                frame_path = row.pop(0).split("frames")[1].strip()
                frame_path=frame_path.replace("\\","")
                x=row.pop(0)
                y=row.pop(0)
                left_mouse=row.pop(0)
                right_mouse=row.pop(0)
                buttons= literal_eval(row.pop(0))
                if left_mouse:
                    buttons.append('key.left_mouse_button')
                if right_mouse:
                    buttons.append('key.right_mouse_button')

                if self.s['fast_training'] =='2' and (abs(x)<10 and abs(y)<10):
                    continue

                if self.s['fast_training'] == '1' and buttons == last_buttons:
                    continue

                self.buttons.append(
                    (x,y,buttons)
                )
                self.photos.append(f"{folder}/frames/{frame_path}")
                last_buttons=buttons

        print(f'Loaded {len(self.buttons)} rows of data!')

    def set_partition(self,x):
        self.parts=x

    def __len__(self):
        return int(len(self.photos)/self.parts)

    def set_raw(self,x):
        self.raw=x

    def __getitem__(self, index):

        index=int(index*self.parts)
        index=int(clip(index,0,int(len(self.photos)/self.parts)))

        index_list = []
        coded_buttons = []

        img_og=Image.open(self.photos[index])
        if self.raw:
            return img_og, self.buttons[index]

        img=img_og.convert('RGB')
        img_og.close()
        if self.s['cropping'] is not False:
            width, height = img.size
            img = img.crop((width / 4, height / 4, 3 * (width / 4), 3 * (height / 4)))
        img=img.resize((self.s['image_size'],
                        self.s['image_size']))
        img=pil_to_tensor(img)
        img = Tensor(img).type(self.s['dtype'])


        x=self.buttons[index][0]
        y=self.buttons[index][1]
        buttons=self.buttons[index][2]

        for button in buttons:
            try:
                index_list.append(self.dictionary[button])
            except KeyError:
                if button not in self.broken_keys:
                    print(f"Unkown key: {button}")
                    self.broken_keys.append(button)

        for x1 in range(1,len(self.dictionary)+1):
            if x1 in index_list:
                coded_buttons.append(1)
            else:
                coded_buttons.append(0)

        x=takeClosest(x,self.s['space_x'])
        y=takeClosest(y,self.s['space_y'])
        x=self.s['space_x'].index(x)
        y=self.s['space_y'].index(y)
        coded_buttons=[x]+[y]+coded_buttons

        coded_buttons=Tensor(coded_buttons).type(self.s['dtype'])

        return img, coded_buttons