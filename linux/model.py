from lightning import  LightningModule
import pickle
from einops.layers.torch import Rearrange
from mamba_blocks import MambaTower,ThresholdLayer
from torch import bfloat16,float16,float32,ones,isnan,any,save,autocast,no_grad
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss,Linear,LayerNorm,Sequential
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from torch.cuda.amp import GradScaler
from torcheval.metrics import MulticlassAccuracy,MultilabelAccuracy
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import array
from os import getcwd
from dataset import MyDataset_Ultimate
from helpers import get_type
from torch import where,tensor
class Model_Ultimate(LightningModule):
    def __init__(self, s):
        super().__init__()

        self.counter=0
        self.xpoints=[]
        self.accu_points=[]
        self.loss_points = []

        s=get_type(s)

        with open(s['dictionary'], 'rb') as file:
            self.dictionary = pickle.load(file)

        s['buttons_size'] = len(self.dictionary)+len(s["space_x"])+len(s["space_y"])

        patch_dim = s['n_channels']*s['patch_size']*s['patch_size']

        self.part1 = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=s['patch_size'], p2=s['patch_size'])

        self.part2=Sequential(LayerNorm(patch_dim),
                                  Linear(patch_dim, s['dim']),
                                  LayerNorm(s['dim']))

        self.part3 = Sequential(
                                  MambaTower(
                                      s['dim'], s['n_layers'],
                                      global_pool=True, dropout=s['dropout'],
                                      dtype=s['dtype'],mamba2=True
                                  ),
                                  Linear(s['dim'], s['buttons_size']))

        self.loss_for_buttons=BCEWithLogitsLoss()
        self.optimizer = Adam(params=self.parameters(), lr=0.00001, weight_decay = 1e-5)
        self.scaler = GradScaler()
        self.scheduler = CyclicLR(self.optimizer, base_lr=0.000005, max_lr=0.00001, step_size_up=5, gamma=0.1)
        self.automatic_optimization = False
        self.button_label=MultilabelAccuracy()
        s2=s.copy()
        s2["training_dataset"]=s["validation_dataset"]
        s2['fast_training']=False
        s['dataset']=MyDataset_Ultimate(s2)
        s['dataset'].set_partition(3)

        self.test_loader = DataLoader(s['dataset'], batch_size=s['batch_size'],
                                                  shuffle=False, num_workers=5)
        self.buttons_cross_entropy=[]
        self.buttons_accuracy_metrics=[]
        self.losses_for_keys={}
        self.accuracy_for_keys={}
        self.full_dicitonary={"mouse_x":39, "mouse_y":39}|self.dictionary.copy()
        self.keys_list=list(self.full_dicitonary.keys())
        self.losses_for_keys['keys']=[]
        self.accuracy_for_keys['keys'] = []
        for key in self.keys_list:
            self.accuracy_for_keys[key]=[]
            self.losses_for_keys[key]=[]

        for c in range(len(self.full_dicitonary)):
            self.buttons_cross_entropy.append(CrossEntropyLoss().to('cuda'))
            self.buttons_accuracy_metrics.append(MulticlassAccuracy().to('cuda'))

        self.s = s

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-4*3,weight_decay=1e-5)
        return optimizer

    def calculate(self,out,target,add=True):
        x_len=len(self.s['space_x'])
        y_len=len(self.s['space_y'])
        x_target=target[:,:1]
        x=out[:, :x_len]
        y=out[:, x_len:x_len+y_len]
        y_target = target[:, 1:2]
        buttons=[]
        buttons_target=[]
        rest=out[:, x_len+y_len:]
        rest_target=target[:, 2:]
        buttons.insert(0,y)
        buttons.insert(0, x)
        buttons_target.insert(0,y_target)
        buttons_target.insert(0,x_target)
        global_loss=0
        global_acc=0

        loss1=self.loss_for_buttons(rest,rest_target)
        accu1=self.button_label.update(rest,rest_target).compute()
        global_loss+=loss1
        global_acc+=accu1

        self.losses_for_keys['keys'].append(float(loss1))
        self.accuracy_for_keys['keys'].append(float(accu1))
        for c in range(len(buttons)):
            buttons_target[c]=buttons_target[c].squeeze().long()
            loss=self.buttons_cross_entropy[c](buttons[c],buttons_target[c])
            accuracy=self.buttons_accuracy_metrics[c].update(buttons[c],buttons_target[c]).compute()
            self.buttons_accuracy_metrics[c].reset()
            if add:
                self.losses_for_keys[self.keys_list[c]].append(float(loss))
                self.accuracy_for_keys[self.keys_list[c]].append(float(accuracy))
            global_loss += loss
            global_acc += float(accuracy)
        global_acc=global_acc/(len(buttons)+1)
        return global_loss,global_acc

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        x,y=batch

        with autocast(self.s['device']):
            out = self(x)
            nans = isnan(out)
            nans = any(nans)
            nans = nans.item()
            loss,_ = self.calculate(out,y,False)
            if nans:
                print(nans,out)
                raise Exception("Nans detected!")

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.counter+=1
        self.scheduler.step()
        return loss

    @no_grad()
    def validation(self):
        self.eval()
        c=0
        losses=[]
        accuracies=[]
        for x, y in tqdm(self.test_loader, leave=True):
            x=x.to(self.s['device'])
            y=y.to(self.s['device'])
            preds = self(x)
            nans=isnan(preds)
            nans=any(nans)
            nans=nans.item()
            if nans:
                print(nans)
                print(preds)
                raise Exception("Nans detected!")
            loss,accuracy = self.calculate(preds,y)
            losses.append(loss)
            accuracies.append(accuracy)
            c+=1

        accuracy=float(sum(accuracies)/len(accuracies))*100
        loss=float(sum(losses)/len(losses))
        print(f"VALIDATION ACCURACY: {accuracy}"
              f"% LOSS: {loss}")

        self.accu_points.append(accuracy)
        self.loss_points.append(loss)
        self.train()

    def forward(self, x):
        #print(x.shape)
        x=self.part1(x)
        #print(x.shape)
        x=self.part2(x)
        #print(x.shape)
        x=self.part3(x)
        #print(x.shape)
        return x

    def on_train_epoch_end(self):

            print('SAVED!')
            save(self, getcwd() + '/models' + f'/model-{self.counter}.pth')

            self.validation()
            plt.plot(self.accu_points)
            plt.title(f"Global Accuracy {int(self.counter)}")
            plt.savefig(getcwd() + '/plots' + f'/accuracy-{self.counter}.png', dpi=1300)
            plt.close()
            plt.plot(self.loss_points)
            plt.title(f"Global Loss {self.counter}")
            plt.savefig(getcwd() + '/plots' + f'/loss-{self.counter}.png', dpi=1300)
            plt.close()

            for c in range(len(self.keys_list)):
                key=self.keys_list[c]
                plt.plot(self.accuracy_for_keys[key])
                plt.title(f"{key} Accuracy {self.counter}")
                plt.savefig(getcwd() + '/plots' + f'/{key}-accuracy-{self.counter}.png', dpi=1300)
                plt.close()
                plt.plot(self.losses_for_keys[key])
                plt.title(f"{key} Loss {self.counter}")
                plt.savefig(getcwd() + '/plots' + f'/{key}-loss-{self.counter}.png', dpi=1300)
                plt.close()


