import time
from pynput import keyboard,mouse
from timeit import default_timer as timer
import pygame
import psutil
from unidecode import unidecode
import os
import xlsxwriter
import cv2
import json
import threading
from screeninfo import get_monitors
from pyautogui import click,getAllWindows
from windows_capture import WindowsCapture, Frame, InternalCaptureControl



class Program():
    def __init__(self):
        pygame.mixer.init()
        self.wcisniete=[]
        self.dziala=True
        self.x = 0
        self.y = 0
        self.fps =0
        self.time =0
        self.frames=0
        self.prawy = False
        self.lewy = False
        self.counter = 2
        self.monitor = 1
        self.curr_photo="Null"
        #   rozdzielczość w jakiej zapisuje się zdjecie
        self.start_x=1280
        self.start_y=720
        self.t1 = threading.Thread(target=self.myszka)
        self.config={
            'program':'notepad.exe',
            'title': 'Notatnik',
            'lok':os.path.dirname(os.path.realpath(__file__))+r'\ruchy',
            'monitor': 1,
            'first_person_view':False,
            'record_screen': False,
            'compress_image':True,
            'save_format':"jpg",
            'quality':75,
            'res_x': self.start_x,
            'res_y': self.start_y,
        }
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except:
            try:
                os.mkdir(self.config['lok'])
            except:
                pass
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
        self.config['lok']=self.config['lok'].strip()
        self.width=get_monitors()[self.monitor-1].width
        self.height=get_monitors()[self.monitor-1].height
        self.middle_x=self.width/2
        self.middle_y=self.height/2

        #print(self.middle_x, self.middle_y)
        if os.path.isfile("zegar.wav") ==False:
            raise Exception("I quess you don't have zegar.wav?")
        print(f"That's the resoultion of your screen, right?:\t {self.width}x{self.height}")

    def start(self):
        arr = os.listdir(self.config['lok'])
        maxik = 0
        for c in arr:
            if c.startswith("analiza-"):
                if int(c[8:]) > maxik:
                    maxik = int(c[8:])
        maxik = maxik + 1
        anal_folder = os.path.join(self.config['lok'], f"analiza-{maxik}")
        self.frame_folder = os.path.join(anal_folder, f"frames")
        os.mkdir(anal_folder)
        os.mkdir(self.frame_folder)
        self.workbook = xlsxwriter.Workbook(os.path.join(anal_folder, f"dane.xlsx"))
        self.w = self.workbook.add_worksheet()
        self.w.write(f'A1', "Filename")
        self.w.write(f'B1', "X")
        self.w.write(f'C1', "Y")
        self.w.write(f'D1', "LewyMyszek")
        self.w.write(f'E1', "PrawyMyszek")
        self.w.write(f'F1', "Przyciski")

        print(f"CZEKAM NA {self.config['program'][:-4]}")
        lock = True
        while lock:
            if self.process_exists(True):
                lock = False

        lock=True
        while lock:
            wynik = self.get_title(self.config['title'])
            if wynik is None:
                pass
                #raise Exception
            else:
                self.config['title'] = wynik
                lock = False
        print(f"NAZWA OKNA '{wynik}'")

        self.odliczanie()

    def on_scroll(self,x, y, dx, dy):
       if dy < 0:
           if 'key.scroll_down' not in self.wcisniete:
            self.wcisniete.append('key.scroll_down')
       else:
           if 'key.scroll_up' not in self.wcisniete:
            self.wcisniete.append('key.scroll_up')

    def nagrywarka(self,title):

        if self.config['record_screen']==False:
            capture = WindowsCapture(
                cursor_capture=None,
                draw_border=None,
                monitor_index=None,
                window_name=title,
            )
        else:
            capture = WindowsCapture(
                cursor_capture=None,
                draw_border=None,
                monitor_index=int(self.config['monitor']),
                window_name=None,
            )

        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):

            if self.dziala==False:
                print("Zatrzymuje nagrywanie")
                capture_control.stop()

            self.curr_photo = os.path.join(self.frame_folder, f"{self.frames}.{self.config['save_format']}")

            if self.config['compress_image']:
                cv2.imwrite(self.curr_photo, cv2.resize(frame.frame_buffer,
                                                        (self.config['res_x'], self.config['res_y'])),
                            [cv2.IMWRITE_JPEG_QUALITY, self.config['quality']])
            else:
                cv2.imwrite(self.curr_photo, cv2.resize(frame.frame_buffer,
                                                        (self.config['res_x'], self.config['res_y'])))

            self.frames += 1
            self.time = timer() - self.start_time
            self.fps = self.frames / self.time
            self.save_to_file()

        @capture.event
        def on_closed():
            self.nagrywanie_zakoncz()

        capture.start_free_threaded()

    def myszka(self):
        with mouse.Listener(
                on_move=self.on_move,
                on_click=self.on_click,on_scroll=self.on_scroll) as self.listener_m:
            self.listener_m.join()

    def process_exists(self,print1=False):
        processName = self.config['program']
        for proc in psutil.process_iter():
            try:
                if processName[:-4].lower() in proc.name().lower():
                    if print1:
                        print(f"WYKRYTO {processName[:-4]}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def on_press(self,key):
        przycisk=format(key)
        przycisk = przycisk.lower().strip()

        if przycisk.startswith(r"\\"):
            return

        if len(przycisk) == 3:
            przycisk = przycisk[1:-1]
        przycisk = unidecode(przycisk)

        if przycisk in ['#','@']:
            return
        if 'home' in przycisk:
            self.nagrywanie_zakoncz()
            return

        for p in self.wcisniete:
            if 'x0' in p:
                self.wcisniete.remove(p)
            if 'x1' in p:
                self.wcisniete.remove(p)

        if przycisk not in self.wcisniete:
            self.wcisniete.append(przycisk)

    def on_release(self,key):
        przycisk=format(key)
        przycisk=przycisk.lower().strip()
        if len(przycisk)==3:
            przycisk = przycisk[1:-1]
        try:
            self.unidecode = unidecode(przycisk)
            przycisk = self.unidecode
            self.remove = self.wcisniete.remove(przycisk)
        except:
            pass

    def get_title(self,x):
        x=x.lower()
        x=x.strip()
        for c in getAllWindows():
            d=c.title
            if x in d.lower().strip():
                return d.strip()

    def nagrywanie_zakoncz(self):
        print("KONIEC")
        self.workbook.close()
        self.dziala=False
        self.listener_m.stop()
        self.listener_k.stop()

    def save_to_file(self):

        self.x1 = int(self.x)
        self.y1 = int(self.y)

        self.w.write(f'A{self.counter}', self.curr_photo)
        self.w.write(f'B{self.counter}', self.x1)
        self.w.write(f'C{self.counter}', self.y1)
        self.w.write(f'D{self.counter}', self.lewy)
        self.w.write(f'E{self.counter}', self.prawy)
        self.w.write(f'F{self.counter}', str(self.wcisniete))
        self.counter += 1
        if self.counter%10==0:
            print(f"{self.x1}  {self.y1} L={self.lewy} P={self.prawy} P={self.wcisniete} T={round(self.time,2)} FPS={round(self.fps,2)}")
        if 'key.scroll_down' in self.wcisniete:
            self.wcisniete.remove('key.scroll_down')
        if 'key.scroll_up' in self.wcisniete:
            self.wcisniete.remove('key.scroll_up')

        self.x = 0
        self.y = 0

    def on_move(self,x, y):

        self.x+=x-self.middle_x
        self.y+=y-self.middle_y

    def on_click(self,x, y, button, pressed):

        if pressed:
            if "eft" in str(button):
                self.lewy=True
            else:
                self.prawy=True
        else:
            if "eft" in str(button):
                self.lewy=False
            else:
                self.prawy=False

    def odliczanie(self):
        pygame.mixer.music.load("zegar.wav")
        pygame.mixer.music.play(loops=0)
        time.sleep(4.5)
        click()
        self.start_time=timer()
        self.listener_k= keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener_k.start()
        self.t1.start()
        self.nagrywarka(self.config['title'])


if __name__ == "__main__":

    t = Program()

    t.start()

    #while t.dziala:
        #   rozwiązanie typu ZJEBANE
        #print(end="")
