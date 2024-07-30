import socket
import pickle
import time
server_address = ('192.168.2.5', 12345)
import pyautogui
import mouse
import win32api
import win32.lib.win32con as win32con
c=0

buttons=[]

while True:

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

        sock.bind(server_address)

        serialized_data, client_address = sock.recvfrom(4096)

        data = pickle.loads(serialized_data)

        """


        x=int(data[0]*2)
        y=int(data[1]*2)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y*(-1), 0, 0)
        print(x,y,data[2] > 2)
        if data[2] > 2:
            pyautogui.click()


        
        if data[2]>0:
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()
        """
        #print(data)
        x=data[0]
        y=data[1]

        if len(data[2]) !=0:
            pyautogui.click()
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x*2, y * (-1)*2, 0, 0)
        """
        
        for button in buttons:
            if button not in data[2]:
                #pyautogui.keyUp(button)
                buttons.remove(button)

        for button in data[2]:
            if button not in buttons:
                #pyautogui.keyDown(button)
                buttons.append(button)

        """
        print(x,y,len(data[2]) !=0)
        c+=1
