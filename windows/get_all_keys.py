from pynput import keyboard
from unidecode import unidecode
import pickle


wcisniete={}

def dodaj(x):
    wcisniete[x] =len(wcisniete)+1


def on_press(key):
    przycisk = format(key)
    przycisk = przycisk.lower().strip()

    if przycisk.startswith(r"\\"):
        return
    if r"\\" in przycisk:
        return

    if len(przycisk) == 3:
        przycisk = przycisk[1:-1]
    przycisk = unidecode(przycisk)

    if przycisk in ['#', '@']:
        return

    for p in wcisniete:
        if 'x0' in p:
            wcisniete.remove(p)
        if 'x1' in p:
            wcisniete.remove(p)

    if przycisk not in wcisniete:
        dodaj(przycisk)
    print(wcisniete)



#dodaj('key.right_mouse_button')
dodaj('key.left_mouse_button')
#dodaj('key.scroll_down')
#dodaj('key.scroll_up')
#dodaj('key.esc')
#dodaj('key.f3')

k=keyboard.Listener(
            on_press=on_press)
k.start()
while 'key.home' not in wcisniete.keys():
    pass

dbfile = open('counter_strike2.obj', 'wb')
pickle.dump(wcisniete, dbfile)
dbfile.close()