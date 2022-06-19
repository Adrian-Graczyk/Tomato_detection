# napisz to w pliku głównym:
# from JsonToPng import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pylab
import cv2
import numpy

"""

W hex pliku pikseli idą od końcowych do początkowych i jak będę wstawiąć ich kolejno po prostu na pierwsze miejsce w liście,
zajmie to dużo czasu, również colormap jest BGR zamiast RGB więc konwertuje to w "odwrotny" text.png plik, otwieram go, zamieniam colormap i obracam o 180 stopni
i odzwierciedlam go. ładuje do pliku "input.png"

"""


# width = 848
# height = 480

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))


def Load_Png_From_Hex(file_name):
    with open(file_name) as f:
        Rec = []
        Recon = []
        EndRec = []
        data = f.read(6)
        for h in range(height):
            for w in range(width):
                data = f.read(6)
                if not data:
                    break  
                rgbdata = hex_to_rgb(data)
                if w < 18:
                    EndRec.append(rgbdata)  
                else:
                    Recon.append(rgbdata)
            Recon.extend(EndRec)
            EndRec.clear()

    arr = numpy.array(Recon)
    pylab.figure(num=None, figsize=(10, 6))
    image = plt.imshow(np.array(arr).reshape((height, width, 3)))
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')

    #rotate image by 180 degrees
    im = Image.open("test.png")
    angle = 180
    out = im.rotate(angle, expand=True)   
    out.save('input.png')
    img = cv2.imread('input.png')
    img = cv2.flip(img, 1)
    imgnew = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('input.png', imgnew)
    print("input.png image loaded")

def Create_Png_From_Hex(hex_string):
    width = 848
    height = 480
    Rec = []
    Recon = []
    EndRec = []
    n = 6
    k=0
    list: Any = [hex_string[i:i + n] for i in range(0, len(hex_string), n)]
    for h in range(height):
        for w in range(width):
            data = list[k]
            k=k+1
            if not data:
                break
            rgbdata = hex_to_rgb(data)
            if w < 18:
                EndRec.append(rgbdata)
            else:
                Recon.append(rgbdata)
        Recon.extend(EndRec)
        EndRec.clear()

    arr = numpy.array(Recon)
    pylab.figure(num=None, figsize=(10, 6))
    image = plt.imshow(np.array(arr).reshape((height, width, 3)))
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')

    # rotate image by 180 degrees
    im = Image.open("test.png")
    angle = 180
    out = im.rotate(angle, expand=True)
    out.save('input.png')
    img = cv2.imread('input.png')
    img = cv2.flip(img, 1)
    imgnew = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('input.png', imgnew)
    print("input.png image loaded")

###!!! Zakomentuj funkcję i użyj ją w pliku głównym


#Create_Png_From_Hex(hex)

#file_name = 'json_bmp_c.json'
#Load_Png_From_Hex(file_name)