from __future__ import annotations
import DetectionClass as DC
from PIL import Image

while True:
    text = input()
    try:
        print("Read Image")
        network = DC.DetectionClass()
        network.detection(file="Img/"+text+".png")
        im = Image.open("out.png")
        im.show()
        print("SHOW")
    except:
        print("ERROR")
        pass
