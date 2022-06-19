import cv2 
import numpy as np

def test():
    return "test"

def get_center(obj):
    return [(obj[0]+obj[2])/2, (obj[1]+obj[3])/2]

def get_radius(obj):
    return (((obj[0]-obj[2])/2)**2+((obj[1]-obj[3])/2)**2)**.5

def dist(obj, obj2):
    return ((obj[0]-obj2[0])**2+(obj[1]-obj2[1])**2)**.5

def angle(obj1, obj2):
    x = get_center(obj1)[0] - get_center(obj2)[0]
    y = get_center(obj1)[1] - get_center(obj2)[1]
    val = abs(np.angle(-y+x*1j, deg=True))
    return (180-val)/180