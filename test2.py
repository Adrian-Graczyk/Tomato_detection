import cv2
import numpy as np
from tkinter import *
from PIL import Image,ImageTk
import cmapy

def camera_view():
    Mybool = False

    def Changebool():
        global Mybool
        if Mybool == True:
            Mybool = False
            #print('false')
        else:
            Mybool = True
            #print('true')


    root = Tk()
    root.geometry("950x700")
    root.configure(bg="white")
    Label(root,text="TesTojest",font=("times new roman",20,"bold"),bg="white",fg="black").pack()
    L1 = Label(root,bg="black")
    L1.place(x = 20, y = 50)
    L2 = Label(root,bg="black")
    L2.place(x = 250, y = 100)
    cap = cv2.VideoCapture(0)
    But = Button(root,text="Change",font=("times new roman",15,"bold"),bg="lightgray",fg="black", command=Changebool)
    But.place(x = 20, y = 210)

    # Capture the current frame
    ret, frame = cap.read() 
    # Save image
    cv2.imwrite('cam.png', frame) 
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

     

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    frame = ImageTk.PhotoImage(Image.fromarray(frame))
    '''L1['image'] = frame
    img = cv2.imread('cam.png')
    imgnew = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if Mybool == True:
        #####here to add image edition img and write it to imgnew
        imgnew = cv2.applyColorMap(img, cmapy.cmap('magma'))
        #####
    imgnew = ImageTk.PhotoImage(Image.fromarray(imgnew))
    L2['image'] = imgnew'''
    cap.release()
    return frame
    

"""
while True:
    '''# Capture the current frame
    ret, frame = cap.read() 
    # Save image
    cv2.imwrite('cam.png', frame) 
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

     

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    frame = ImageTk.PhotoImage(Image.fromarray(frame))
    L1['image'] = frame
    img = cv2.imread('cam.png')
    imgnew = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if Mybool == True:
        #####here to add image edition img and write it to imgnew
        imgnew = cv2.applyColorMap(img, cmapy.cmap('magma'))
        #####
    imgnew = ImageTk.PhotoImage(Image.fromarray(imgnew))
    L2['image'] = imgnew
    root.update()'''

    L2['image'] = camera_view()
    img = cv2.imread('cam.png')
    imgnew = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if Mybool == True:
        #####here to add image edition img and write it to imgnew
        imgnew = cv2.applyColorMap(img, cmapy.cmap('magma'))
        #####
    imgnew = ImageTk.PhotoImage(Image.fromarray(imgnew))
    L2['image'] = imgnew
"""

