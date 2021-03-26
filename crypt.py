#coding:utf-8
import cv2
import numpy as np
import math


def encrypt(input):
    img=input
    for i in range (1,4):
        img = transform(img,i)
    return img

def transform(img, num):

    [rows,cols] = img.shape
    if (rows == cols):
        n = rows
        img2 = np.zeros([rows, cols])

        for x in range(0, rows):
            for y in range(0, cols):
                #a=2,b=2
                img2[x][y] = img[(x+2*y)%n][(2*x+5*y)%n]
        
        return img2

    else:
        if(rows > cols):
            H=rows
        else:
            H=cols
        
        padded_img = np.zeros([H,H])
        
        padded_img[0:rows,0:cols] = img[0:rows,0:cols]
        
        img3=transform(padded_img,num)
        
        return img3

def imgreshape(details):

    h = int(''.join(filter(str.isdigit, details[0])))
    w = int(''.join(filter(str.isdigit, details[1])))


    array = np.zeros(h*w).astype(int)


    k = 0
    i = 2
    x = 0
    j = 0


    while k < array.shape[0]:

        if(details[i] == ';'):
            break

        if "-" not in details[i]:
            array[k] = int(''.join(filter(str.isdigit, details[i])))        
        else:
            array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

        if(i+3 < len(details)):
            j = int(''.join(filter(str.isdigit, details[i+3])))

        if j == 0:
            k = k + 1
        else:                
            k = k + j + 1        

        i = i + 2

    array = np.reshape(array,(h,w))
    return array

#-----------读取图片--------------------------

block_size = 8
# 取image.text
with open('imageR.txt', 'r') as myfile:
    imageR=myfile.read()
with open('imageB.txt', 'r') as myfile:
    imageB=myfile.read()
with open('imageG.txt', 'r') as myfile:
    imageG=myfile.read()


Rdetails = imageR.split()
Bdetails = imageB.split()
Gdetails = imageG.split()

# 将image.txt image
imageR=imgreshape(Rdetails)
imageB=imgreshape(Bdetails)
imageG=imgreshape(Gdetails)

#
R=encrypt(imageR)
B=encrypt(imageB)
G=encrypt(imageG)

IMG=cv2.merge([R,B,G])

cv2.imwrite('ENCRYPT.jpg', np.uint8(IMG))

a = open('akash.jpg','r')
