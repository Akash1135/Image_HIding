#coding:utf-8
import matplotlib as plt
import cv2
import numpy as np
import math
from zigzag import *


def decrypt():
    img=cv2.imread("encrypt.jpg")
    for i in range (1,8):
        img = reverse(img,i)

def reverse(img, num):

    [rows,cols,ch] = img.shape
    n = rows
    img2 = np.zeros([rows, cols,ch])

    for x in range(0, rows):
        for y in range(0, cols):

            img2[x][y] = img[(2*x-y)%n][(-1*x+y)%n]
    
    if(num == 7):
        cv2.imwrite("decrypt.jpg", img2)
    
    return img2


def decrypt(image):


    details = image.split()


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


    i = 0
    j = 0
    k = 0


    padded_img = np.zeros((h,w))

    while i < h:
        j = 0
        while j < w:        
            temp_stream = array[i:i+8,j:j+8]                
            block = inverse_zigzag(temp_stream.flatten(), int(block_size),int(block_size))            
            de_quantized = np.multiply(block,QUANTIZATION_MAT)                
            padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)        
            j = j + 8        
        i = i + 8

    #8
    padded_img[padded_img > 255] = 255
    padded_img[padded_img < 0] = 0

    return padded_img


#Q-Matrix
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
#Block Size 8
block_size = 8
# image.text
with open('imageR.txt', 'r') as myfile:
    Rimage=myfile.read()
with open('imageB.txt', 'r') as myfile:
    Bimage=myfile.read()
with open('imageG.txt', 'r') as myfile:
    Gimage=myfile.read()

R=decrypt(Rimage)
B=decrypt(Bimage)
G=decrypt(Gimage)
IMG=cv2.merge([B,G,R])

cv2.imwrite('decrypt.jpg', np.uint8(IMG))
im = plt.imread('/home/akash/Downloads/Maths Project/jpegcrypto-master/decrypt.jpg')
plt.imshow('decrypt.jpg')



