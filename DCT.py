#coding:utf-8
import cv2
import numpy as np
import math
from zigzag import *

#--------------------------------------------------------------------------
def get_run_length_encoding(img_arr):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    img_arr = img_arr.astype(int)

    while i < img_arr.shape[0]:
        #img_arr.shape[0]
        if img_arr[i] != 0:            
            stream.append((img_arr[i],skip))
            bitstream = bitstream + str(img_arr[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream

#------------------------------------------------------------------------------------


block_size = 8
# Q-Matrix
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

img = cv2.imread('akash.jpg')

(B, G, R) = cv2.split(img)

#-----------------------------------------------------------------------------------


[h , w , ch] = img.shape

height = h
width = w
h = np.float32(h) 
w = np.float32(w) 

#math.ceil上
nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)
nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)

#
H =  block_size * nbh
W =  block_size * nbw

#
padded_R = np.zeros((H,W))
padded_G = np.zeros((H,W))
padded_B = np.zeros((H,W))

#将imgpadded_img [0：h，0：w]
padded_R[0:height,0:width] = R[0:height,0:width]
padded_G[0:height,0:width] = G[0:height,0:width]
padded_B[0:height,0:width] = B[0:height,0:width]

#
# cv2.imwrite('uncompressed1.bmp', np.uint8(padded_R))
# cv2.imwrite('uncompressed2.bmp', np.uint8(padded_B))
# cv2.imwrite('uncompressed3.bmp', np.uint8(padded_G))

#----------------分块----------------------------
# bitstreamR=""
# bitstreamB=""
# bitstreamG=""
for i in range(nbh):
    

        row_ind_1 = i*block_size                
        row_ind_2 = row_ind_1+block_size
        
        for j in range(nbw):
            

            col_ind_1 = j*block_size                       
            col_ind_2 = col_ind_1+block_size


            blockR = padded_R[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            blockB = padded_B[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            blockG = padded_G[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
                   

            DCTR = cv2.dct(blockR)
            DCTB = cv2.dct(blockB)
            DCTG = cv2.dct(blockG)


            DCTR_normalized = np.divide(DCTR,QUANTIZATION_MAT).astype(int)            
            DCTB_normalized = np.divide(DCTB,QUANTIZATION_MAT).astype(int)            
            DCTG_normalized = np.divide(DCTG,QUANTIZATION_MAT).astype(int)            
            
            #通过调用zigzag
            reorderedR = zigzag(DCTR_normalized)
            reorderedB = zigzag(DCTB_normalized)
            reorderedG = zigzag(DCTG_normalized)
            # str1 = ','.join(str(i) for i in reorderedR)
            # file1 = open("zigzag.txt","a+")
            # file1.write(str1)
            # file1.close()

            # bitstreamR = bitstreamR + get_run_length_encoding(reorderedR)
            # bitstreamB = bitstreamB + get_run_length_encoding(reorderedB)
            # bitstreamG = bitstreamG + get_run_length_encoding(reorderedG)

            #8*8
            reshapedR= np.reshape(reorderedR, (block_size, block_size))
            reshapedB= np.reshape(reorderedB, (block_size, block_size))
            reshapedG= np.reshape(reorderedG, (block_size, block_size))

            #padded_img
            padded_R[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedR   
            padded_B[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedB
            padded_G[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedG                     

# file1 = open("bitstream.txt","w")
# file1.write(bitstreamR)
# file1.close()
# #输出预处理后的灰度图像
# cv2.imwrite('uncompressed1.bmp', np.uint8(padded_R))
# cv2.imwrite('uncompressed2.bmp', np.uint8(padded_B))
# cv2.imwrite('uncompressed3.bmp', np.uint8(padded_G))

#----------------------编码-----------------------------------------------------------

#将量化后的padded_img转为一维数组
arrangedR = padded_R.flatten()
arrangedB = padded_B.flatten()
arrangedG = padded_G.flatten()

# #输出arrangedR
# str1 = ','.join(str(i) for i in arrangedR)
# file1 = open("arr.txt","w")
# file1.write(str1)
# file1.close()

#-----------------游程编码/压缩------------------------------------------------------------------------------

bitstreamR = get_run_length_encoding(arrangedR)
bitstreamB = get_run_length_encoding(arrangedB)
bitstreamG = get_run_length_encoding(arrangedG)

#
bitstreamR = str(padded_R.shape[0]) + " " + str(padded_R.shape[1]) + " " + bitstreamR + ";"
bitstreamB = str(padded_B.shape[0]) + " " + str(padded_B.shape[1]) + " " + bitstreamB + ";"
bitstreamG = str(padded_G.shape[0]) + " " + str(padded_G.shape[1]) + " " + bitstreamG + ";"

#image.txt
file1 = open("imageR.txt","w")
file1.write(bitstreamR)
file1.close()
file1 = open("imageB.txt","w")
file1.write(bitstreamB)
file1.close()
file1 = open("imageG.txt","w")
file1.write(bitstreamG)
file1.close()
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Completed ")
