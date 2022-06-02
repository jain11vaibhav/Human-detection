import cv2
import os
import glob
from random import shuffle
from shutil import copyfile


a=[]
for i in os.listdir('./tdataset'):
    b=glob.glob('./tdataset/'+i+'/'+'*.jpg')
    a.extend(b)

shuffle(a)
validation = a[0:int(len(a)/8)]
testing    = a[int(len(a)/8):int(len(a)/4)]
training   = a[int(len(a)/4):]

for i in validation:
    copyfile(i,'data_final/data/validation/'+i[13:])
    copyfile('./ground_truth/'+i[11:-4]+'.pickle','data_final/ground_truth/validation/'+i[13:-4]+'.pickle')
    

for i in testing:
    copyfile(i,'data_final/data/testing/'+i[13:])
    copyfile('./ground_truth/'+i[11:-4]+'.pickle','data_final/ground_truth/testing/'+i[13:-4]+'.pickle')


for i in training:
    copyfile(i,'data_final/data/training/'+i[13:])
    copyfile('./ground_truth/'+i[11:-4]+'.pickle','data_final/ground_truth/training/'+i[13:-4]+'.pickle')
