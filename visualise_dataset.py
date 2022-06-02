import cv2
import pickle
import numpy as np
with (open('3319.pickle', "rb")) as openfile:
            while True:
                try:
                    obj=(pickle.load(openfile))
                except EOFError:
                    break
z=np.zeros((384,384))
k=0
img=cv2.imread('3319.jpg')
print(img.shape)
for i in obj:
    # cv2.rectangle(z,(int(obj[i]['center'][0])-25, int(obj[i]['center'][1])-int(int(obj[i]['height'])/2))
    #         ,(int(obj[i]['center'][0])+25, int(obj[i]['center'][1])+int(int(obj[i]['height'])/2)),1,-1)
    cv2.circle(img, (i['center'][0], (k*384)+i['center'][1]), int(384/32), (255, 255, 255), -1)
    cv2.line(img,(int(i['center'][0]),int((k*384)+i['center'][1]-(i['height']/2))),(int(i['center'][0]),int((k*384)+i['center'][1]+(i['height']/2))),(255,255,255),5)
    k=k+1
cv2.imwrite('rs.jpg',img)
