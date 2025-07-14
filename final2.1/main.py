# # geçen hafta slayt sonu ödevin kodu çözümü 
import cv2
import numpy as np

I=cv2.imread('./images/peppers.png',1)
I2=(I[:,:,0]>220) & (I[:,:,1]>220) & (I[:,:,2]>220)
print(I2.dtype)
I3=np.uint8(I2)*255
I4=cv2.morphologyEx(I3,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
indis=np.bool_(I4)
Icopy=I.copy()
Icopy[indis,0]=200
Icopy[indis,1]=214
Icopy[indis,2]=48
# 48 214 200 rgb
xy=np.column_stack(np.where(I4>0))
I_last=Icopy[np.min(xy[:,0]):np.max(xy[:,0]+1),np.min(xy[:,1]):np.max(xy[:,1]+1)]
cv2.imshow('I',I)
cv2.imshow('I3',I3)
cv2.imshow('I4',I4)
cv2.imshow('Icopy',Icopy)
cv2.imshow('I crop',I_last)
cv2.waitKey()
cv2.destroyAllWindows()