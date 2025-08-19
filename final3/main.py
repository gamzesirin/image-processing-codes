import cv2
import numpy as np


# I=cv2.imread('./images/coins_filled.jpg',0)
# thresh,I2=cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(thresh)
# sayi,I3,stats,centroid=cv2.connectedComponentsWithStats(I2)
# print('sayi=',sayi)
# print('stats=',stats)
# print('centroid=',centroid)
# print(np.max(I3))
# # I4=np.uint8(I3==1)*255
# # I5=np.uint8(I3==2)*255
# for i in range(1,sayi):
#     cv2.imshow('I',np.uint8(I3 == i) * 255)
#     cv2.waitKey(1000)

# cv2.imshow('I',I)
# cv2.imshow('I2',I2)
# # cv2.imshow('I4',I4)
# # cv2.imshow('I5',I5)
# cv2.waitKey()
# cv2.destroyAllWindows()

#----------------------------------------

# import cv2
# import numpy as np

# i = cv2.imread('./images/blobs.jpg', 0)
# i2= cv2.bitwise_not(i) # 255-i ile aynı işlemi yapar resmin siyah beyanızı ters çevirir
# thresh,i3 = cv2.threshold(i2, 127, 255, cv2.THRESH_BINARY ) # i2 resmini alır ve threshold uygular  siyah ve beyaz oldu tamamen 

# i4_x = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=1, dy=0 , ksize=5) # x yönünde kenar bulma işlemi yapar
# i4_y = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=0, dy=1 , ksize=5) # x yönünde kenar bulma işlemi yapar
# i4_xy = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=1, dy=1 , ksize=5) # x ve y yönünde kenar bulma işlemi yapar

# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4_x', np.uint8(np.abs(i4_x)) )
# cv2.imshow('i4_y', np.uint8(np.abs(i4_y)))
# cv2.imshow('i4_xy', np.uint8(np.abs(i4_xy)))

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------

# import cv2
# import numpy as np

# i = cv2.imread('./images/blobs.jpg', 0)
# i2= cv2.bitwise_not(i) # 255-i ile aynı işlemi yapar resmin siyah beyanızı ters çevirir
# thresh,i3 = cv2.threshold(i2, 127, 255, cv2.THRESH_BINARY ) # i2 resmini alır ve threshold uygular  siyah ve beyaz oldu tamamen 

# i4_x = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=1, dy=0 , ksize=5) # x yönünde kenar bulma işlemi yapar
# i4_y = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=0, dy=1 , ksize=5) # x yönünde kenar bulma işlemi yapar
# i4_xy = cv2.Sobel(i3, ddepth=cv2.CV_64F , dx=1, dy=1 , ksize=5) # x yönünde kenar bulma işlemi yapar

# aaa= np.sqrt(i4_x**2 + i4_y**2) # x ve y yönündeki kenarları toplar

# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4_x', np.uint8(np.abs(i4_x)) )
# cv2.imshow('i4_y', np.uint8(np.abs(i4_y)))
# cv2.imshow('i4_xy', np.uint8(np.abs(i4_xy)))
# cv2.imshow('aaa',np.uint8(aaa))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------

## BAKK FİNAL ÖRNEK 

# import cv2
# import numpy as np

# i = cv2.imread('./images/blobs.jpg', 0)
# i2= cv2.bitwise_not(i)

# i_canny = cv2.Canny(i2, threshold1=100, threshold2=200) # Canny kenar bulma işlemi yapar

# indis2 = np.bool_(i_canny)
# # i6=i.copy()
# # i6[indis2] = 190 

# i6 = np.zeros((i2.shape[0], i2.shape[1], 3), dtype=np.uint8) # i6 resmini oluşturur
# i6[:,:,0]=i6[:,:,1]=i6[:,:,2]=i2 # i6 resminin 3 kanalını i2 resminin 3 kanalına kopyalar
# i6[indis2,0] = 0 # i_canny resminin 255 olan yerlerini i6 resminin 255 olan yerlerine kopyalar
# i6[indis2,1] = 0 
# i6[indis2,2] = 255


# _, binary = cv2.threshold(i2, 127, 255, cv2.THRESH_BINARY)


# sayi, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# # 9. Her nesnenin merkezinden floodFill ile içini doldur
# for i in range(1, sayi):  # 0 = arka plan, o yüzden 1'den başla
#     x, y = int(centroids[i][0]), int(centroids[i][1])
#     mask = np.zeros((i2.shape[0] + 2, i2.shape[1] + 2), np.uint8)
#     cv2.floodFill(i6, mask, seedPoint=(x, y), newVal=(255, 0, 0))


# cv2.imshow("i", i)
# cv2.imshow('i2', i2)
# cv2.imshow('i_canny', i_canny)
# cv2.imshow('i6', i6)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------

## BAKK FİNAL ÖRNEK

import cv2
import numpy as np

i = cv2.imread('./images/coins.png', 1)
thresh, i2 = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY )

i3=i2[:,:,0].copy()
mask = np.zeros((i3.shape[0]+2, i3.shape[1]+2), dtype=np.uint8) # maskeyi oluşturur
cv2.floodFill(i3, mask, seedPoint=(0, 0), newVal=255 ) # flood fill işlemi yapar
i4 = cv2.bitwise_not(i3) # i3 resminin tersini alır
i5 = (i2[:,:,0]) | (i4) # i2 resminin 0. kanalını i4 resminin 0. kanalı ile toplar


sayi , i6, stats,centroid = cv2.connectedComponentsWithStats(i5) # etiketleme işlemi yapar

maxx = np.max(stats[1:, 4]) # en büyük alanı bulur
print(maxx) # en büyük alanı yazdırır

yer = np.argmax(stats[1:, 4]) + 1 # en büyük alanın indeksini bulur
print(yer) # en büyük alanın indeksini yazdırır

i7 =np.uint8(i6==yer)*255

i8 =cv2.Canny(i7, 100, 200) # Canny kenar bulma işlemi yapar
indis = np.bool_(i8) # i8 resminin 255 olan yerlerini bulur
i_rgb = i.copy() # i resmini kopyalar
i_rgb[indis, 0] = 0 # i8 resminin 255 olan yerlerini i resminin 255 olan yerlerine kopyalar
i_rgb[indis, 1] = 0 # i8 resminin 255 olan yerlerini i resminin 255 olan yerlerine kopyalar
i_rgb[indis, 2] = 255 # i8 resminin 255 olan yerlerini i resminin 255 olan yerlerine kopyalar

cv2.imshow('i', i) 
cv2.imshow('i2', i2) 
cv2.imshow('i3', i3)
cv2.imshow('i4', i4)
cv2.imshow('i5', i5) # i2 ve i4 resimlerini birleştirir
# cv2.imshow('i6', i6) # etiketlenmiş resmi gösterir
cv2.imshow('i7', i7) # en büyük alanı gösterir
cv2.imshow('i8', i8) # etiketlenmiş resmi gösterir
cv2.imshow('i_rgb', i_rgb) # kenarları kırmızı yapar
cv2.waitKey(0)
cv2.destroyAllWindows()