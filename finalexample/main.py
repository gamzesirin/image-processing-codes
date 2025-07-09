import numpy as np
import cv2
import matplotlib.pyplot as plt

# I=cv2.imread('images/cicek.jpg',1)
# I2=(I[:,:,2]>150) & (I[:,:,0]<150) & (I[:,:,1]<150)
# I2=np.uint8(I2*255)
# I3=cv2.medianBlur(I2,11)
# I4=cv2.dilate(I3,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
# I5,I6=I.copy(),I.copy()
# I5[I4>0]=0  # nesne silindi arkaplan kaldı
# I6[I4==0]=0 # arkaplan silindi nesne kaldı
# kernel1=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# arkaplan=cv2.GaussianBlur(I5,ksize=(9,9),sigmaX=0,sigmaY=0)
# nesne=cv2.filter2D(I6,-1,kernel1)
# arkaplan[I4>0]=0  # nesne bölgesi tekrar silindi
# nesne[I4==0]=0 # arkaplan bölgesi tekrar silindi
# I_last=arkaplan+nesne
# cv2.imshow('I',I)
# cv2.imshow('I2',I2)
# cv2.imshow("i3",I3)
# cv2.imshow("i4",I4)
# cv2.imshow("i5",I5)
# cv2.imshow("i6",I6)
# cv2.imshow('arkaplan',arkaplan)
# cv2.imshow('nesne',nesne)
# cv2.imshow('I_last',I_last)
# cv2.waitKey()
# cv2.destroyAllWindows()

#-----------------------------------------

## EN BÜYÜK PARAYI KIRMIZI YAPMA

# i = cv2.imread('./images/coins.png',0)
# thresh,i2 = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY ) # i2 resmini alır ve threshold uygular  siyah ve beyaz oldu tamamen
# h, w = i.shape[:2]
# mask= np.zeros((i.shape[0]+2, i.shape[1]+2), np.uint8)
# i3=i2.copy()
# cv2.floodFill(i3, mask, (0,0), 255)

# i4 = cv2.bitwise_not(i3) # 255-i ile aynı işlemi yapar resmin siyah beyanızı ters çevirir
# i5=(i2)|(i4) # i2 ve i4 resimlerini birleştirir

# sayi,i6,stats,centroids = cv2.connectedComponentsWithStats(i5)

# alan = stats[1:,4] # alanı alır
# yer = np.argmax(alan) + 1 # en büyük alanın indeksini alır

# indis = np.bool_(i6 == yer) # en büyük alanın indeksini alır
# i_bgr= np.zeros((i.shape[0],i.shape[1],3),np.uint8) # 3 kanallı bir resim oluşturur
# i_bgr[:,:,0]= i_bgr[:,:,1]=i_bgr[:,:,2]=i 
# i_bgr[indis,0]=0
# i_bgr[indis,1]=0
# i_bgr[indis,2]=255 # i5 resminde en büyük alanı kırmızı yapar

# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4', i4)
# cv2.imshow('i5', i5)
# cv2.imshow('i_bgr', i_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------
# # KÜÇÜK PARAYI YEŞİL YAPMA

# i = cv2.imread('./images/coins.png',0)
# thresh,i2 = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY ) # i2 resmini alır ve threshold uygular  siyah ve beyaz oldu tamamen
# h, w = i.shape[:2]
# mask= np.zeros((i.shape[0]+2, i.shape[1]+2), np.uint8)
# i3=i2.copy()
# cv2.floodFill(i3, mask, (0,0), 255)
# i4 = cv2.bitwise_not(i3) # 255-i ile aynı işlemi yapar resmin siyah beyanızı ters çevirir
# i5=(i2)|(i4) # i2 ve i4 resimlerini birleştirir
# sayi,i6,stats,centroids = cv2.connectedComponentsWithStats(i5)
# alan = stats[1:,4] # alanı alır
# yer = np.argmin(alan) + 1 # en küçük alanın indeksini alır
# indis = np.where(i6 == yer) # en küçük alanın indeksini alır
# i_bgr= np.zeros((i.shape[0],i.shape[1],3),np.uint8) # 3 kanallı bir resim oluşturur
# i_bgr[:,:,0]= i_bgr[:,:,1]=i_bgr[:,:,2]=i
# i_bgr[indis]=[0,255,0] # i5 resminde en küçük alanı siyah yapar
# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4', i4)
# cv2.imshow('i5', i5)
# cv2.imshow('i_bgr', i_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#----------------------------------

## KALIN KENARDAN İNCE KENAR ÇIKARMA İÇ İÇE HALKA SORUSUNA BAK   

# i = cv2.imread('./images/blobs.jpg',0)
# i2 = cv2.bitwise_not(i)
# i3 = cv2.Canny(i2, 100, 200)
# i4 = cv2.dilate(i3, np.ones((5, 5), np.uint8), iterations=1) # 
# i5 = cv2.erode(i4, np.ones((5, 5), np.uint8), iterations=1)

# i6 = i4-i5
# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4', i4)
# cv2.imshow('i5', i5)
# cv2.imshow('i6', i6)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 2. yol

# i = cv2.imread('./images/blobs.jpg',0)
# i2 = cv2.bitwise_not(i)
# i3_x= cv2.Sobel(i2, cv2.CV_64F, 1, 0, ksize=5)
# i3_y= cv2.Sobel(i2, cv2.CV_64F, 0, 1, ksize=5)
# i3_xy = cv2.Sobel(i2, cv2.CV_64F, 1, 1, ksize=5)
# i3= np.uint8(np.sqrt(i3_x**2 + i3_y**2))
# i4 = cv2.dilate(i3, np.ones((5, 5), np.uint8), iterations=1) # 
# i5 = cv2.erode(i4, np.ones((5, 5), np.uint8), iterations=1)

# i6 = i4-i5
# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i4', i4)
# cv2.imshow('i5', i5)
# cv2.imshow('i6', i6)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

## resmi 1 ile okutup adamın ceketini kırmızı yapma

# i = cv2.imread('./images/cameraman.tif',1)
# i2 = cv2.calcHist([i], [0], None, [256], [0, 256])
# print(np.sum(i==17))
# i[(i[:, :, 0] >= 8) & (i[:, :, 0] <= 17),0]=0
# i[(i[:, :, 1] >= 8) & (i[:, :, 1] <= 17),1]=0
# i[(i[:, :, 2] >= 8) & (i[:, :, 2] <= 17),2]=255
# cv2.imshow("Original",i) # resmi ekrana bastırdım
# plt.plot(i2) # histogramı çizdim
# plt.show()
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım

#----------------------------------
## Resmi -1 ile okutup adamın ceketini kırmızı yapma

# i = cv2.imread('./images/cameraman.tif',-1)
# i2 = cv2.calcHist([i], [0], None, [256], [0, 256])
# print(np.sum(i==17))
# mask = (i >= 8) & (i <= 17)

# i_bgr=np.zeros((i.shape[0],i.shape[1],3),np.uint8)
# i_bgr[:,:,0]= i_bgr[:,:,1]=i_bgr[:,:,2]=i
# i_bgr[mask] = [0, 0, 255] # i5 resminde en büyük alanı kırmızı yapar
# cv2.imshow("Original",i) # resmi ekrana bastırdım
# cv2.imshow("i_bgr",i_bgr) # resmi ekrana bastırdım
# plt.plot(i2) # histogramı çizdim

# plt.show()
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım

#----------------------------------
##Resmi -1 ile okutup adamın ceketini beyaz yapma

# i = cv2.imread('./images/cameraman.tif',-1)
# i2 = cv2.calcHist([i], [0], None, [256], [0, 256])
# print(np.sum(i==17))
# mask = (i >= 8) & (i <= 17)
# i[mask] = 255
# cv2.imshow("Original",i) # resmi ekrana bastırdım
# plt.plot(i2) # histogramı çizdim
# plt.show()
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım

#----------------------------------

#Ağaçtaki vişne sayısını bulma

## 1. yol
# #Görüntüyü oku ve HSV'ye çevir
# img = cv2.imread('./images/VisneFidani.jpeg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Kırmızı renk için iki aralıkta maske oluştur
# lower1 = np.array([0, 120, 70])
# upper1 = np.array([10, 255, 255])
# lower2 = np.array([170, 120, 70])
# upper2 = np.array([180, 255, 255])
# mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

# # Gürültü temizliği (opening + closing)
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# # Nesne sayımı
# _, labels = cv2.connectedComponents(mask)
# print("Toplam vişne sayısı:", _ - 1)  # Arka plan hariç

# # Görüntüleri göster
# cv2.imshow("Maske", mask)
# cv2.imshow("Orijinal", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

### 2. yol

# # Görüntüyü oku ve HSV'ye çevir
# img = cv2.imread('./images/VisneFidani.jpeg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # HSV kanallarını ayır
# h, s, v = cv2.split(hsv)

# mask = np.zeros(h.shape, dtype=np.uint8)
# cond1 = (h >= 0) & (h <= 10) & (s >= 120) & (v >= 70)
# cond2 = (h >= 170) & (h <= 180) & (s >= 120) & (v >= 70)
# mask[cond1 | cond2] = 255

# # Gürültü temizliği (Opening + Closing)
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# # Nesne sayımı
# num_labels, _ = cv2.connectedComponents(mask)
# print("Toplam vişne sayısı:", num_labels - 1)  # Arka plan hariç

# # Görüntüleri göster
# cv2.imshow("Maske (np.zeros)", mask)
# cv2.imshow("Orijinal", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------
## 3. yol  içini doldurma ve saydırma

img = cv2.imread('./images/VisneFidani.jpeg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Maske oluştur (cv2.inRange yerine np.zeros ile)
mask = np.zeros(h.shape, dtype=np.uint8)
cond1 = (h >= 0) & (h <= 10) & (s >= 120) & (v >= 70)
cond2 = (h >= 170) & (h <= 180) & (s >= 120) & (v >= 70)
mask[cond1 | cond2] = 255

# Gürültü temizliği (Opening)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# floodFill için kopya al, 2 piksel genişliğinde border eklenmeli
flooded = mask.copy()
h, w = flooded.shape
mask_ff = np.zeros((h + 2, w + 2), np.uint8)  # floodFill maskesi 2 piksel büyük olmalı

# Floodfill işlemi – arka plandan başlayarak dışarıyı doldur
cv2.floodFill(flooded, mask_ff, seedPoint=(0, 0), newVal=255)

# floodfill ile dışı dolduruldu, şimdi içi bulmak için tersini al
flooded_inv = cv2.bitwise_not(flooded)

# mask + iç = tamamen dolu nesneler
filled_objects = cv2.bitwise_or(mask, flooded_inv)

# Nesne sayımı
num_labels, _ = cv2.connectedComponents(filled_objects)
print("Toplam vişne sayısı:", num_labels - 1)

# Görüntüleri göster
cv2.imshow("Orijinal", img)
cv2.imshow("Maske", mask)
cv2.imshow("İçi Doldurulmuş", filled_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()


#----------------------------------

# # kenar bulup kırmızı yapma

# i = cv2.imread('./images/blobs.jpg',0)
# i2=cv2.bitwise_not(i)
# i3 = cv2.Canny(i2, 100, 200)

# i_bgr= np.zeros((i.shape[0], i.shape[1],3), np.uint8)
# i_bgr[:,:,0]= i_bgr[:,:,1]=i_bgr[:,:,2]=i
# indis=np.bool_(i3)
# i_bgr[indis,0]=0
# i_bgr[indis,1]=0
# i_bgr[indis,2]=255 
# # i_bgr[i3>0]=[0,0,255]

# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)
# cv2.imshow('i_bgr', i_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

## kaç tane nesne var 

# i = cv2.imread('./images/blobs.jpg',0)
# i2=cv2.bitwise_not(i)
# thresh,i3 = cv2.threshold(i2, 127, 255, cv2.THRESH_BINARY ) # i2 resmini alır ve threshold uygular  siyah ve beyaz oldu tamamen
# i4=i3.copy()
# sayi,i5,stats,centroids = cv2.connectedComponentsWithStats(i3)
# print("Toplam nesne sayısı:", sayi-1)  # Arka plan hariç
# cv2.imshow('i', i)
# cv2.imshow('i2', i2)
# cv2.imshow('i3', i3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

## içini doldurma 

# # Görüntüyü gri olarak oku
# img = cv2.imread('./images/blobs.jpg', 0)

# # Renkli görüntü oluştur
# color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # Görüntüyü ters çevir
# img2 = cv2.bitwise_not(img)


# # Siyah-beyaz hale getir
# _, binary = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# # Bağlı bileşenleri bul
# num_labels, labels = cv2.connectedComponents(binary)

# # Her nesnenin maskesini kırmızı yap
# for label in range(1, num_labels):
#     color_img[labels == label] = [0, 0, 255]  # BGR kırmızı

# # Sonucu göster
# cv2.imshow("Kirmizi Doldurulmus Nesneler", color_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2. yol 

# i = cv2.imread('./images/blobs.jpg', 0)
# i2=cv2.bitwise_not(i)
# thresh,i3 = cv2.threshold(i2, 127, 255, cv2.THRESH_BINARY ) 
# i4=i3.copy()
# mask= np.zeros((i.shape[0]+2, i.shape[1]+2), np.uint8)
# cv2.floodFill(i4, mask, (0,0), 255) 
# i5=cv2.bitwise_not(i4) 
# i6=(i3)|(i5) 
# indis= np.bool_(i6)
# ibgr=cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)
# ibgr[indis]= [0,0,255]

# cv2.imshow('ibgr', ibgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
#----------------------------------

## closing ile içini doldurma 

# img = cv2.imread('./images/blobs.jpg', 0)

# img2 = cv2.bitwise_not(img)
# # Binary yap (objeler beyaz, arka plan siyah)
# _, binary = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# # Morfolojik closing uygulamak için kernel tanımla
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# # Closing işlemi
# closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# # Renkli görüntü oluştur
# color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # Kapalı (closed) alanları kırmızı ile boya
# color_img[closed == 255] = (0, 0, 255)

# cv2.imshow("Closing ile Doldurulmus Nesneler", color_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

## floodfill ile içini doldurma <<<<<<<<<< BAK
