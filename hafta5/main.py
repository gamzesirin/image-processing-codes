import cv2
import numpy as np

# ölçekleme scale 

# i=cv2.imread("./images/peppers.png",1)
# dx=2 # x ekseninde genişletme
# dy=0.5 # y ekseninde daraltma
# height,width=i.shape[:2] # resmin yükseklik ve genişlik değerlerini alıyoruz
# M=np.float32([[dx,0,0],[0,dy,0]]) # ölçekleme matrisi
# print(M)
# i2=cv2.warpAffine(i,M,(width*2,height//2))

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# ölçekleme scale 

# i=cv2.imread("./images/peppers.png",1)
# dx=0.5 
# dy=2
# height,width=i.shape[:2]
# M=np.float32([[dx,0,0],[0,dy,0]])
# print(M)
# i2=cv2.warpAffine(i,M,(width//2,height*2))

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------------------------

# öteleme translation
# i=cv2.imread("./images/peppers.png",1)

# height,width=i.shape[:2]
# M=np.float32([[1,0,50],[0,1,100]]) # 50 sağa 100 aşağı kaydır 
# print(M)
# i2=cv2.warpAffine(i,M,(width+50,height+100)) 

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-------------------------------------------

# i=cv2.imread("./images/peppers.png",1)

# height,width=i.shape[:2]
# M=np.float32([[0.5,1,300],[0.5,-2,600]])
# # Görüntü x yönünde %50 küçülecek 
# # Görüntü y yönünde 2 kat büyüyecek ve ters çevrilecek -2
# # Görüntü (300, 600) piksel kadar ötelenecek
# print(M)
# i2=cv2.warpAffine(i,M,(width*2,height*2 )) 

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------------------------
#vize sınav soruları 


# scale i verip  affine matrisini bulup affine ile yapma

# 3 nokta çevirme 

# aşınma genleşme kodu opening closing de bak


# renk dönüşümü kodu

# görüntüyü hsv ye çevir v ye clahe uygula vs  FİNAL SORUSU ÖRNEK



#-----------------------------------------------------

# kırmızı kanalı arttırma yapmak resmi parlaklaştırır

# i=cv2.imread("./images/peppers.png",1)

# i[:,:,2]=i[:,:,2]+10


# cv2.imshow("Original",i)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# i=cv2.imread("./images/peppers.png")

# ib,ig,ir = cv2.split(i)

# i2 = cv2.merge([ir,ig,ib])
# ir = ir + 10

# i3 = cv2.merge([ir,ig,ib])

# cv2.imshow("Original",i)
# cv2.imshow("Red Channel Swapped",i2)
# cv2.imshow("Red Channel Increased",i3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

#   SCALE VERİP AFFINE MATRİSİNİ BULMAK VE AYNISINI AFFINE İLE YAPMA 2 cevabı da burada

i=cv2.imread("./images/peppers.png",1)
# dx=0.5
# dy=3
# height,width=i.shape[:2]
# M=np.float32([[dx,0,0],[0,dy,0]])
# print(M)
# i2=cv2.warpAffine(i,M,(width//2,height*3))

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #2.yol

# i3 =cv2.resize(i,None,fx=0.5,fy=3,interpolation=cv2.INTER_LINEAR) # bu şekilde de yapılabilir
# cv2.imshow("Original",i)
# cv2.imshow("Resized",i3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#-----------------------------------------------------

# 3 nokta çevirme

# i=cv2.imread("./images/peppers.png",1)
# height,width=i.shape[:2]
# pts1=np.float32([[0,0],[width-1,0],[0,height-1]])
# pts2=np.float32([[width-1,0],[0,0],[width-1,height-1]])
# M=cv2.getAffineTransform(pts1,pts2)
# print(M)
# i2=cv2.warpAffine(i,M,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Transformed",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------


# renk dönüşümü kodu

# i=cv2.imread("./images/peppers.png",1)
# i_hsv=cv2.cvtColor(i,cv2.COLOR_BGR2HSV)
# cv2.imshow("Original",i)
# cv2.imshow("HSV",i_hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------------------------

# öteleme translation

# i=cv2.imread("./images/peppers.png",1)
# height,width=i.shape[:2]
# M=np.float32([[1,0,-50],[0,1,-100]]) # 50 sola 100 yukarı kaydır
# print(M)
# i2=cv2.warpAffine(i,M,(width,height))

# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# #scle verip affine matrisini bulma ve aynısını affine ile yapma

# i=cv2.imread("./images/peppers.png",1)

# i2=cv2.resize(i,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
# cv2.imshow("Original",i)    
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 2.yol

# i=cv2.imread("./images/peppers.png",1)
# height,width=i.shape[:2]
# M=np.float32([[0.5,0,0],[0,0.5,0]]) # 0.5 kat küçültme
# print(M)
# i2=cv2.warpAffine(i,M,(width//2,height//2))
# cv2.imshow("Original",i)
# cv2.imshow("Resized",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# # dikeyde ayna görüntüsü yapma / ters döndürme

# i=cv2.imread("./images/ucgen1.png",1)
# print(i.shape)
# height,width=i.shape[:2]

# pts1=np.float32([[0,0],[width,0],[0,height]])
# pts2=np.float32([[0,height],[width,height],[0,0]])
# M=cv2.getAffineTransform(pts1,pts2)
# print(M)
# i2=cv2.warpAffine(i,M,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Transformed",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# resmi 180 derece döndürme = hem yatay hem dikeyde ayna görüntüsü yapma

# i=cv2.imread("./images/ucgen1.png",1)
# print(i.shape)
# height,width=i.shape[:2]
# center=(width//2,height//2)
# angle=180
# M=cv2.getRotationMatrix2D(center,angle,1)
# i2=cv2.warpAffine(i,M,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Rotated",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2.yol

# i1=cv2.imread("./images/ucgen1.png",1)
# print(i1.shape)
# height,width=i1.shape[:2]

# pts1=np.float32([[0,0],[0,height],[width,height]])
# pts2=np.float32([[width,height],[width,0],[0,0]])
# M=cv2.getAffineTransform(pts1,pts2)
# print(M)
# i3=cv2.warpAffine(i1,M,(width,height))
# cv2.imshow("Original",i1)
# cv2.imshow("Transformed",i3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------------

# Renk Kanallarını Ayırma

# # BGR formatındaki görüntüyü oku
# image = cv2.imread("./images/peppers.png")

# # B, G, R kanallarını ayır
# b, g, r = cv2.split(image)

# # Kanalları birleştirerek yeniden görüntü oluştur
# merged = cv2.merge([b, g, r])
# i2=cv2.merge([r,g,b])

# # Görüntüyü göster
# cv2.imshow("Blue Channel", b)
# cv2.imshow("Green Channel", g)
# cv2.imshow("Red Channel", r)
# cv2.imshow("Merged Image", merged)
# cv2.imshow("Merged Image2", i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------------------------

# image = cv2.imread("./images/peppers.png")

# height, width= image.shape[:2]

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray_3ch = cv2.merge([gray, gray, gray])

# image[:, width//2:width] = gray_3ch[:, width//2:width] # 3 kanallı gri görüntüyü renkli görüntünün sağına kopyaladım 

# cv2.imshow("Half Color - Half Grayscale", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------

# # saat yönünün tersinde döndürme

# i=cv2.imread("./images/peppers.png",1)
# height,width=i.shape[:2] 
# center = (width//2, height//2)
# m=cv2.getRotationMatrix2D(center,45,1)
# i2=cv2.warpAffine(i,m,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Rotated",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #-----------------------------------

# # saat yönünün tersinde döndürme

# i=cv2.imread("./images/peppers.png",1)
# height,width=i.shape[:2] 
# center = (width//2, height//2)
# m=cv2.getRotationMatrix2D(center,200,1)
# i2=cv2.warpAffine(i,m,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Rotated",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #-----------------------------------

# kareyi paralelogram yap

# i=cv2.imread("./images/images.png",1)
# height,width=i.shape[:2] 
# center = (width//2, height//2)
# m=cv2.getRotationMatrix2D(center,45,1)
# i2=cv2.warpAffine(i,m,(width,height))
# cv2.imshow("Original",i)
# cv2.imshow("Rotated",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -----------------------------
# bu kodu affine matrisi ile yapınca

# i=cv2.imread("./images/images.png",1)
# print(i.shape)
# height,width=i.shape[:2] 
# height_center = height//2
# width_center = width//2

# pts1=np.float32([[0,0],[width,0],[0,height]])
# pts2=np.float32([[0,height_center],[width_center,0],[width_center,height]])
# M=cv2.getAffineTransform(pts1,pts2)
# print(M)
# i2=cv2.warpAffine(i,M,(width,height))

# cv2.imshow("Original",i)
# cv2.imshow("Rotated",i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

