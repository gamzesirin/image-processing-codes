import cv2 #open cv import ettim
import numpy as np #numpy import ettim
import matplotlib.pyplot as plt #matplotlib import ettim
## uint8 fromatında 0 ile 255 arasında değer alır 255 den fazla olunca yanlış değerler geliyor

## resme 128 ekleyince resim parlaklaşır çıkarınca karanlıklaşır 

i=cv2.imread("./images/cameraman.tif",1) # resmi okudum
# print(i.shape) # resmin boyutunu yazdırdım

# print(i[70,70])#14 14 14 köşegen

# # MANTIK
# deneme1=np.array([0,254,255],dtype=np.uint8) # 0 254 255 değerlerini uint8 formatında bir diziye atadım
# deneme1 = deneme1-1  # 0 olan 255 oldu 
# deneme1 = deneme1+1  # 255 olan 1 oldu
# deneme1 = deneme1+ 2 # 0: 2 oldu 254: 0 oldu 255: 1 oldu
# deneme1 = np.int32(deneme1)+2 # deneme1 dizisini int32 formatına dönüştürdüm
# deneme1[deneme1 < 0 ]=0
# deneme1 [deneme1 >255 ]=255
# deneme1= np.uint8(deneme1) # deneme1 dizisini uint8 formatına dönüştürdüm
# print(deneme1)


# cv2.imshow("Original",i) # resmi ekrana bastırdım
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım

#------------------------------------------

# i2 = i.copy() # i2 değişkenine i değişkeninin kopyasını atadım
# i3=i.copy()
# i2=np.int16(i2)+50
# i2[i2 < 0 ]=0
# i2 [i2 >255 ]=255
# i2=np.uint8(i2) # i2 dizisini uint8 formatına dönüştürdüm

# i3 = np.int16(i3)-50 
# i3[i3 < 0 ]=0
# i3 [i3 >255 ]=255
# i3=np.uint8(i3) # i3 dizisini uint8 formatına dönüştürdüm

# cv2.imshow("Original",i) # resmi ekrana bastırdım
# cv2.imshow("Resized1",i2) # resmi ekrana bastırdım
# cv2.imshow("Resized",i3) # resmi ekrana bastırdım
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım


# #-----------------------------------------------------



## NEGATİF ALMA TÜMLEYEN ALMA GİBİ DÜŞÜN
# BEYAZ KAĞIT ÜSTÜNE SYAH KALEMLE ÇİZİM YAPTIK SİYAHLAR 0 BUNLARI BİR GİBİ YAPIP İŞLEM YAPMAM GEREK BU NEDENLE TÜLEYEN ALMA YAPILIR

# i=cv2.imread("./images/cameraman.tif",1) # resmi okudum
# i4= i.copy()
# i4=255-i4
# cv2.imshow("Original",i) # resmi ekrana bastırdım
# cv2.imshow("i4",i4) # resmi ekrana bastırdım
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows()
 
#  # 2. örnek

# i5=cv2.imread("./images/circles.png",1)
# i6= i5.copy()
# i6=255 -i6
# cv2.imshow("Original",i5) # resmi ekrana bastırdım
# cv2.imshow("i6",i6) # resmi ekrana bastırdım
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows()

#-----------------------------------------------------


# HİSTOGRAM : FREKANS ÇIKARTAN BİR YAPIDIR YANİ KAÇ TANE OLDUĞUNU BULUR
# ÇOK KOYU RESİM DEMEK ÇOK FAZLA DÜŞÜK DEĞERLİ VERİ VAR DEMEK
# NORMALİZASYONDA KONSTRASTA GÖRE 


# I7 = cv2.imread("./images/cameraman.tif",1) 
# hist= cv2.calcHist([I7],[0],None,[256],[0,256]) # histogramı hesapladım

# print(np.sum(I7 ==0))
# print(np.sum(I7 ==1))
# print(np.sum(I7 ==2))
# print(np.sum(I7 ==3))
# print(np.sum(I7 ==4))
# print(np.sum(I7 ==5))
# print(np.sum(I7 ==6))
# print(np.sum(I7 ==7))
# print(np.sum(I7 ==8))
# print(np.sum(I7 ==23))
# print(np.sum(I7 ==22))
# print(np.sum(I7 ==17))

# # 8 ile 17 arası adam olduğunu bulduk

# I7[(I7[:,:,0]>=8) & (I7[:,:,0]<=17),0]=0
# I7[(I7[:,:,1]>=8) & (I7[:,:,1]<=17),1]=0
# I7[(I7[:,:,2]>=8) & (I7[:,:,2]<=17),2]=255

# plt.plot(hist) # histogramı çizdim
# plt.hist(I7 .flat,[256],[0,256])
# plt.show()

# cv2.imshow("Original",I7 ) # resmi ekrana bastırdım
# cv2.waitKey(0) # ekranda kalmasını sağladım
# cv2.destroyAllWindows() # ekrandaki tüm pencereleri kapattım


# bize histogram verilince ne olduğunu anlayabilmemiz lazım sınav sorusu


## öncekinin kısa hali aynılar

# i8 = cv2.imread("./images/cameraman.tif", 1)  # renkli olarak oku
# hist = cv2.calcHist([i8], [0], None, [256], [0, 256])  
# print(np.sum(i8==17))

# i8[(i8[:, :, 0] >= 8) & (i8[:, :, 0] <= 17),0]=0 
# i8[(i8[:, :, 1] >= 8) & (i8[:, :, 1] <= 17),1]=0 
# i8[(i8[:, :, 2] >= 8) & (i8[:, :, 2] <= 17),2]=255

# cv2.imshow("Original",i8) # resmi ekrana bastırdım
# plt.plot(hist) # histogramı çizdim
# plt.show()
# cv2.waitKey(0) # ekranda kalmasını sağladım 
# cv2.destroyAllWindows() 

#-------------------------------------------------

# -1 ile 1 okutulunca histogram farkı nasıl buna baktım +

# i9 = cv2.imread("./images/peppers.png",1)
# color = ("b","g","r")
# for i, col in enumerate(color):
#     print([i,col])
#     hist = cv2.calcHist([i9], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])

# plt.title("Histogram")
# plt.xlabel("Pixel Value")
# plt.ylabel("Frequency")
# plt.legend(["Blue", "Green", "Red"])
# plt.show()

# 100 den sonra 250 ye kadar az piksel var ama geniş bir aralıkta bir sürü var 
# maviden 0 da çok fazla var yani resimde mavi pek yok 
# 50 ile 100 arasında kırmızı ve mavi karışımı mor veya kırmızı ve mavi eşit şekilde olabilir
# 250 de kırmızı çok fazla var bu da resmin kırmızı yoğunluklu olduğunu gösterir


# piksel konum bilgisini vermez yerini vermez renk dağılımı ile ilgili bilgi verir
# nesne tespit ederken histogram kullanılabilir
# düşük konstrat koyu renk 
# konstratı yüksek imge çok iyi sonuç demek değil

#-------------------------------------------------

# histogram eşitleme KENDİ DENEMEM

# i10 = cv2.imread("./images/coins.png", 0)  
# i11 = cv2.equalizeHist(i10)  # histogram eşitleme işlemi
# cv2.imshow("Original", i10)  # resmi ekrana bastırdım
# cv2.imshow("hist eq",i11)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  

#-------------------------------------------------

# histogram eşitleme

# i12 = cv2.imread("./images/peppers.png",0)
# histi12 = cv2.calcHist([i12], [0], None, [256], [0, 256])
# i13 = cv2.equalizeHist(i12)  # histogram eşitleme işlemi
# histi13 = cv2.calcHist([i13], [0], None, [256], [0, 256])
# cv2.imshow("Original", i12)  # resmi ekrana bastırdım
# cv2.imshow("hist eq",i13)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.figure(1)
# plt.plot(histi12)
# plt.figure(2)
# plt.plot( histi13)
# plt.show()

#-------------------------------------------------
# CLAHE ÖRNEK 

# i12 = cv2.imread("./images/peppers.png",0)
# histi12 = cv2.calcHist([i12], [0], None, [256], [0, 256])
# i13 = cv2.equalizeHist(i12)  # histogram eşitleme işlemi
# histi13 = cv2.calcHist([i13], [0], None, [256], [0, 256])

# # clahe1= cv2.createCLAHE()
# clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# i14=clahe1.apply(i12)
# histi14 = cv2.calcHist([i14], [0], None, [256], [0, 256])
# cv2.imshow("Original", i12)  # resmi ekrana bastırdım
# cv2.imshow("hist eq",i13)
# cv2.imshow("clahe",i14)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.figure(1)
# plt.plot(histi12)
# plt.figure(2)
# plt.plot( histi13)
# plt.figure(3)
# plt.plot( histi14)
# plt.show()


#-------------------------------------------------
#RENKLİ GÖRÜNTÜDE CLAHE ÖRNEK FİNAL 

# i_bgr = cv2.imread("./images/peppers.png",1)
# i_hsv = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(i_hsv)
# clahe1= cv2.createCLAHE(clipLimit=2)
# v = clahe1.apply(v)
# i_hsv2 = cv2.merge((h,s,v))
# i_bgr2 = cv2.cvtColor(i_hsv2, cv2.COLOR_HSV2BGR)
# cv2.imshow("Original", i_bgr)  # resmi ekrana bastırdım
# cv2.imshow("clahe",i_bgr2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
