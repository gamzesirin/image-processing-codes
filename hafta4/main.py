import cv2 #open cv import ettim
import numpy as np #numpy import ettim
# import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt #matplotlib import ettim


print(cv2.__version__) # open cv nin versiyonunu yazdırdım (4.11.0 çıktı)
# I = cv2.imread('./images/circles.png',1) #resmi okuttum default : 1 oluyor istersek yazabiliriz RENKLİ ALDIK
I = cv2.imread('./images/circles.png',0) #resmi okuttum  GRİ TONLU ALDIK 

# print(I.shape)
# I[128,128,0]=255
# cv2.imshow('I1',I) #resmi gösterdim  
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#----------------------------------------------

# se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)) #40x40 lük bir elips şeklnde matris oluşturdum
# print(se.shape)
# print(se.dtype)
# print(se)
# # I2=cv2.dilate(I,se,iterations=1) #resmi erode ettim 1 iterasyon yaptım 
# I2 = cv2.morphologyEx(I, cv2.MORPH_DILATE, se)

# cv2.imshow('orinal',I) #resmi gösterdim 
# cv2.imshow('delation',I2) #resmi gösterdim  
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma
# print(np.max(I)) #en büyük değeri yazdırdım

#----------------------------------------------

I3= cv2.imread('./images/fig1.jpg',1) #resmi okuttum   default : 1 oluyor istersek yazabiliriz RENKLİ ALDIK
# se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40)) #15*15 lük bir elips şeklnde matris oluşturdum
# print(se.shape)
# print(se.dtype)
# print(se)
# I4 = cv2.morphologyEx(I3, cv2.MORPH_OPEN, se)

# # I4_control1 = cv2.erode(I3,se,iterations=1)
# # I4_control2 = cv2.dilate(I4_control1,se,iterations=1)

# I4_control1 = cv2.morphologyEx(I3, cv2.MORPH_ERODE, se)
# I4_control2 = cv2.morphologyEx(I4_control1,cv2.MORPH_DILATE, se)


# # print(np.sum(np.sum(I4-I4_control2)))
# cv2.imshow('orinal',I3) #resmi gösterdim 
# cv2.imshow('opening',I4) #resmi gösterdim  önce açma işlemi yaptım sonra kapatma işlemi yaptım resmin boyutu korunur

# cv2.imshow('control1',I4_control1) #resmi gösterdim
# cv2.imshow('control2',I4_control2) #resmi gösterdim

# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma


#----------------------------------------------

# se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40)) #15*15 lük bir elips şeklnde matris oluşturdum
# print(se.shape)
# print(se.dtype)
# print(se)
# I5 = cv2.morphologyEx(I, cv2.MORPH_CLOSE, se)

# # I5_control1 = cv2.morphologyEx(I5, cv2.MORPH_DILATE, se)
# # I5_control2 = cv2.morphologyEx(I5_control1,cv2.MORPH_ERODE, se)

# cv2.imshow('orinal',I) #resmi gösterdim 
# cv2.imshow('closing',I5) #resmi gösterdim ÖNCE KAPATMA YAP SONRA AÇMA YAP
# # cv2.imshow("control1",I5_control1)
# # cv2.imshow("control2",I5_control2)


# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#----------------------------------------------

# interpolation :  bilinen Değerlerden yola çıkarak bilinmeyen değeri bulma algoritmaları 

I6 =cv2.imread('./images/peppers.png') #resmi okuttum 
# print(I6.shape)
# I_small = cv2.resize(I6, (200,200))
# # I_small2 = cv2.resize(I6, (200,200),interpolation=cv2.INTER_CUBIC)
# # vertical =np.concatenate((I_small,I_small2),axis=0)
# print(I_small.shape)

# I_small2 = cv2.resize(I6, None, fx=2,fy=0.5,interpolation=cv2.INTER_LINEAR)

# I_big = cv2.resize(I6, (1000,1000)) #resmi büyüttüm
# print(I_big.shape)

# cv2.imshow('orj',I6) #resmi gösterdim  
# # cv2.imshow('vertical',vertical) #resmi gösterdim
# cv2.imshow('small',I_small) #resmi gösterdim  
# cv2.imshow('small2',I_small2) #resmi gösterdim 

# cv2.imshow('big',I_big) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma


#----------------------------------------------

# I9 = I6[230:330,400:510] #resmi kırpma işlemi yaptım
# # I9 = I6[80:130,170:250] #resmi kırpma işlemi yaptım   // satır ve sütunları belirleyerek kırpma işlemi yaptım , satırda 80-130 arasını sütunda 170-250 arasını aldım satır: yüksekliği ve sütun: genişliği belirler 
# # I9 = I6[250:360,0:120] #domatesi kırptım
# print(I6.shape)
# print(I9.shape)
# cv2.imshow('orj',I6) #resmi gösterdim
# cv2.imshow('crop',I9) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma


#------------------AFİN DÖNÜŞÜMLERİ----------------------------

# I10 =cv2.imread('./images/peppers.png') #resmi okuttum 
# print(I10.shape)
# # heigth,witdh= I10.shape[0],I10.shape[1]
# heigth,witdh=I10.shape[:2]
# print(heigth,witdh)
# center = [heigth/2,witdh/2]
# print(center)
# angle=45 #90
# rotation_matrix = cv2.getRotationMatrix2D(center,angle,1)
# print(rotation_matrix)
# print(rotation_matrix.dtype) #float64

# rm2 = np.array([[np.cos(np.pi/4),np.sin(np.pi/4),0],[-np.sin(np.pi/4),np.cos(np.pi/4),400]],dtype=np.float64)
# print(rm2)

# I11 = cv2.warpAffine(I10,rm2,(witdh+500,heigth+500)) 

# I12 = cv2.warpAffine(I10,rotation_matrix,(witdh,heigth))
# I13 = cv2.rotate(I10,cv2.ROTATE_90_COUNTERCLOCKWISE)
# cv2.imshow('orj',I10) #resmi gösterdim
# cv2.imshow('rotate',I11) #resmi gösterdim
# cv2.imshow('rotate2',I12) #resmi gösterdim
# cv2.imshow('rotate3',I13) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#------------------------------------------------

# I14 = cv2.imread('./images/peppers.png')
# heigth,witdh=I14.shape[:2]  
# print(heigth,witdh)

# pts1=np.float32([[50,50],[200,50],[50,200]]) #3 nokta belirledim
# pts2=np.float32([[10,100],[200,50],[100,250]]) #3 nokta belirledim
# M = cv2.getAffineTransform(pts1,pts2) #affine dönüşümü yaptım ilk 3 noktayı 2. 3 noktaya dönüştürdüm
# print(M)
# I15 = cv2.warpAffine(I14,M,(witdh,heigth)) #dönüşümü yaptım

# cv2.imshow('rotate3',I14) #resmi gösterdim
# cv2.imshow('rotate4',I15) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#------------------------------------------------
#GEÇEN SENE SINAV SORUSU

# I14 = cv2.imread('./images/peppers.png')
# heigth,witdh=I14.shape[:2]
# print(heigth,witdh)
# sol_üst=(0,0)
# sağ_üst=(witdh,0)
# sol_alt=(0,heigth)  
# sağ_alt=(witdh,heigth)

# pts3=np.float32([sol_üst,sağ_üst,sol_alt])
# pts4=np.float32([sağ_üst,sol_üst,sağ_alt])

# M = cv2.getAffineTransform(pts3,pts4)
# print(M)

# I15 = cv2.warpAffine(I14,M,(witdh,heigth))

# cv2.imshow('rotate3',I14) #resmi gösterdim
# cv2.imshow('rotate4',I15) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#------------------------------------------------

# I14 = cv2.imread('./images/peppers.png')
# heigth,witdh=I14.shape[0:2]
# print(heigth,witdh)
# sol_üst=(0,0)
# sağ_üst=(witdh,0)
# sağ_alt=(witdh,heigth)
# sol_alt=(0,heigth)

# pts3=np.float32([sol_üst,sağ_üst,sağ_alt])
# pts4=np.float32([sol_alt,sağ_alt,sağ_üst])

# M = cv2.getAffineTransform(pts3,pts4)
# print(M)

# I15 = cv2.warpAffine(I14,M,(witdh,heigth))

# cv2.imshow('rotate3',I14) #resmi gösterdim
# cv2.imshow('rotate4',I15) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma


#------------------------------------------------

# I14 = cv2.imread('./images/peppers.png')
# heigth,witdh=I14.shape[0:2]
# print(heigth,witdh)
# sol_üst=(0,0)
# sağ_üst=(witdh,0)
# sağ_alt=(witdh,heigth)
# sol_alt=(0,heigth)

# pts3=np.float32([sol_üst,sağ_üst,sağ_alt])
# pts4=np.float32([sağ_alt,sol_alt,sol_üst])

# M = cv2.getAffineTransform(pts3,pts4)
# print(M)

# I15 = cv2.warpAffine(I14,M,(witdh,heigth))

# cv2.imshow('rotate3',I14) #resmi gösterdim
# cv2.imshow('rotate4',I15) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma


# # # 2.YOL

# I20 = cv2.imread('./images/peppers.png')

# heigth,witdh=I20.shape[:2]
# print(heigth,witdh)
# center = (witdh//2,heigth//2)
# print(center)
# angle=180
# m= cv2.getRotationMatrix2D(center,angle,1)

# I30= cv2.warpAffine(I20,m,(witdh,heigth))
# cv2.imshow('orj',I20)
# cv2.imshow('İ30',I30)
# cv2.waitKey(0)
# cv2.destroyAllWindows()