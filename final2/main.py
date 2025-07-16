import cv2
import numpy as np

# # varyans: farklılık

# I= cv2.imread("./images/threshold.jpg",0)
# thresh,I2=cv2.threshold(I,230,255,cv2.THRESH_BINARY)
# print(thresh)
# print(I2.dtype)
# # I3=I.copy()
# # I3[(I3>=11)]=0
# # I3[(I3>1) & (I3<10)]=255
# esik,I_otsu=cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(['Otsu esik= ',esik])
# cv2.imshow('I',I)
# cv2.imshow('I2',I2)
# # cv2.imshow('I3',I3)
# cv2.imshow('I_otsu',I_otsu)
# cv2.waitKey()
# cv2.destroyAllWindows()

##---------------------------------------------------------

# i4=cv2.imread("./images/cameraman.tif",0) # resmi okudum //  i4=cv2.imread("./images/peppers.png",0)
# thresh1, i5=cv2.threshold(i4, 127, 255, cv2.THRESH_BINARY) # binary eşikleme yaptım
# print(thresh1)
# thresh2, i6=cv2.threshold(i4, 127, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) # binary ters eşikleme yaptım
# print(thresh2)
# print(i5.dtype)
# print(i6.dtype)
# cv2.imshow("Original",i4) # resmi ekrana bastırdım
# cv2.imshow("thresh1",i5) # resmi ekrana bastırdım
# cv2.imshow("thresh2",i6) # resmi ekrana bastırdım

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 255 -i dersek siyah beyaz ters olur


# #---------------------------------------------------------

# ## boşluk doldurma
## kapalı nesnelerde kullanılır nesnenin sınırları kapalı olmak zorunda

i7=cv2.imread("./images/coins.png",0) # resmi okudum
thresh,i8=cv2.threshold(i7,127,255,cv2.THRESH_BINARY) # binary eşikleme yaptım

h,w=i7.shape[:2] # resmin boyutunu aldım
mask = np.zeros((h+2,w+2), dtype= np.uint8) # maskeyi oluşturdum
i9=i8.copy() # i8 kopyasını aldım
cv2.floodFill(i9,mask,(0,0),255) # floodfill uyguladım

i10 = cv2.bitwise_not(i9) # i9 un tersini aldım
print(i9.dtype)
i11 = (i8 | i10) # i8 ve i10 un bitwise or işlemini yaptım

# # ##--------------------------------
# print("***********")
# print(i11.dtype)
# print(i11.shape)
# print((type(i11)))
# print(np.max(i11))
# print(np.min(i11))
i12= np.bool_(i11)
# print("***********")

# print(i12.dtype)
# print(i12.shape)
# print((type(i12)))
# print(np.max(i12))
# print(np.min(i12))
# print((np.max(np.uint8(i12))))
i13= i11.copy()# i12 dizisini uint8 formatına dönüştürdüm


i13[i12]=128 #   i13[~i12]=128

# i_bgr = np.zeros((h,w,3), dtype=np.uint8) # boş bir bgr resmi oluşturdum
# i_bgr[i11==255] = [0, 0, 255] # i11 deki beyaz pikselleri kırmızıya boyadım



cv2.imshow("i7",i7) # resmi ekrana bastırdım
cv2.imshow("i8",i8) # resmi ekrana bastırdım
cv2.imshow("i9",i9) # resmi ekrana bastırdım
cv2.imshow("i10",i10) # resmi ekrana bastırdım
# cv2.imshow("i11",i11) # resmi ekrana bastırdım
# cv2.imshow("i12",np.uint8(i12)*255) # resmi ekrana bastırdım
cv2.imshow("i13",i13) # resmi ekrana bastırdım
# cv2.imshow("i_bgr",i_bgr) # resmi ekrana bastırdım
cv2.waitKey(0)
cv2.destroyAllWindows()


##### bağlantılı örnek


# i10_v2 = cv2.morphologyEx(i10,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))) # morfolojik kapama işlemi yaptım

# i10_v3 = cv2.morphologyEx(i10_v2,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) # morfolojik kapama işlemi yaptım

# cv2.imshow("i10",i10) # resmi ekrana bastırdım
# cv2.imshow("i10_v2",i10_v2) # resmi ekrana bastırdım
# cv2.imshow("i10_v3",i10_v3) # resmi ekrana bastırdım
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# xy = np.column_stack(np.where(i10_v3==255)) 
# i14 = i7[np.min(xy[:,0]):np.max(xy[:,0]),np.min(xy[:,1]):np.max(xy[:,1])] # i dizisinin xy koordinatları arasındaki kısmını aldım
# print(xy)
# cv2.imshow("i14",i14) # resmi ekrana bastırdım
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #---------------------------------------------------------
