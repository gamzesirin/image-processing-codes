import cv2 #open cv import ettim
import numpy as np #numpy import ettim
from matplotlib import pyplot as plt #matplotlib import ettim


# print(cv2.__version__) # open cv nin versiyonunu yazdırdım (4.11.0 çıktı)
I = cv2.imread('./images/peppers.png',1) #resmi okuttum   default : 1 oluyor istersek yazabiliriz
# print(I.shape) #resmin boyutlarını yazdırdım
# print(type(I)) #resmin tipini yazdırdım
# print(I.ndim) #resmin boyut sayısını yazdırdım   matrsinin boyutu   satır sütun ,  3 bgr demek 0 1 2 olur yatayda uzun 
# print(I.size) # piksel sayısı
# print(I.dtype) #resmin veri tipini yazdırdım
# print("*"*20)

# #---------------------------------------------- 

# print(I[10,10]) # matristeki 10,10 olan pikselin bgr değerlerini yazdırdım
# print(I[50,50,:]) # matristeki 50,50 olan pikselin bgr değerlerini yazdırdım
# # I2=I #ı yı ı2 ye atadım bunlar heapte tutuluyor aynı yerdeler birinde yapılan değişiklik diğerini de etkiler
# I2=I.copy() #doğru olan bu bunlar heapte tutuluyor aynı yerdeler
# I2[10,10]=[0,0,0] # I2 de 10,10 olan pikselin bgr değerlerini 0,0,0 yaptım
# print(I[10,10]) 
# print(I2[10,10])

#----------------------------------------------

# cv2.imshow('peppers1',I) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma

#----------------------------------------------

# I3=np.random.random((300,500,3)) #300x500x3 lük bir matris oluşturdum
# print(I3.shape) #boyutlarını yazdırdım
# I3=np.uint8(np.round(I3*255)) # veri tipini değiştirdim
# print(I3.dtype) #veri tipini yazdırdım

# cv2.imshow('random1',I3) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi 
# cv2.destroyAllWindows() #pencereleri kapatma

##############################################
# I8=cv2.cvtColor(I3,cv2.COLOR_BGR2GRAY)
# cv2.imshow('random1',I8) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma
# #----------------------------------------------

# plt.figure()
# plt.title('Image Peppers Orj')

# plt.xticks([]), plt.yticks([]) #eksenleri kaldırır

# plt.imshow(I)
# plt.waitforbuttonpress()
# plt.close()


# plt.imshow(I, cmap='Greys', interpolation='nearest')
# plt.imshow(I, cmap='binary', interpolation='nearest')

# #----------------------------------------------

# I2 =I.copy()
# px=I2[50,50]
# px_blue=I2[50,50,0]
# print(px)
# print(px_blue)
# px_blue=I2[50,50,0]=100
# print(px_blue)
# I2[50,50]= [255,255,255]
# cv2.imshow('Image Peppers with white dot',I2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------------

# I4=np.zeros((I.shape[0],I.shape[1],3),dtype=np.uint8)  #sıfırlardan oluşan bir matris oluşturdum 
# print(I4.shape) #boyutlarını yazdırdım
# print(I4.dtype) #veri tipini yazdırdım
# print(I4[10,10]) #10,10 olan pikselin bgr değerlerini yazdırdım
# print(I4) #matrisi yazdırdım

# I_r=I[:,:,2]
# # print(I_b.shape)
# I_g=I[:,:,1]
# I_b=I[:,:,0]

# I4[:,:,0] = I_r
# I4[:,:,1] = I_g
# I4[:,:,2] = I_b

# cv2.imshow("bgr İMAGE",I)
# cv2.imshow('RGB image yeniden olusturma',I4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.figure()
# plt.imshow(I4)
# plt.waitforbuttonpress(0)
# plt.close()

#----------------------------------------------

# I5_b,I5_g,I5_r=cv2.split(I) #resmi bgr olarak ayırdım
# I6=cv2.merge([I5_r,I5_g,I5_b]) #resmi rgb olarak birleştirdim
# print(I6.shape) #boyutlarını yazdırdım

# cv2.imshow("example",I6)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(I6)
# plt.waitforbuttonpress(0)
# plt.close()

#----------------------------------------------

# I7=I[:,:,::-1] #resmi ters çevirdim

# cv2.imshow("example",I7)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(I7) #resmi gösterdim
# plt.waitforbuttonpress(0) #bekleme süresi
# plt.close() #pencereleri kapatma

#----------------------------------------------

# print(I.shape)
# I8=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY) # COLOR_BGR2HSV
# print(I8.shape)

# cv2.imshow("example",I8)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------------------
#Resim kaydetme 1

# cv2.imwrite("peppers_gray.tif",I8)

#Resim kaydetme 2

# plt.imshow(I8,cmap='gray',interpolation='Nearest')
# plt.savefig('peppers_gray2.tif',dpi=600)

#----------------------------------------------

# imageText = I.copy()
# text = 'Merhaba Dunya'
# org = (50,80)
# cv2.putText(imageText, text, org, fontFace =cv2.FONT_HERSHEY_COMPLEX, fontScale = .5, color =(250,225,400))
# cv2.imshow("Image Text",imageText)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------------------
# I2=I.copy() #ı yı ı2 ye kopayaladım artık ı2 değişiklik yapabilirim bu ı yı etkilemez
# cv2.line(I2,(100,100),(200,200),(255,0,0),thickness=3) # 3 kalınlığında mavi bir çizgi çizdim
# cv2.imshow('example',I2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------          MORFOLOJİ       -----------------------

i = cv2.imread('./images/fig1.jpg') #resmi okuttum   default : 1 oluyor istersek yazabiliriz
print(i.shape) #resmin boyutlarını yazdırdım
# cv2.imshow('fig1',i) #resmi gösterdim
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma 

#----------------------------------------------

# se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40)) #40x40 lük bir elips şeklnde matris oluşturdum
# print(se.shape)
# print(se.dtype)
# print(se)
# # i1=cv2.erode(i,se,iterations=1) #resmi erode ettim 1 iterasyon yaptım 
# i1 = cv2.morphologyEx(i, cv2.MORPH_ERODE, se)

# cv2.imshow('ORJ',i) #resmi gösterdim
# cv2.imshow('fig1',i1) #resmi gösterdim  
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma
# print(np.max(i1)) #en büyük değeri yazdırdım



#----------------------------------------

# se1=np.array([[0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0]],dtype=np.uint8) #1x22 lik bir matris oluşturdum yani bir satır 22 sütun
# se2=np.array([[0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0]],dtype=np.uint8).T #22x1 lik bir matris oluşturdum yani 22 satır 1 sütun
# print(se2.shape)
# print(se2.dtype)
# print(se2)
# i2=cv2.erode(i,se2) #resmi erode ettim 1 iterasyon yaptım  

# cv2.imshow('ORJ',i) #resmi gösterdim
# cv2.imshow('fig1',i2) #resmi gösterdim  
# cv2.waitKey(0) #bekleme süresi
# cv2.destroyAllWindows() #pencereleri kapatma




