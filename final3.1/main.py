import cv2
import numpy as np

# i= cv2.imread('./images/peppers.png')
# kernel = np.array([[0,0,0],
#                    [0,1,0],
#                    [0,0,0]]) # 3x3 lük bir kernel oluşturduk identity matrix gibi düşün

# i2 = cv2.filter2D(i, -1, kernel) # i resmine kernel uyguladık 3x3 lük bir filtre uyguladık
# print(np.sum(np.abs(i-i2)))  # i ve i2 resimlerinin farkını alıyoruz 3x3 lük bir filtre uyguladık 2 resim birebir aynı olmalı
# #opencv de zero padding yaparak yapıyor derin öğrenmede zero padding yapmıyoruz resim küçülüyor orda gerçek hayat uygulaması yapıyoruz

# deneme1=np.array([[1,2,3],
#                    [4,5,6],
#                    [7,8,9]], dtype=np.uint8)

# # deneme11=np.array([[0,0,0,0,0],
# #                    [0,1,2,3,0],
# #                    [0,4,5,6,0],
# #                    [0,7,8,9,0],
# #                    [0,0,0,0,0]], dtype=np.uint8)

# kernel2=np.array([[1,2,3],
#                    [1,2,3],
#                    [1,2,3]], dtype=np.uint8)

# deneme2 = cv2.filter2D(deneme1, -1, kernel2) # i resmine kernel uyguladık 3x3 lük bir filtre uyguladık
# print(deneme2) # deneme1 resmine kernel uyguladık 3x3 lük bir filtre uyguladık
# print(np.sum(np.abs(deneme1-deneme2)))  # i ve i2 resimlerinin farkını alıyoruz 3x3 lük bir filtre uyguladık 2 resim birebir aynı olmalı

# # deneme2 = cv2.filter2D(deneme11, -1, kernel2) # i resmine kernel uyguladık 3x3 lük bir filtre uyguladık
# # print(deneme2) # deneme1 resmine kernel uyguladık 3x3 lük bir filtre uyguladık
# # print(np.sum(np.abs(deneme11-deneme2)))  # i ve i2 resimlerinin farkını alıyoruz 3x3 lük bir filtre uyguladık 2 resim birebir aynı olmalı


# cv2.imshow('i', i)
# cv2.imshow('i2',i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------

# i = cv2.imread('./images/peppers.png',0)
# i3=i.copy()

# kernel_dx = np.array([[-1,1]])
# kernel2_dy = np.array([[-1],[1]])

# i3x= cv2.filter2D(i3, -1, kernel_dx) # i resmine kernel uyguladık 
# i3y= cv2.filter2D(i3, -1, kernel2_dy) # i resmine kernel uyguladık 
# i3_last = np.uint8(np.sqrt(np.float32(i3x)**2 + np.float32(i3y)**2) )

# # i3x2 = np.uint8(i3x>30)*255 
# # i3y2 = np.uint8(i3y>30)*255 

# cv2.imshow('i3',i3)
# cv2.imshow('i3x',i3x)
# cv2.imshow('i3y',i3y)
# # cv2.imshow('i3x2',i3x2)
# # cv2.imshow('i3y2',i3y2)
# cv2.imshow('i3_last',i3_last) # x ve y yönündeki kenarları toplar
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------

## kısmi türevin filtre uygulanrak yapılan hali 

# i = cv2.imread('./images/peppers.png',0) 
# i3=i.copy()

# kernel_dx = np.array([[-1,1]])
# kernel2_dy = np.array([[-1],[1]])

# i3x= cv2.filter2D(i3, cv2.CV_64F, kernel_dx) # i resmine kernel uyguladık 
# i3y= cv2.filter2D(i3, cv2.CV_64F, kernel2_dy) # i resmine kernel uyguladık 

# i3x2 = np.uint8(np.abs(i3x>30))*255
# i3y2 = np.uint8(np.abs(i3y>30))*255 
# i3_last = np.uint8(np.sqrt(i3x**2 + i3y**2) )# x ve y yönündeki kenarları toplar
# cv2.imshow('i3',i3)
# cv2.imshow('i3x',i3x)
# cv2.imshow('i3y',i3y)
# cv2.imshow('i3x2',i3x2)
# cv2.imshow('i3y2',i3y2)
# cv2.imshow('i3_last',np.uint8(i3_last>30)*255) # x ve y yönündeki kenarları toplar

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#------------------------------------

# i = cv2.imread('./images/peppers.png',0) # renkli okutup cvtColor ile Bgr to gray yaptık derste 
# i4=i.copy()

# kernel_dx = np.array([[-1,0,1],
#                        [-2,0,2],
#                        [-1,0,1]])

# kernel2_dy = np.array([[-1,-2,-1],
#                        [0,0,0],
#                        [1,2,1]])


# i4x= cv2.filter2D(i4, cv2.CV_64F, kernel_dx) # i resmine kernel uyguladık 
# i4y= cv2.filter2D(i4, cv2.CV_64F, kernel2_dy) # i resmine kernel uyguladık 
# i4_last = np.uint8(np.sqrt(i4x**2 + i4y**2) )# x ve y yönündeki kenarları toplar


# i4x_sobel = cv2.Sobel(i4, ddepth=cv2.CV_64F , dx=1, dy=0 ) # x yönünde kenar bulma işlemi yapar
# i4y_sobel = cv2.Sobel(i4, ddepth=cv2.CV_64F , dx=0, dy=1 ) # x yönünde kenar bulma işlemi yapar
# i4_last_sobel =np.uint8(np.abs( cv2.Sobel(i4, ddepth=cv2.CV_64F , dx=1, dy=1 )) )# x yönünde kenar bulma işlemi yapar

# i4_last_sobelv2 = np.uint8(np.sqrt(i4x_sobel**2 + i4y_sobel**2) )# x ve y yönündeki kenarları toplar

# print(np.sum(np.abs(i4_last-i4_last_sobelv2))) # i ve i2 resimlerinin farkını alıyoruz 3x3 lük bir filtre uyguladık 2 resim birebir aynı olmalı

# cv2.imshow('i4',i4)
# cv2.imshow('i4x',i4x)
# cv2.imshow('i4y',i4y)
# cv2.imshow('i4_last',i4_last) # x ve y yönündeki kenarları toplar

# cv2.imshow('i4x_sobel',i4x_sobel)
# cv2.imshow('i4y_sobel',i4y_sobel)
# cv2.imshow('i4_last_sobel',i4_last_sobel)# x ve y yönündeki kenarları toplar
# cv2.imshow('i4_last_sobelv2',i4_last_sobelv2)# x ve y yönündeki kenarları toplar

# cv2.waitKey(0)
# cv2.destroyAllWindows()
