import cv2
import numpy as np

# i= cv2.imread("./images/peppers.png")
# i2 = cv2.blur(i, (5, 5))
# #kendimiz blur yapalım
# # 5x5 boyutunda bir matris oluşturup, her bir elemanına 1/25 değerini atıyoruz
# mean_kernel = np.ones((5, 5), np.float32) / 25 # box blur kernel
# i3 = cv2.filter2D(i, -1, mean_kernel)
# print(np.sum(np.abs(i2 - i3)))
# # gaussian blur yapalım
# i4 = cv2.GaussianBlur(i, (15, 15), 3, 3)

# cv2.imshow("Original", i)
# cv2.imshow("Blurred", i2)
# cv2.imshow("Mean Filtered", i3)
# cv2.imshow("Gaussian Filtered", i4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------

# # tuz ve biber gürültüsü ekleyelim
# i= cv2.imread("./images/circles.png",0)
# i2 = i.copy()
# olasilik=0.05
# ##olasilik=0.2 # %20 oranında gürültü ekleyeceğiz
# rnd1= np.random.random(i.shape[0:2]) # 0-1 arasında rastgele sayılar üretir
# i2[rnd1<olasilik/2]=0
# i2[rnd1>1-olasilik/2]=255

# cv2.imshow("Original", i)
# cv2.imshow("Noisy", i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------

# # tuz ve biber gürültüsü ekleyelim bunu gidermke için median filter kullanalılır 

# i= cv2.imread("./images/circles.png",0)
# i2 = i.copy()
# olasilik=0.2 # %20 oranında gürültü ekleyeceğiz
# rnd1= np.random.random(i.shape[0:2]) # 0-1 arasında rastgele sayılar üretir
# i2[rnd1<olasilik/2]=0
# i2[rnd1>1-olasilik/2]=255

# i3 = cv2.medianBlur(i2, 5) 

# cv2.imshow("Original", i)
# cv2.imshow("Noisy", i2)
# cv2.imshow("Median Filtered", i3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------------------------------

# tuz ve biber gürültüsü ekleyelim bunu gidermke için median filter kullanalılır renkli resimde denedik

# i= cv2.imread("./images/peppers.png",1)
# i2 = i.copy()
# olasilik=0.2 # %20 oranında gürültü ekleyeceğiz
# rnd1= np.random.random(i.shape) # 0-1 arasında rastgele sayılar üretir
# i2[rnd1<olasilik/2]=0
# i2[rnd1>1-olasilik/2]=255

# i3 = cv2.medianBlur(i2, 5) 

# cv2.imshow("Original", i)
# cv2.imshow("Noisy", i2)
# cv2.imshow("Median Filtered", i3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------

# i= cv2.imread("./images/peppers.png",1)
# i2 = cv2.bilateralFilter(i, 9, 75, 75) # 9x9 boyutunda bir kernel kullanıyoruz, sigmaColor ve sigmaSpace değerleri 75
# cv2.imshow("Original", i)
# cv2.imshow("bilateral", i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------

# i= cv2.imread("./images/peppers.png",1)

# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]]) # 3x3 boyutunda bir kernel kullanıyoruz
                  
# i2 = cv2.filter2D(i, -1, kernel) # 2D filtreleme işlemi yapıyoruz

# cv2.imshow("Original", i)
# cv2.imshow("i2", i2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
