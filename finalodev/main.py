import cv2
import numpy as np

i= cv2.imread("./images/tac_mahal2.jpg",1)
height, width = i.shape[:2]
pts1= np.float32([[0, 0], [width, 0], [0, height]])
pts2= np.float32([[0, height], [width, height],[0, 0]])
m= cv2.getAffineTransform(pts1, pts2)
i2= cv2.warpAffine(i, m, (width, height))
m2 = np.float32([[1, 0, 0], [0, 1, -30]])
i2= cv2.warpAffine(i2, m2, (width, height-30))
i4=cv2.GaussianBlur(i2, (5, 5), 0) #i5= cv2.medianBlur(i2, 5)

i3=np.concatenate((i, i4), axis=0)

# cv2.imshow("tac mahal",i)
# cv2.imshow("tac mahal2",i2)
cv2.imshow("tac mahal3",i3)
# cv2.imshow("tac mahal4",i4)
# cv2.imshow("tac mahal5",i5)

cv2.waitKey(0)
cv2.destroyAllWindows()