import cv2
import numpy as np

i = cv2.imread('./images/frozen_rose.jpg', 1)
print(i.shape)
height, width = i.shape[:2]
print(height, width)

i4 = i.copy()

# 3x3 parçalar (total 9 parça)
# Üst satır (ilk 1/3 yükseklik)
i11 = i4[0:height//3, 0:width//3]                     # Üst sol
i12 = i4[0:height//3, width//3:2*width//3]            # Üst orta
i13 = i4[0:height//3, 2*width//3:width]               # Üst sağ

# Orta satır (ikinci 1/3 yükseklik)
i14 = i4[height//3:2*height//3, 0:width//3]           # Orta sol
i15 = i4[height//3:2*height//3, width//3:2*width//3]  # Orta orta
i16 = i4[height//3:2*height//3, 2*width//3:width]     # Orta sağ

# Alt satır (son 1/3 yükseklik)
i17 = i4[2*height//3:height, 0:width//3]              # Alt sol
i18 = i4[2*height//3:height, width//3:2*width//3]     # Alt orta
i19 = i4[2*height//3:height, 2*width//3:width]        # Alt sağ

# Beyaz çizgi kalınlığı
line_thickness = 5

# Parçaların tam boyutlarını al
h1 = i11.shape[0]  # Üst satır yüksekliği
h2 = i14.shape[0]  # Orta satır yüksekliği
h3 = i17.shape[0]  # Alt satır yüksekliği

w1 = i11.shape[1]  # Sol sütun genişliği
w2 = i12.shape[1]  # Orta sütun genişliği
w3 = i13.shape[1]  # Sağ sütun genişliği

# Her parça için doğru boyutlarda beyaz çizgiler oluştur
vertical_line1 = np.ones((h1, line_thickness, 3), dtype=np.uint8) * 255  # Üst satır için dikey çizgi
vertical_line2 = np.ones((h2, line_thickness, 3), dtype=np.uint8) * 255  # Orta satır için dikey çizgi
vertical_line3 = np.ones((h3, line_thickness, 3), dtype=np.uint8) * 255  # Alt satır için dikey çizgi

# Yatay çizgiler
horizontal_line1 = np.ones((line_thickness, w1 + line_thickness + w2 + line_thickness + w3, 3), dtype=np.uint8) * 255  # Üst ve orta satır arası
horizontal_line2 = np.ones((line_thickness, w1 + line_thickness + w2 + line_thickness + w3, 3), dtype=np.uint8) * 255  # Orta ve alt satır arası

# Her satır için parçaları ve dikey çizgileri birleştir
top_row = np.concatenate((i11, vertical_line1, i12, vertical_line1, i13), axis=1)
middle_row = np.concatenate((i14, vertical_line2, i15, vertical_line2, i16), axis=1)
bottom_row = np.concatenate((i17, vertical_line3, i18, vertical_line3, i19), axis=1)

# Satırları ve yatay çizgileri birleştirerek son görüntüyü oluştur
final_image = np.concatenate((top_row, horizontal_line1, middle_row, horizontal_line2, bottom_row), axis=0)

# Sonucu göster
cv2.imshow('Original', i4)

cv2.imshow("i11", i11)
cv2.imshow("i12", i12)
cv2.imshow("i13", i13)
cv2.imshow("i14", i14)
cv2.imshow("i15", i15)
cv2.imshow("i16", i16)
cv2.imshow("i17", i17)
cv2.imshow("i18", i18)
cv2.imshow("i19", i19)


cv2.imshow('Top Row', top_row)
cv2.imshow('Middle Row', middle_row)
cv2.imshow('Bottom Row', bottom_row)



cv2.imshow('Final Image with White Lines', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()