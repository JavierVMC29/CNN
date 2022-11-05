import cv2
import numpy as np

TAMANO_IMG = 50

img = cv2.imread('example.jpg', 1)

img = cv2.resize(img, (TAMANO_IMG, TAMANO_IMG))

# converting to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl, a, b))

# Converting image from LAB Color model to BGR color space
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Stacking the original image with the enhanced image
result = np.hstack((img, enhanced_img))
cv2.imshow('Result', result)
cv2.waitKey(0)
