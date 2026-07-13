import kagglehub
import cv2
import os
import numpy as np

path = kagglehub.dataset_download(
    "somesh24/multiclass-images-for-weather-classification")

print(path)

image_path = os.path.join(path, "Cloudy", "cloudy1.jpg")

img = cv2.imread(image_path)

print(path)
print(image_path)
print(os.path.exists(image_path))

img = cv2.imread(image_path)

cv2.imshow("Bild", img)
cv2.waitKey(0)