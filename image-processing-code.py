import cv2
import numpy as np
from tkinter import Tk, filedialog



# image picking
print("select an image")
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not file_path:
    raise ValueError("no image selected")

# load image
img = cv2.imread(file_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# counting red pixels
R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
red_mask = (R > 150) & (R-G > G) & (R-B > B)
print("red pixel count:", np.sum(red_mask))



# creating grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# converting grayscale to BGR so we can stack images
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# resizing both images to match
h = 400
orig_resized = cv2.resize(img, (int(img.shape[1] * (h / img.shape[0])), h))
gray_resized = cv2.resize(gray_bgr, (orig_resized.shape[1], h))

# putting grayscale and BGR side by side
side_by_side = np.hstack((orig_resized, gray_resized))

# display the combined image
cv2.imshow("original (left)  |  grayscale (right)", side_by_side)



# finding dominant color using stable OPENCV K-MEANS
small = cv2.resize(rgb, (300, 300))
pixels = small.reshape(-1, 3).astype(np.float32)

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 50, 1.0)

compactness, labels, centers = cv2.kmeans(
    pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

dominant_color = centers[0].astype(int)
print("dominant color (RGB):", dominant_color)

# create pop-up window of the dominant color
dom_img = np.zeros((200, 200, 3), dtype=np.uint8)
dom_img[:] = dominant_color[::-1]  # RGB to BGR

cv2.imshow("dominant color", dom_img)



# creating a threshold image (black and white image)
threshold , thresh = cv2.threshold(gray , 150 , 255 , cv2.THRESH_BINARY)
cv2.imshow("Thresholded image" , thresh)



# creating a hsv image (hue, saturation, value)
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", HSV)



# creating masks (masking the red colored objects)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(HSV, lower_red1, upper_red1)
mask2 = cv2.inRange(HSV, lower_red2, upper_red2)

masks = mask1 + mask2

cv2.imshow("masks", masks)

#contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (255,0,0), 2)
cv2.imshow("Contours", img)



# remove pop-up windows by pressing any key
cv2.waitKey(0)
cv2.destroyAllWindows()
