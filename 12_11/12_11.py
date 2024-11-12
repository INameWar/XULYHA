import cv2
import numpy as np
import matplotlib.pyplot as plt


# Hàm để áp dụng bộ lọc Prewitt
def prewitt_filter(image):
    kernel_x = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])

    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient


# Đọc ảnh
image = cv2.imread('download.jpg', cv2.IMREAD_GRAYSCALE)

# Bước 1: Làm mịn ảnh bằng bộ lọc Gaussian
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Bước 2: Phát hiện biên bằng Sobel
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

# Bước 3: Phát hiện biên bằng Prewitt
prewitt_gradient = prewitt_filter(blurred_image)

# Bước 4: Phát hiện biên bằng Canny
canny_edges = cv2.Canny(blurred_image, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel Edges')
plt.imshow(sobel_gradient, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Prewitt Edges')
plt.imshow(prewitt_gradient, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Canny Edges')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()