import cv2
import numpy as np
import sys

kernel_size = 3


def get_corners(edges):
    dst = cv2.cornerHarris(edges, 2, 3, 0.04)

    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return dst_norm


def round_corners(corners, picture):
    dst = picture.copy()
    threshold = 100
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if int(corners[i,j]) > threshold:
                cv2.circle(dst, (j,i), 2, 255)
    return dst


def median_filter_process_pixel(img, x, y, radius):
    w, h, _ = img.shape
    b_arr = []
    g_arr = []
    r_arr = []
    for j in range(-radius, radius + 1):
        for i in range(-radius, radius + 1):
            index_x = np.clip(x + j, 0, w - 1)
            index_y = np.clip(y + i, 0, h - 1)
            b_arr.append(img[index_x][index_y][0])
            g_arr.append(img[index_x][index_y][1])
            r_arr.append(img[index_x][index_y][2])
    filter_pixel = np.array([np.median(b_arr), np.median(g_arr), np.median(r_arr)], dtype=np.int8)
    return filter_pixel


def median_filter(img, distance, k):
    filtered_image = img.copy()
    w, h, _ = img.shape
    for j in range(w):
        for i in range(h):
            filtered_image[j][i] = median_filter_process_pixel(img, j, i, int(k * distance[j][i]))
    return filtered_image


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray_image)
    edges = cv2.Canny(equalized, 150, 250, 3)
    corners = get_corners(edges)
    img_corners = round_corners(corners, edges)

    return img_corners


def main():
    image = cv2.imread("gollum_small.jpg")
    img_corners = process_image(image)
    cv2.imshow("img_corners", img_corners)
    distance = cv2.distanceTransform(img_corners, cv2.DIST_L2, 3)
    filtered_img = median_filter(image, distance, 10)
    processed_filtered_image = process_image(filtered_img)
    cv2.imshow("filtered_img", processed_filtered_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    sys.exit(main() or 0)
