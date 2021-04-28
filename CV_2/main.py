import cv2
import numpy as np
import sys


def relu(src):
    res = src.copy()
    w, h, d = src.shape

    for i in range(w):
        for j in range(h):
            for k in range(d):
                res[i][j][k] = max(0, res[i][j][k])

    return res


def filter_3x3x3(src, x, y, weight):
    def check(pos, max_pos):
        return (pos >= 0) and (pos < max_pos)

    res = 0
    w, h, d = src.shape

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(3):
                if check(x + i, w) and check(y + j, h):
                    res += src[x + i][y + j][k] * weight[i + 1][j + 1][k]

    return res


def conv(src, weights, new_d):
    w, h, d = src.shape
    res = np.zeros((w - 1, h - 1, new_d))
    w, h, d = res.shape

    for i in range(w):
        for j in range(h):
            for k in range(d):
                res[i][j][k] = filter_3x3x3(src, i, j, weights)

    return res


def filter_2x2(src, x, y, d):
    def check(pos, max_pos):
        return (pos >= 0) and (pos < max_pos)

    w, h = src.shape[:2]

    res = src[x][y][d]
    for i in range(2):
        for j in range(2):
            if check(x + i, w) and check(y + j, h):
                res = max(res, src[x + i][y + j][d])

    return res

def max_pooling_2x2(src):
    w, h, d = src.shape
    res = np.zeros((int(w / 2), int(h / 2), d))
    w, h, d = res.shape

    for i in range(w):
        for j in range(h):
            for k in range(d):
                res[i][j][k] = filter_2x2(src, 2 * i, 2 * j, k)

    return res


def main():
    src = cv2.imread("gollum_small.jpg")
    res = conv(src, np.random.rand(3, 3, 3), 5)
    res = relu(res)
    res = max_pooling_2x2(res)
    print(res.shape)


if __name__ == "__main__":
    sys.exit(main() or 0)