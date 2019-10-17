#encoding=utf-8

import numpy as np
import cv2

def svd(x, k):
    u, Sigma, vt = np.linalg.svd(x)
    s = np.zeros(shape=(k, k), dtype=np.float32)
    for i in range(k):
        s[i, i] = Sigma[i]
    return np.dot(np.dot(u[:, 0:k], s), vt[0:k, :])


if __name__ == '__main__':
    path = '../data/SVD_test.jpg'
    out = '../data/SVD_res_'
    pic = cv2.imread(path)
    res = np.zeros_like(pic, dtype=np.float32)
    for k in range(5, 51, 5):
        for i in range(3):
            r = svd(pic[:, :, i], k)
            res[:, :, i] = r
        cv2.imwrite(out + str(k) + '.jpg', res)