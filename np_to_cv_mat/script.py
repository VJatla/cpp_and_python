
import numpy as np
import example
import cv2

A = cv2.imread('bear.jpg')
A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
B = example.cv_mat_example(A)
cv2.imshow("A",A)
cv2.imshow("B",B)
cv2.waitKey(0)

