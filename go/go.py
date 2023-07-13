import cv2 as cv
import numpy as np
import math

#이미지 로드
src_gray = cv.imread('GO1.jpg',cv.IMREAD_GRAYSCALE)

if src_gray is None:
    print("Image load failed")

cv.imshow('src_gray',src_gray)

#노이즈 제거
blur_src_gray=cv.GaussianBlur(src_gray,(5,5),0)
dst_l = cv.cvtColor(src_gray, cv.COLOR_GRAY2BGR)

# 바둑알 검출
circles = cv.HoughCircles(src_gray, cv.HOUGH_GRADIENT,1,10,param1=100,param2=20,minRadius= 11,maxRadius=13)

if circles is not None:
    for i in range(circles.shape[1]):
        cx,cy,radius = circles[0][i]
        cv.circle(dst_l,(int(cx),int(cy)),int(radius),(0,0,255),2,cv.LINE_AA)

# 바둑알이 적은 직선 검출
edge = cv.Canny(blur_src_gray,70,300)
cv.imshow("edge",edge) 
lines = cv.HoughLines(edge,1,math.pi/90,220)

if lines is not None:
    for i in range(lines.shape[0]):
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        cos_t=math.cos(theta)
        sin_t=math.sin(theta)
        x0,y0=rho*cos_t,rho*sin_t
        alpha=1000
        # theta 값 기준하여 수평/수직선 1차 검출
        if theta > np.pi / 180 * 170 or theta < np.pi / 180 * 10:
            pt1 = (int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * (cos_t)))
            pt2 = (int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * (cos_t)))
            cv.line(dst_l, pt1, pt2, (0,128, 0), 1, cv.LINE_AA)

        if theta > np.pi / 180 * 80 and theta < np.pi / 180 * 100:
            pt1 = (int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * (cos_t)))
            pt2 = (int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * (cos_t)))
            cv.line(dst_l, pt1, pt2, (0,128, 0), 1, cv.LINE_AA)

#3. 바둑알 중점 좌표로 직선 검출
center_dst = np.zeros((dst_l.shape[0], dst_l.shape[1]), dtype=np.uint8)
for circle in circles[0,:]:
    center = (int(circle[0]), int(circle[1]))
    center_dst[center] = 255

center_lines = cv.HoughLines(center_dst, 1, np.pi / 180, 5, 0, 0)
if center_lines is not None:
    for line in center_lines:
        rho, theta = line[0]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x0 = cos_t * rho
        y0 = sin_t * rho
        alpha=1000
        # theta 값 이용한 수평/수직선 최종
        if theta > np.pi / 180 * 170 or theta < np.pi / 180 * 10:
            pt1 = (int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * (cos_t)))
            pt2 = (int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * (cos_t)))
            cv.line(dst_l, pt1, pt2, (0,128, 0), 1, cv.LINE_AA)

        if theta > np.pi / 180 * 80 and theta < np.pi / 180 * 100:
            pt1 = (int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * (cos_t)))
            pt2 = (int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * (cos_t)))
            cv.line(dst_l, pt1, pt2, (0,128, 0), 1, cv.LINE_AA)

# 바둑돌 중점 표시
if circles is not None:
    for circle in circles[0,:]:
        center = (int(circle[0]), int(circle[1]))
        cv.line(dst_l, center, center, (255,0, 0), 6, cv.LINE_AA)


cv.imshow('lines',dst_l)
cv.waitKey()
cv.destroyAllWindows() 