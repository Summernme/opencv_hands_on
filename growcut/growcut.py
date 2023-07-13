import cv2
import math
import numpy as np
from scipy.spatial import distance


# Growcut algorithm
def growcut(image):
  image = image.astype(np.float64)
  global theta, seed
  count = 1

  while (count != 0):
    count = 0
    t1_seed = np.copy(seed)
    t1_theta = np.copy(theta)
    for y in range(0, seed.shape[0]):
      for x in range(0, seed.shape[1]):

        for cx in range(-1, 2):
          for cy in range(-1, 2):
            if cx == 0 and cy == 0:
              continue
            new_x = x + cx
            new_y = y + cy
            if new_x < 0 or new_x >= image.shape[1] or new_y < 0 or new_y >= image.shape[0]:
              continue
            
            dist = np.sqrt(np.sum(image[y,x]-image[new_y,new_x])**2)
            g_x = 1 - (dist / max)
            rule = g_x * theta[new_y,new_x]
            if rule > t1_theta[y][x]:
              t1_seed[y][x] = seed[new_y][new_x]
              count+=1
              t1_theta[y][x] = rule
    seed = t1_seed
    theta = t1_theta
    cv2.imshow("theta", theta)
    cv2.imshow("seed", seed)
    cv2.waitKey(1)
    print(count)


# Seed Setting
def drawing(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    cv2.circle(image,(x,y),5,(0,0,255),-1)
    cv2.circle(seed,(x,y),5,255,-1)
    cv2.circle(theta,(x,y),5,1,-1)
  elif event == cv2.EVENT_RBUTTONDOWN:
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    cv2.circle(seed,(x,y),5,0,-1)
    cv2.circle(theta,(x,y),5,1,-1)
  elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
    cv2.circle(image,(x,y),5,(0,0,255),-1)
    cv2.circle(seed,(x,y),5,255,-1)
    cv2.circle(theta,(x,y),5,1,-1)
  elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    cv2.circle(seed,(x,y),5,0,-1)
    cv2.circle(theta,(x,y),5,1,-1)

  cv2.imshow('image', image)
  cv2.imshow('seed', seed)
  cv2.imshow('theta', theta)


# Load image
image = cv2.imread('flowers.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
seed = np.full(gray.shape, 128, dtype=np.uint8)

if image is None:
  raise ValueError('Image loading failed.')

# Resize image
# height, width = image.shape[:2]
# half_width = width // 3
# half_height = height // 3

# image = image[:half_height, 30:half_width]
# gray = gray[:half_height, 30:half_width]
# seed = seed[:half_height, 30:half_width]

theta = np.full(gray.shape, 0, dtype=np.float64)
origin = np.copy(image)

cv2.imshow("image", origin)
cv2.imshow("seed", seed)
cv2.imshow('theta', theta)

max = np.sqrt(255**2+255**2+255**2)

# Set mouse callback
cv2.setMouseCallback('image', drawing)
cv2.waitKey(0)

growcut(origin)
print("finish")


cv2.waitKey(0)
