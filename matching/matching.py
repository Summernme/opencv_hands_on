import cv2 as cv

def test_sift(img_gray) :
    sift = cv.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(img_gray, None)
    img_draw = cv.drawKeypoints(img, keypoints, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite("SIFT.jpeg", img_draw)
    # 장점 : 이미지 피라미드를 통해 영상 속 객체가 다양한 스케일에 강인한 특징을 추출할 수 있음.
    # 단점 : 하나의 이미지 만을 사용하는 것이 아니라 서로 다른 크기의 이미지들을 이용해 피라미드를 구성하기 때문에 속도가 느림.

def test_orb(img_gray) :
    orb = cv.ORB_create()
    keypoints, descriptor = orb.detectAndCompute(img_gray, None)
    img_draw = cv.drawKeypoints(img, keypoints, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite("ORB.jpeg", img_draw)
    # 장점 : 특징점 검출로 FAST 알고리즘을 사용하기 때문에 속도가 굉장히 빠름. 따라서 SIFT와 SURF의 대안으로 사용가능.
    # 단점 : ORB알고리즘은 상대적으로 노이즈에 민감함. 조명에 따라 매칭되는 성능이 크게 달라질 수 있다.

def test_surf(img_gray) :
    surf = cv.xfeatures2d.SURF_create(1000, 3)
    keypoints, descriptor = surf.detectAndCompute(img_gray, None)
    img_draw = cv.drawKeypoints(img, keypoints, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite("SURF.jpeg", img_draw)
    # 장점 : 다양한 스케일에 대응하기 위해 이미지 피라미드를 사용하지 않고 필터의 크기를 변화시킴으로써 속도가 빠름.
    # 단점 : 그레이 스케일의 이미지를 사용하기 때문에 컬러 공간상에서 다양한 필터의 취득이 어려움.

img = cv.imread("input.jpeg")
if (img is None) : raise "failed to load the image."
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

test_sift(img_gray)
test_orb(img_gray)
test_surf(img_gray)

cv.waitKey()
cv.destroyAllWindows()