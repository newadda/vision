import cv2

# 이미지를 읽어옵니다
image = cv2.imread('./files/2.jpg')  # 'image.jpg'를 원하는 이미지 파일 경로로 변경하세요.
cv2.imshow('Image Window', image)

cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
#equalized  = cv2.equalizeHist(gray )

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# 이미지를 화면에 표시합니다
cv2.imshow('Image Window', binary)

# 키 입력 대기 (0이면 무한 대기)
cv2.waitKey(0)



contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.imshow('Image Window', contour)
    cv2.waitKey(0)
    area = cv2.contourArea(contour)  # 면적 계산





# 모든 창을 닫습니다
cv2.destroyAllWindows()