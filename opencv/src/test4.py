import cv2
import numpy as np

# 1. 이미지 읽기
image_path = "./files/2.jpg"  # 바다와 배가 있는 이미지 경로
image = cv2.imread(image_path)

# 2. BGR -> HSV 색 공간 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 바다(파란색 계열)의 HSV 범위 설정
lower_blue = np.array([30, 34, 59])  # 파란색 계열 하한
upper_blue = np.array([40, 36, 63])  # 파란색 계열 상한

# 4. 바다 영역 검출
sea_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
sea_mask = cv2.morphologyEx(sea_mask, cv2.MORPH_CLOSE, kernel)  # 노이즈 제거


# 5. 바다를 흰색으로 대체
result_image = image.copy()
result_image[sea_mask > 0] = [255, 255, 255]  # 바다 영역을 흰색으로 변경

# 6. 결과 저장 및 출력
#output_path = "sea_removed.jpg"
#cv2.imwrite(output_path, result_image)

cv2.imshow("Original Image", image)
cv2.imshow("Sea Removed", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"바다가 흰색으로 대체된 결과 이미지가 저장되었습니다: {output_path}")