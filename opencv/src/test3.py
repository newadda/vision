import cv2
import numpy as np

# 1. 이미지 읽기
image_path = "./files/2.jpg"  # 입력 이미지 경로
image = cv2.imread(image_path)

# 2. 이미지 변환 (BGR -> HSV)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 파란색 범위 정의 (HSV)
# Hue 값: 100~140 (파란색 계열), Saturation과 Value는 최소-최대 범위로 설정
lower_blue = np.array([130,140,140])  # 파란색 하한
upper_blue = np.array([180,200,200])  # 파란색 상한

# 4. 마스크 생성 (파란색만 검출)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 5. 파란색 제거 (마스크의 반전 사용)
non_blue_mask = cv2.bitwise_not(blue_mask)
result_image = cv2.bitwise_and(image, image, mask=non_blue_mask)




# 6. 결과 저장 및 출력
#output_path = "output_image.jpg"
#cv2.imwrite(output_path, result_image)

cv2.imshow("Original Image", image)
cv2.imshow("Blue Removed", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(f"파란색이 제거된 결과 이미지가 저장되었습니다: {output_path}")