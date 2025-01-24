import cv2
import numpy as np

# 1. 이미지 읽기
image_path = "./files/2.jpg"  # 하수 물 이미지 경로
image = cv2.imread(image_path)


# 2. 전처리 (그레이스케일 변환 및 블러링)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
blurred = cv2.GaussianBlur(gray, (5, 5), 0)    # 가우시안 블러로 노이즈 제거

# 3. 이진화 (Thresholding)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. 컨투어 검출
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 컨투어를 기반으로 오물 시각화
for contour in contours:
    # 오물의 면적 계산
    area = cv2.contourArea(contour)
    if area > 500:  # 최소 면적 필터링 (작은 노이즈 제거)
        # 경계 상자 그리기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 녹색 박스
        # 면적 표시
        cv2.putText(image, f"Area: {int(area)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 6. 결과 출력
cv2.imshow("Sewage Image - Debris Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 저장
output_path = "detected_sewage_image.jpg"
cv2.imwrite(output_path, image)
print(f"결과 이미지가 저장되었습니다: {output_path}")