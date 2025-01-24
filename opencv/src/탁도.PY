import cv2
import numpy as np

def calculate_turbidity(image_path):
    # 1. 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("이미지를 찾을 수 없습니다.")
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 노이즈 제거 (Gaussian 블러 적용)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. 밝기 값의 표준편차 계산
    mean_intensity = np.mean(blurred)  # 평균 밝기
    std_dev_intensity = np.std(blurred)  # 밝기의 표준편차 (탁도 지표)
    
    # 5. 탁도 계산 (표준편차 값을 0~100 범위로 정규화)
    turbidity = min(100, max(0, std_dev_intensity / 255 * 100))
    
    # 결과 출력
    print(f"평균 밝기: {mean_intensity:.2f}")
    print(f"밝기 표준편차 (탁도 지표): {std_dev_intensity:.2f}")
    print(f"계산된 탁도: {turbidity:.2f}")
    
    # 결과 시각화 (원본 이미지와 전처리된 이미지 표시)
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return turbidity

# 테스트 실행
image_path = "./files/2.jpg"  # 샘플 이미지 경로
turbidity_value = calculate_turbidity(image_path)
print(f"최종 탁도 값: {turbidity_value:.2f}")