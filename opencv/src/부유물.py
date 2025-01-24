import cv2
import numpy as np

def remove_isolated_blue(image_path, output_path):
    # 1. 이미지 읽기
    image = cv2.imread(image_path)

    # 2. BGR -> HSV 색 공간 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 파란색 범위 설정

    lower_blue = np.array([60, 0, 0])  # 
    upper_blue = np.array([130, 150, 200])  # 
    #lower_blue = np.array([150,148,115])  # 파란색 하한 (HSV)
    #upper_blue = np.array([200,153,120])  # 파란색 상한 (HSV)

    # 4. 파란색 영역 검출
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    print("mask",blue_mask[0])

    # 5. 주변 색 확인: 커널 크기 정의
    kernel_size = 2  # 주변 탐색 크기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 6. 이웃에 파란색이 있는지 확인
    dilated_mask = cv2.dilate(blue_mask, kernel)  # 주변 영역 확장
    refined_mask = cv2.bitwise_and(blue_mask, dilated_mask)  # 주변 파란색 영역 유지

    # 7. 파란색 영역을 흰색으로 대체
    result_image = image.copy()
    result_image[refined_mask > 0] = [255, 255, 255]  # 파란색 영역을 흰색으로 변경

    # 8. 결과 저장 및 출력
    #cv2.imwrite(output_path, result_image)
    cv2.imshow("Original Image", image)
    cv2.imshow("Blue Removed", result_image)
    cv2.imshow("ㅅㄷㄴㅅㅅ", cv2.bitwise_xor(image, result_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
  

    print(f"결과 이미지가 저장되었습니다: {output_path}")


# 실행
input_image_path = "./files/1.jpg"  # 원본 이미지 경로
output_image_path = "output_image.jpg"  # 결과 이미지 저장 경로
remove_isolated_blue(input_image_path, output_image_path)