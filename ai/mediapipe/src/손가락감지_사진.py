import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손 감지 모델 설정
hands = mp_hands.Hands(
    static_image_mode=True,      # True: 정지 이미지 처리
    max_num_hands=2,             # 감지할 최대 손 개수
    min_detection_confidence=0.5 # 감지 최소 신뢰도
)

# 확인할 이미지 경로
image_path = "image002.png"  # 처리할 이미지 파일 경로 입력

# 이미지 읽기
image = cv2.imread(image_path)

# BGR 이미지를 RGB로 변환
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mediapipe로 손 감지
result = hands.process(rgb_image)

# 감지 결과 확인
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 랜드마크 좌표 출력
        for i, landmark in enumerate(hand_landmarks.landmark):
            height, width, _ = image.shape
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    print("손이 감지되었습니다!")
else:
    print("손이 감지되지 않았습니다.")

# 결과 이미지 표시
cv2.imshow('Hand Detection', image)

# 이미지 저장
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
print(f"결과 이미지 저장 완료: {output_path}")

# 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
