import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손 감지 모델 설정
hands = mp_hands.Hands(
    static_image_mode=False,  # False: 동영상 입력, True: 정지 이미지 입력
    max_num_hands=2,         # 감지할 최대 손 개수
    min_detection_confidence=0.5,  # 감지 최소 신뢰도
    min_tracking_confidence=0.5    # 추적 최소 신뢰도
)

# 웹캠 입력
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe로 손 감지
    result = hands.process(rgb_frame)

    # 손 랜드마크 그리기
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 랜드마크 좌표 출력
            for i, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # 결과 화면 출력
    cv2.imshow('Hand Tracking', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()