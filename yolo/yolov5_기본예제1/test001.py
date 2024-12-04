import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# YOLOv5 모델 불러오기 (사전 학습된 'small' 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 감지할 이미지 경로
image_path = 'image.jpg'  # 대상 이미지 파일 경로
image = Image.open(image_path)

# 객체 감지
results = model(image_path)

# 감지된 결과 가져오기
detections = results.pandas().xyxy[0]  # 감지된 객체의 좌표와 정보
print(detections)  # 데이터프레임 형태로 출력

# 결과 시각화
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for _, row in detections.iterrows():
    # 경계 상자 정보
    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    label = f"{row['name']} {row['confidence']:.2f}"
    
    # 경계 상자 그리기
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', lw=2))
    ax.text(x1, y1 - 5, label, color='red', fontsize=12, weight='bold')

plt.axis('off')
plt.savefig('output.png')
plt.show()

