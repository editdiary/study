import cv2 as cv
import numpy as np
import time

min_confidence = 0.7
margin = 30
file_name = "parking.jpg"

# Load YOLO
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco_names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]      ## strip(): remove '\n' at the end of the line
print(classes)
layer_names = net.getLayerNames()                           ## getLayerNames(): YOLO 모델의 모든 레이어 이름을 반환
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]  ## getUnconnectedOutLayers(): 출력 레이어의 인덱스를 반환

# Loading image
start_time = time.time()
img = cv.imread(file_name)
height, width, channels = img.shape     ## 원본 이미지의 크기 저장

# Detecting objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
## blobFromImage(): 이미지를 전처리하여 네트워크 입력으로 사용할 수 있는 blob 객체를 생성
## blobFromImage(image, scalefactor, size, mean, swapRB, crop, ddepth)
## (1) image: 입력 이미지
## (2) scalefactor: 일반적으로 YOLO 모델은 입력 이미지의 픽셀 값을 0과 1사이로 정규화하기 때문에 1/255(0.00392)을 사용
## (3) size: 네트워크에 입력될 이미지 크기(YOLO v3 모델은 416x416 크기의 이미지를 받는다)
## (4) mean: 입력 이미지의 RGB 평균 값을 빼기 위한 값
## (5) swapRB: R과 B 채널을 서로 바꿔줄지 여부(OenCV는 기본적으로 BRG을 사용하기에 변환이 필요)
## (6) crop: 이미지를 crop 여부(False: 입력 이미지를 crop하지 않고 지정된 크기(416x416)로 resize)

net.setInput(blob)      ## setInput(): blob 객체를 네트워크의 입력으로 설정
outs = net.forward(output_layers)       ## 신경망 전방 계산 수행(output_layers가 출력한 텐서를 저장)

# Showing informations on the screen
boxes, confidences, ids = [], [], []

for out in outs:                        ## 네트워크의 출력 레이어에서 얻은 결과
    for detection in out:               ## 각 출력 결과는 총 85차원으로 구성되어 있음(x, y, w, h, objectness score, class scores)
        scores = detection[5:]          ## 5번째 이후의 값은 클래스별 확률값(coco에서는 총 80개의 class가 존재)
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # Filter only 'car'
        if class_id == 2 and confidence > min_confidence:       ## class_id가 2이고 confidence가 0.5 이상인 경우만 추출
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            ## center_x, center_y를 박스의 왼쪽 상단 좌표로 변환(이미지의 좌표는 왼쪽 상단이 (0, 0)이라고 한다)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
## NMSBoxes(): Non-Maximum Suppression - 겹치는 박스를 제거하는 알고리즘(가장 적절한 것을 선택하고 나머지는 제거)
## (1) boxes: 박스 좌표  /  (2) confidences: 각 박스의 신뢰도  /  (3) confidence_threshold: 신뢰도 임계값(낮은 신뢰도의 박스는 제거)
## (4) nms_threshold: 겹치는 박스를 제거하기 위한 IOU(Intersection Over Union) 임계값 - 값이 낮을수록 더 많은 상자가 제거
font = cv.FONT_HERSHEY_PLAIN
color = (0, 255, 0)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = f"{confidences[i]: .2f}"
        print(i, label)
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv.putText(img, label, (x, y - 5), font, 1, color, 1)

text = f"Number of Car is : {len(indexes)}"
cv.putText(img, text, (margin, margin), font, 2, color, 2)

cv.imshow("Number of Car - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print(f"=== A frame took {process_time:.3f} seconds")

cv.waitKey(0)
cv.destroyAllWindows()