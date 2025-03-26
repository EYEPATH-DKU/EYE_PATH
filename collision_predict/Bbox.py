import cv2
from ultralytics import YOLO

# YOLO 모델 로드
#model = YOLO("best_all_label.pt")
#model = YOLO("yolo11x.pt")
model = YOLO("last11.pt")

# 비디오 파일 열기
source = "2.mp4"
cap = cv2.VideoCapture(source)

# 비디오 저장을 위한 설정
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 비디오 코덱 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))      # 원본 비디오의 FPS 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 비디오가 열려 있는지 확인
if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

# 프레임별 탐지 실행
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오 끝까지 도달하면 종료

    # YOLO 모델로 객체 탐지
    results = model(frame)

    # 결과 그리기
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            conf = box.conf[0]  # 신뢰도 점수
            class_id = int(box.cls[0])  # 클래스 ID
            label = f"{model.names[class_id]} {conf:.2f}"  # 클래스명 + 신뢰도 표시

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLO Object Detection", frame)

    # 비디오 저장
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()




# 그림 확인
# ll = [f"ex{i}.png" for i in range(1,5)]
# results = model(ll)

# # Process results list
# i = 0
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename=f"result{i}_a.jpg")  # save to disk
#     i += 1
