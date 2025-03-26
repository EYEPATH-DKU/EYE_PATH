# import os
# import csv
# from collections import deque
# import cv2
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ############################################
# # 0. 하이퍼파라미터 & 설정
# ############################################
# MAX_QUEUE_SIZE = 10  # 사용자가 500~1000 사이로 조절 가능
# INPUT_DIM_PER_OBJECT = 5  # [time, x, y, w, h] 등 -> 5개

# # Dense Layer 구조: 2560 -> 512 -> 128 -> 32 -> 2
# HIDDEN1 = 512
# HIDDEN2 = 128
# HIDDEN3 = 32
# OUTPUT_DIM = 2  # ON/OFF

# # 학습 관련
# NUM_EPOCHS = 1000
# BATCH_SIZE = 1  # 여기서는 queue 꽉 찰 때마다 1 step씩 업데이트
# LEARNING_RATE = 1e-5

# # CSV 경로 (train/test 분리)
# TRAIN_CSV_PATH = './frame_labeled_data_train.csv'
# TEST_CSV_PATH = './frame_labeled_data_test.csv'

# # 이미지 폴더 구조 예: images/train/1/..., images/test/2/...
# # 편의상 아래처럼 두 경로로 구분 가능
# TRAIN_IMAGES_ROOT_DIR = './images/train'
# TEST_IMAGES_ROOT_DIR = './images/test'

# ############################################
# # 1. 모델 정의 (단순 FC 예시)
# ############################################
# class DenseModel(nn.Module):
#     def __init__(self, queue_size=512, input_dim_per_obj=5):
#         """
#         queue_size * input_dim_per_obj = Flatten input dimension.
#         예: 512*5 = 2560
#         """
#         super().__init__()
#         self.input_dim = queue_size * input_dim_per_obj
        
#         self.fc1 = nn.Linear(self.input_dim, HIDDEN1)
#         self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
#         self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
#         self.fc4 = nn.Linear(HIDDEN3, OUTPUT_DIM)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         # x shape: (batch, input_dim)  예: (batch, 2560)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         out = self.fc4(x)  # (batch, 2)
#         return out

# ############################################
# # 2. YOLO(가정) 추론 함수 (Placeholder)
# ############################################
# def yolo_inference(image):
#     """
#     실제론 ultralytics, yolov5, etc. 모델을 로드해서 추론해야 합니다.
#     여기서는 개념 예시로, 임의의 bbox 결과를 반환.
    
#     return: list of (x, y, w, h)  (0~1 범위라 가정)
#     """
#     # 예: 임의로 [ (0.1, 0.2, 0.05, 0.1), (0.4, 0.5, 0.2, 0.2) ] 
#     dummy_bboxes = [
#         (0.1, 0.2, 0.05, 0.1),
#         (0.4, 0.5, 0.2, 0.2),
#     ]
#     return dummy_bboxes

# ############################################
# # 3. 학습/검증용 공통 함수
# ############################################

# def run_one_epoch(model, optimizer, csv_path, images_root_dir, device='cpu', is_train=True):
#     """
#     한 에폭 동안 CSV를 순회하며 모델을 학습(또는 평가)하는 함수.
#     - queue 기반 로직
#     - is_train=True면 학습(backprop), False면 평가(추론만)
#     - 반환: epoch가 끝난 뒤의 (avg_loss, all_preds, all_labels)
#       all_preds, all_labels는 각 step마다 예측/실제 라벨을 모아둔 리스트
#     """
#     if is_train:
#         model.train()
#     else:
#         model.eval()
    
#     data_queue = deque(maxlen=MAX_QUEUE_SIZE)
#     criterion = nn.CrossEntropyLoss()
    
#     running_loss = 0.0
#     step_count = 0
    
#     all_preds = []
#     all_labels = []
    
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)  # video, frame, time, label
        
#         for row in reader:
#             video_name = row['video']  # 예: "1.mp4"
#             frame_idx = int(row['frame'])
#             time_sec = float(row['time'])
#             label = int(row['label'])  # 0/1
            
#             # 영상 디렉토리 (train/test 구분), video_name에서 확장자 제거
#             base_name, _ = os.path.splitext(video_name)  # e.g. "1"
#             img_filename = f"{base_name}_{frame_idx}.jpg"
#             img_path = os.path.join(images_root_dir, base_name, img_filename)
            
#             # 이미지 로드
#             if not os.path.exists(img_path):
#                 # 실제론 여기서 continue보다 에러처리할 수도
#                 continue
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue
            
#             # YOLO 추론
#             bboxes = yolo_inference(img)
            
#             # bbox마다 queue에 삽입
#             for (x, y, w, h) in bboxes:
#                 data_queue.append((time_sec, x, y, w, h))
            
#             # queue가 꽉 찼는지 확인
#             if len(data_queue) == MAX_QUEUE_SIZE:
#                 arr = np.array(data_queue, dtype=np.float32)  # (512, 5)
#                 flat = arr.flatten()[None, ...]               # (1, 2560)
                
#                 X = torch.tensor(flat, dtype=torch.float).to(device)
#                 y_true = torch.tensor([label], dtype=torch.long).to(device)
                
#                 with torch.set_grad_enabled(is_train):
#                     logits = model(X)  # (1, 2)
#                     loss = criterion(logits, y_true)
                    
#                     if is_train:
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
                
#                 # 예측 결과
#                 y_pred = logits.argmax(dim=1).item()
                
#                 # 기록
#                 running_loss += loss.item()
#                 step_count += 1
                
#                 all_preds.append(y_pred)
#                 all_labels.append(label)
    
#     if step_count > 0:
#         avg_loss = running_loss / step_count
#     else:
#         avg_loss = 0.0
    
#     return avg_loss, all_preds, all_labels

# ############################################
# # 4. 메인 (학습 + 검증 + 지표 + Plot)
# ############################################
# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 모델 & 옵티마이저
#     model = DenseModel(queue_size=MAX_QUEUE_SIZE,
#                        input_dim_per_obj=INPUT_DIM_PER_OBJECT).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # epoch별 기록용
#     train_loss_history = []
#     test_loss_history = []
#     train_acc_history = []
#     test_acc_history = []
#     train_f1_history = []
#     test_f1_history = []
    
#     for epoch in range(1, NUM_EPOCHS+1):
#         print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        
#         # 1) Train
#         train_loss, train_preds, train_labels = run_one_epoch(
#             model, optimizer,
#             csv_path=TRAIN_CSV_PATH,
#             images_root_dir=TRAIN_IMAGES_ROOT_DIR,
#             device=device,
#             is_train=True
#         )
        
#         # 정확도, 정밀도, 재현율, F1 계산
#         if len(train_preds) > 0:
#             train_acc = accuracy_score(train_labels, train_preds)
#             prec, recall, f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='binary')
#         else:
#             train_acc, prec, recall, f1 = 0, 0, 0, 0
        
#         # 2) Test
#         test_loss, test_preds, test_labels = run_one_epoch(
#             model, optimizer,
#             csv_path=TEST_CSV_PATH,
#             images_root_dir=TEST_IMAGES_ROOT_DIR,
#             device=device,
#             is_train=False
#         )
        
#         if len(test_preds) > 0:
#             test_acc = accuracy_score(test_labels, test_preds)
#             prec_test, recall_test, f1_test, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
#         else:
#             test_acc, prec_test, recall_test, f1_test = 0, 0, 0, 0
        
#         # 출력
#         print(f"[Train] loss={train_loss:.4f}, acc={train_acc:.4f}, prec={prec:.4f}, rec={recall:.4f}, f1={f1:.4f}")
#         print(f"[Test]  loss={test_loss:.4f}, acc={test_acc:.4f}, prec={prec_test:.4f}, rec={recall_test:.4f}, f1={f1_test:.4f}")
        
#         # 기록
#         train_loss_history.append(train_loss)
#         test_loss_history.append(test_loss)
#         train_acc_history.append(train_acc)
#         test_acc_history.append(test_acc)
#         train_f1_history.append(f1)
#         test_f1_history.append(f1_test)
        
#     # 학습 완료 후 최종 confusion matrix 예시
#     if len(test_preds) > 0:
#         cm = confusion_matrix(test_labels, test_preds)
#         print("Confusion Matrix (Test):")
#         print(cm)
    
#     # Plot (epoch별 loss & accuracy 등)
#     epochs_range = range(1, NUM_EPOCHS+1)
    
#     fig, axs = plt.subplots(1, 2, figsize=(12,5))
    
#     # 왼쪽 그래프: Loss
#     axs[0].plot(epochs_range, train_loss_history, label='Train Loss')
#     axs[0].plot(epochs_range, test_loss_history, label='Test Loss')
#     axs[0].set_title('Loss per Epoch')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend()
    
#     # 오른쪽 그래프: Accuracy
#     axs[1].plot(epochs_range, train_acc_history, label='Train Acc')
#     axs[1].plot(epochs_range, test_acc_history, label='Test Acc')
#     axs[1].set_title('Accuracy per Epoch')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Accuracy')
#     axs[1].legend()
    
#     plt.tight_layout()
#     plt.show()
    
#     # 필요하면 F1 그래프도 그릴 수 있음
#     # 예: plt.plot(epochs_range, train_f1_history, label='Train F1')
#     #     plt.plot(epochs_range, test_f1_history, label='Test F1')
#     #     ...

# if __name__ == '__main__':
#     main()


import os
import csv
from collections import deque
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

############################################
# 0. 하이퍼파라미터 & 설정
############################################
MAX_QUEUE_SIZE = 512  # 사용자가 500~1000 사이로 조절 가능
INPUT_DIM_PER_OBJECT = 5  # [time, x, y, w, h] 등

# 예: "512 x 5 -> 2560 -> 512 -> 128 -> 32 -> 2"
HIDDEN1 = 512
HIDDEN2 = 128
HIDDEN3 = 32
OUTPUT_DIM = 2  # ON/OFF

FRAME_CSV_PATH = './frame_labeled_data.csv'
IMAGES_ROOT_DIR = './images'

############################################
# 1. 모델 정의 (단순 FC 예시)
############################################
class DenseModel(nn.Module):
    def __init__(self, queue_size=512, input_dim_per_obj=5):
        """
        queue_size * input_dim_per_obj = Flatten input dimension.
        예: 512*5 = 2560
        """
        super().__init__()
        self.input_dim = queue_size * input_dim_per_obj
        
        self.fc1 = nn.Linear(self.input_dim, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
        self.fc4 = nn.Linear(HIDDEN3, OUTPUT_DIM)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, input_dim)  예: (batch, 2560)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)  # (batch, 2)
        return out

############################################
# 2. YOLO(가정) 추론 함수 (Placeholder)
############################################
def yolo_inference(image):
    """
    실제론 ultralytics, yolov5, etc. 모델을 로드해서 추론해야 합니다.
    여기서는 개념 예시로, 임의의 bbox 결과를 반환.
    
    return: list of (x, y, w, h)
    """
    # 예: 임의로 [ (0.1, 0.2, 0.05, 0.1), (0.4, 0.5, 0.2, 0.2), ... ]
    dummy_bboxes = [
        (0.1, 0.2, 0.05, 0.1),
        (0.4, 0.5, 0.2, 0.2),
    ]
    return dummy_bboxes

############################################
# 3. 큐를 사용하여 "1회" 전체 CSV를 순회하는 학습 예시
############################################
def process_video_sequence(model, optimizer, device='cpu'):
    """
    - frame_csv 를 순차적으로 읽으며, 이미지 불러오기 -> YOLO추론 -> 큐에 (time, x, y, w, h) 추가
    - 큐가 가득 차면(=MAX_QUEUE_SIZE)의 배치를 하나 만든 후, label과 함께 학습/추론
    - 모든 프레임 처리가 끝나면, 전체 예측/실제 라벨을 모아 Precision, Recall, F1, Accuracy 출력
    """
    model.train()
    
    data_queue = deque(maxlen=MAX_QUEUE_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    step_count = 0
    
    # 정밀도/재현율/F1 계산 위해 step별 예측/실제 라벨을 저장
    all_preds = []
    all_labels = []
    
    with open(FRAME_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # video, frame, time, label
        
        for row in reader:
            video_name = row['video']  # 예: "1.mp4"
            frame_idx = int(row['frame'])
            time_sec = float(row['time'])
            label = int(row['label'])  # 0 or 1
            
            # 이미지 경로
            base_name, _ = os.path.splitext(video_name)  # ("1", ".mp4")
            img_filename = f"{base_name}_{frame_idx}.jpg"
            img_path = os.path.join(IMAGES_ROOT_DIR, base_name, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: image not found {img_path}")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: failed to read {img_path}")
                continue
            
            # YOLO 추론
            bboxes = yolo_inference(img)  # list of (x, y, w, h)
            
            # bbox마다 queue에 삽입
            for (x, y, w, h) in bboxes:
                data_queue.append((time_sec, x, y, w, h))
            
            # queue가 가득 차면 학습 1회 진행
            if len(data_queue) == MAX_QUEUE_SIZE:
                arr = np.array(data_queue, dtype=np.float32)  # (512,5)
                flat = arr.flatten()[None, ...]               # (1, 2560)
                
                X = torch.tensor(flat, dtype=torch.float).to(device)
                y_true = torch.tensor([label], dtype=torch.long).to(device)
                
                # Forward & Backprop
                optimizer.zero_grad()
                logits = model(X)  # (1,2)
                loss = criterion(logits, y_true)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                step_count += 1
                
                # 예측 라벨
                y_pred = logits.argmax(dim=1).item()
                all_preds.append(y_pred)
                all_labels.append(label)
                
                # step마다 log
                if step_count % 50 == 0:
                    avg_loss = running_loss / 50
                    print(f"Step {step_count}, loss={avg_loss:.4f}")
                    running_loss = 0.0
    
    # 전체 CSV 처리 끝
    print("Training loop finished.")
    
    # 만약 학습 step이 전혀 없었다면 => 지표 계산 불가
    if step_count == 0:
        print("No training steps occurred. Possibly queue never filled up.")
        return
    
    # 전체 예측/실제 라벨로 Precision, Recall, F1, Accuracy 계산
    # (여기서는 binary 분류라 average='binary' 사용)
    accuracy = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Precision= {prec:.4f}, Recall= {rec:.4f}, F1= {f1:.4f}")

############################################
# 4. 메인 (학습 예시)
############################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 생성
    model = DenseModel(queue_size=MAX_QUEUE_SIZE,
                       input_dim_per_obj=INPUT_DIM_PER_OBJECT).to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # train
    process_video_sequence(model, optimizer, device=device)
    
    # (추가) test 세트도 비슷한 구조로 "model.eval()" 후 예측, 지표 계산 가능

if __name__ == '__main__':
    main()




# import os
# import csv
# from collections import deque
# import cv2
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# ############################################
# # 0. 하이퍼파라미터 & 설정
# ############################################
# MAX_QUEUE_SIZE = 512  # 사용자가 500~1000 사이로 조절 가능
# INPUT_DIM_PER_OBJECT = 5  # [time, x, y, w, h] 등
# # 여기서 time을 포함해 5개로 가정 (collision_area 등 더 넣으면 증가 가능)

# # 예: "512 x 5 -> 2560 -> 512 -> 128 -> 32 -> 2"
# HIDDEN1 = 512
# HIDDEN2 = 128
# HIDDEN3 = 32
# OUTPUT_DIM = 2  # ON/OFF

# FRAME_CSV_PATH = './frame_labeled_data.csv'  # 예: preprocess.py에서 생성한 CSV
# IMAGES_ROOT_DIR = './images'  # 이미지들이 "images/<video>/<video_frame>.jpg" 식으로 존재

# # 만약 train/test로 폴더를 나눈다면:
# # IMAGES_ROOT_DIR/train/1, IMAGES_ROOT_DIR/train/2, ...
# # IMAGES_ROOT_DIR/test/1, IMAGES_ROOT_DIR/test/2, ...
# # 이런 식으로 관리할 수도 있음


# ############################################
# # 1. 모델 정의 (단순 FC 예시)
# ############################################
# class DenseModel(nn.Module):
#     def __init__(self, queue_size=512, input_dim_per_obj=5):
#         """
#         queue_size * input_dim_per_obj = Flatten input dimension.
#         예: 512*5 = 2560
#         """
#         super().__init__()
#         self.input_dim = queue_size * input_dim_per_obj
        
#         self.fc1 = nn.Linear(self.input_dim, HIDDEN1)
#         self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
#         self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
#         self.fc4 = nn.Linear(HIDDEN3, OUTPUT_DIM)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         # x shape: (batch, input_dim)  예: (batch, 2560)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         out = self.fc4(x)  # (batch, 2)
#         return out


# ############################################
# # 2. YOLO(가정) 추론 함수 (Placeholder)
# ############################################
# def yolo_inference(image):
#     """
#     실제론 ultralytics, yolov5, etc. 모델을 로드해서 추론해야 합니다.
#     여기서는 개념 예시로, 임의의 bbox 결과를 반환.
    
#     return: list of (x, y, w, h)
#     """
#     # 예: 임의로 [ (0.1, 0.2, 0.05, 0.1), (0.4, 0.5, 0.2, 0.2), ... ] 
#     # 실제론 model(image) -> bboxes...
#     dummy_bboxes = [
#         (0.1, 0.2, 0.05, 0.1),
#         (0.4, 0.5, 0.2, 0.2),
#     ]
#     return dummy_bboxes


# ############################################
# # 3. 큐를 사용하여 실시간(?) 처리 예시
# ############################################
# def process_video_sequence(model, optimizer, device='cpu'):
#     """
#     - frame_csv 를 순차적으로 읽으며, 이미지 불러오기 -> YOLO추론 -> 큐에 (time, x, y, w, h) 추가
#     - 큐가 가득 차면(=MAX_QUEUE_SIZE)의 배치를 하나 만든 후, label과 함께 학습/추론
#     """
#     # 모델 학습 모드
#     model.train()
    
#     # (time, x, y, w, h)를 저장하는 큐. bounding box 한 개가 queue의 한 원소가 됨.
#     data_queue = deque(maxlen=MAX_QUEUE_SIZE)
    
#     # frame_csv 예: [video, frame, time, label]
#     # 여기서 label은 현재 프레임의 ON/OFF이라고 가정.
#     # (time은 float초, label은 0 또는 1)
#     # train/test 분리를 위해, 실제론 train용 csv와 test용 csv를 별도로 관리할 수도 있음.
#     with open(FRAME_CSV_PATH, 'r', encoding='utf-8') as f:
#         # CSV 헤더: video, frame, time, label
#         reader = csv.DictReader(f)
        
#         # 임시적으로 Loss 계산을 위해
#         criterion = nn.CrossEntropyLoss()
#         running_loss = 0.0
#         step_count = 0
        
#         for row in reader:
#             video_name = row['video']  # "1.mp4"
#             frame_idx = int(row['frame'])
#             time_sec = float(row['time'])
#             label = int(row['label'])  # 0/1
            
#             # 1) 이미지 경로 추론
#             #    ex) images/1/1_334.jpg
#             base_name, _ = os.path.splitext(video_name)  # "1.mp4" -> ("1", ".mp4")
#             img_filename = f"{base_name}_{frame_idx}.jpg"
#             img_path = os.path.join(IMAGES_ROOT_DIR, base_name, img_filename)
            
#             # 2) 이미지 로드
#             if not os.path.exists(img_path):
#                 print(f"Warning: image not found {img_path}")
#                 continue
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Warning: failed to read {img_path}")
#                 continue
            
#             # 3) YOLO 추론 (모든 객체 bbox 뽑기)
#             bboxes = yolo_inference(img)  # list of (x, y, w, h)
            
#             # 4) bbox마다 queue에 삽입
#             #    queue에 들어가는 1개 원소 예: (time_sec, x, y, w, h)
#             for (x, y, w, h) in bboxes:
#                 data_queue.append((time_sec, x, y, w, h))
            
#             # 5) 큐가 가득 찼는지 확인
#             if len(data_queue) == MAX_QUEUE_SIZE:
#                 # 이제 모델에 넣을 입력 벡터 만들기
#                 # 큐의 모든 원소 -> numpy or tensor화
#                 # data_queue: MAX_QUEUE_SIZE x 5
#                 arr = np.array(data_queue, dtype=np.float32)  # shape: (512, 5) 가정
#                 # Flatten -> (1, 512*5)
#                 flat = arr.flatten()[None, ...]  # shape (1, 2560)
                
#                 X = torch.tensor(flat, dtype=torch.float).to(device)
#                 y = torch.tensor([label], dtype=torch.long).to(device)  # 현재 프레임의 라벨
                
#                 # 6) Forward & Backprop (학습)
#                 optimizer.zero_grad()
#                 logits = model(X)  # shape (1,2)
#                 loss = criterion(logits, y)
#                 loss.backward()
#                 optimizer.step()
                
#                 running_loss += loss.item()
#                 step_count += 1
                
#                 if step_count % 50 == 0:
#                     avg_loss = running_loss / 50
#                     print(f"Step {step_count}, loss={avg_loss:.4f}")
#                     running_loss = 0.0
                
#                 # 레이블 시간 지연처리(추가 설명):
#                 # 실제로 ON/OFF 상태가 "현재 프레임보다 약간 뒤"에 확정될 수도 있음.
#                 # 그럴 경우, 지금의 y를 '미래 프레임의 라벨'로 매핑하거나,
#                 # 혹은 queue가 꽉 찬 후, 한두 프레임 뒤에야 label 확정... 등의 로직이 필요.
#                 # 프로젝트 목적에 따라 time shift를 적용할 수 있음.
                
#                 # (참고) 시계열에서 "이전 5초 데이터 -> 현재 라벨 예측"이라면 이대로 충분.
#                 # 만약 "이전 5초 데이터 -> 앞으로 1초 뒤 라벨"이라면 label을 1초 뒤 프레임의 것을 사용.
#                 # 이런 식으로 time shift 로직만 바꾸면 됨.

#     print("Training loop finished.")


# ############################################
# # 4. 메인 (학습 예시)
# ############################################
# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 모델 생성
#     model = DenseModel(queue_size=MAX_QUEUE_SIZE,
#                        input_dim_per_obj=INPUT_DIM_PER_OBJECT)
#     model.to(device)
    
#     # 옵티마이저
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
#     # train
#     process_video_sequence(model, optimizer, device=device)
    
#     # 이후 test 세트도 비슷한 구조로 "학습 대신 추론"만 수행하는 코드를 짤 수 있음.
#     # ex) model.eval() 상태에서 loss 계산 or 정확도 측정

# if __name__ == '__main__':
#     main()
