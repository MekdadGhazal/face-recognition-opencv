import cv2
import time
import os
import numpy as np
import json
import csv
from datetime import datetime

"""
Global Variable
"""

# url = "http://192.168.1.102:8080/video"
# cap = cv2.VideoCapture(url)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONFIDENCE_THRESHOLD = 80 

config = {
    "face_cascade" :cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ),
    "closed" : False,
    "count" : 0,
    "counts": {},
    "max_images" : 50,
    "train" : True,
    "next" : True,
    "important" : True,
    "name" : '',
    "choice" : False,
}

recognized_people = {}  # {name: last_seen_timestamp}
log_interval = 3  # ثواني بعد آخر ظهور قبل تسجيل "اختفى"

"""
Cotrol the Video
"""
def controls(frame):
    global config
    key = cv2.waitKey(1) & 0XFF
    if key == ord('x'):
        print('Closed')

        config['closed'] = True
        config['important'] = True

    elif key == ord('s'):
        folder = os.path.join( 'data', 'raw')
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f'screenshot_{int(time.time())}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")


    elif key == ord('y'):
        config['name'] = input('Enter the name: ')
        save_label(config['name'])
        
        config['choice'] = True

    elif key == ord('n'):
        if config['next'] :
            config['next'] = False
        else :
            config['next'] = True
        

"""
Detecting face from 50 taken photos
"""
def detect_faces_live(frame, name = "Mekdad"):
    global config


    try:
        config['next'] = True

        if name not in config['counts']:
            config['counts'][name] = 0
        
        folder = os.path.join('data', 'faces', name)
        os.makedirs(folder, exist_ok=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = config['face_cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            file_name = os.path.join(folder, f"{config['counts'][name]}.jpg")
            cv2.imwrite(file_name, face_img)
            print(f"Saved: {file_name}")
            config['counts'][name] += 1

            if config['counts'][name] >= config['max_images']:
                print(f"Reached max images for {name}")
                config['closed'] = True
                break

        # cv2.imshow('Face Collector', frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            print('Closed')
            config['closed'] = True

    except Exception as e:
        print(f"Error during face detection or saving images: {e}")



"""
Train Model
"""
def train_model(model_path='model.yml', labels_file='labels.json'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images = []
    labels_list = []

    if not os.path.exists(labels_file):
        print("labels.json not found.")
        return

    with open(labels_file, 'r') as f:
        label_dict = json.load(f)

    for name, label_id in label_dict.items():
        folder = os.path.join( 'data', 'faces', name)
        if not os.path.exists(folder):
            print(f"No folder for {name}")
            continue

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                print(f"Invalid file: {filename}")
                continue
            
            # توحيد حجم الصورة
            img_resized = cv2.resize(img, (200, 200))
            
            # تحسين التباين (اختياري)
            img_enhanced = cv2.equalizeHist(img_resized)

            images.append(img_enhanced)
            labels_list.append(label_id)

    if not images:
        print("No valid images found for training.")
        return

    recognizer.train(images, np.array(labels_list))
    recognizer.save(model_path)
    print(f"Trained model saved to {model_path}")


"""
Save Names
"""
def save_label(name):
    label_file = 'labels.json'
    labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)

    if name not in labels:
        labels[name] = len(labels)

    with open(label_file, 'w') as f:
        json.dump(labels, f)



"""
Run the model
"""
def run(frame, recognizer, reverse_labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = config['face_cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)

        if confidence < CONFIDENCE_THRESHOLD:
            # name = reverse_labels.get(label, "Unknown")
            name = get_name_by_label(label)
        else:
            name = "Unknown"

        handle_recognition(name, confidence)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} ({int(confidence)})', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cleanup_disappeared()


def get_name_by_label(label):
    label_file = 'labels.json'
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    for name, lbl in labels.items():
        if lbl == label:
            return name
    return "Unknown"


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
# log_txt_file = os.path.join(log_dir, 'logs.txt')
log_csv_file = os.path.join(log_dir, 'log.csv')

# def log_event(name: str, status: str):
#     """سجل ظهور أو اختفاء الشخص في logs.txt"""
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     with open(log_txt_file, 'a', encoding='utf-8') as f:
#         f.write(f"[{timestamp}] {name} - {status}\n")


def log_recognition_event(name: str, confidence: float, status: str):
    """سجل التعرف على الشخص في log.csv"""
    new_entry = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, round(confidence, 2), status]
    file_exists = os.path.isfile(log_csv_file)

    with open(log_csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Name', 'Confidence', 'status'])
        writer.writerow(new_entry)


def handle_recognition(name: str, confidence: float):
    """تسجيل عند الظهور الأول أو عند العودة بعد انقطاع"""
    now = time.time()
    if name not in recognized_people or now - recognized_people[name] > log_interval:
        # log_event(name, "Appeared")
        log_recognition_event(name, confidence, "Appeared")

    recognized_people[name] = now


def cleanup_disappeared():
    """تسجيل اختفاء الأشخاص بعد مرور المدة المحددة"""
    now = time.time()
    to_remove = []

    for name, last_seen in recognized_people.items():
        if now - last_seen > log_interval:
            # log_event(name, "Disappeared")
            log_recognition_event(name, 0, "Disappeared")
            to_remove.append(name)

    for name in to_remove:
        del recognized_people[name]



"""
Main Part
"""
def main():
    global config

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = 'model.yml'
    if os.path.exists(model_path):
        recognizer.read(model_path)
    else:
        print("Model not found, please train first.")
        return

    with open('labels.json', 'r') as f:
        label_dict = json.load(f)
    reverse_labels = {v: k for k, v in label_dict.items()}

    try:
        while True:
            s, frame = cap.read()
            if not s:
                break

            frame = cv2.flip(frame, 1)

            if config['choice']:
                detect_faces_live(frame, config['name'])

            cv2.putText(frame, 'Press "X" to Exit,and "S" to take a Screenshot', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1)
            cv2.putText(frame, 'Press "N" to Active Detection,and "Y" to Recognize a new face', (10,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1)

            if not config['next']:
                run(frame, recognizer, reverse_labels)

            cv2.imshow('Live', frame)
            controls(frame)

            if config['closed']:
                if config['train'] or config['choice']:
                    train_model()
                    print('Run the code again without Test mode')
                    break
                if config['important']:
                    break
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__' :
    model_path = os.path.join( 'model.yml')
    config['train'] = not os.path.isfile(model_path)
    # if "model.yml" not in os.listdir('face_ai_project'):
    #     train = True
    # else:
    #     train = False

    main()