import os
import cv2
import numpy as np
import json
import time


class FaceRecognizerApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.CONFIDENCE_THRESHOLD = 80
        
        self.user_name = ''
        self.labels = {}
        self.image_count = 0
        self.max_images = 50
        self.label_file = 'labels.json'
        self.main_model_path = 'model.yml'
        
        self.mode = 'running' # all mode is : 'collecting', 'recognizing', 'running', or 'exit'

        try:
            with open(self.label_file, 'r') as f:
                # opposit the dict : {name: id} -> {id: name} 
                original_labels = json.load(f)
                self.labels = {v: k for k, v in original_labels.items()}
        except FileNotFoundError:
            print("labels.json not found. Please register a user first.")


    def run_app(self):
        self.recognizer.read(self.main_model_path)

        while True:
            s, frame = self.cap.read()
            if not s:
                break
            
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, 'Press "X" to Exit,and "S" to take a Screenshot', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1)
            cv2.putText(frame, 'Press "N" to Active Detection,and "Y" to Recognize a new face', (10,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1)

            processed_frame = frame

            if self.mode == 'collecting':
                processed_frame = self.collect_faces(frame)
            elif self.mode == 'recognizing':
                processed_frame = self.recognize_faces(frame)

            self.handle_display_and_keys(processed_frame)

            if self.mode == 'exit':
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


    def get_frame_with_rectangles(self, frame, faces):
        # copy the current frame
        output_frame = frame.copy()

        # loop to dectect all faces in faces (using rectangle)
        for (x, y, w, h) in faces:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_frame

    def get_gray_and_faces(self, frame ):
        # convert frame to Gray to be easy to proccess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in the gray frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        return gray, faces


    def collect_faces(self, frame):
        # create the main folder to save images using input username 
        folder = os.path.join('data', 'faces', self.user_name)
        os.makedirs(folder, exist_ok=True)

        gray, faces = self.get_gray_and_faces(frame)

        # See faces using get_frame_with_rectangles [not printed]
        output_frame = self.get_frame_with_rectangles(frame, faces)


        for (x, y, w, h) in faces:
            # Cut the frame 
            face_img = frame[y:y+h, x:x+w]
            
            # create a file as img 
            file_name = os.path.join(folder, f"{self.image_count}.jpg")

            # save current cutted face from frame in that file
            cv2.imwrite(file_name, face_img)

            # console log
            print(f"Saved: {file_name}")

            self.image_count += 1

            if self.image_count > self.max_images :
                print(f"Finished collecting {self.max_images} images for {self.user_name}.")

                # Return the mode [status] to running  and reset image_count to zero
                self.image_count = 0
                self.mode = 'running'

                # Now train the model with the new entry
                self.train_model()

                return output_frame
            
        return output_frame 

    def recognize_faces(self, frame):
        
        gray, faces = self.get_gray_and_faces(frame)

        output_frame = frame.copy()

        for (x, y, w, h) in faces:
            # Detect gary faces
            face_img = gray[y:y+h, x:x+w]

            # use prediction to predict the person [confidence, label]
            label, confidence = self.recognizer.predict(face_img)
            
            # by default user is "Unknown"
            name = "Unknown"
            if confidence < self.CONFIDENCE_THRESHOLD:
                # get the name from self.labels
                name = self.labels.get(label, "Unknown")

            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output_frame


    def train_model(self):        
        # path for main folder
        base_faces_dir = os.path.join('data', 'faces')

        # check if folder exists
        if not os.path.exists(base_faces_dir):
            print(f"Faces directory not found at '{base_faces_dir}'. Aborting training.")
            return

        # check if lable.json if exists
        if not os.path.exists(self.label_file):
            print(f"Labels file '{self.label_file}' not found. Aborting training.")
            return

        # load the label.json as label_dict
        with open(self.label_file, 'r') as f:
            label_dict = json.load(f)

        # Empty array to store all images with labels
        images = []
        labels_list = []

        print("Starting model training for all users...")
        
        # loop : name and label in label_dict
        for name, label_id in label_dict.items():

            # user pictures saved in a folder with his name
            user_folder = os.path.join(base_faces_dir, name)

            # check if the user_folder exists if not we will ignore it
            if not os.path.isdir(user_folder):
                print(f"Warning: Directory for user '{name}' not found. Skipping.")
                continue

            print(f"-> Processing images for: {name} (Label: {label_id})")

            # loop on all images in the user folder
            for filename in os.listdir(user_folder):

                # to check that file is iamge 
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    # store the image path
                    img_path = os.path.join(user_folder, filename)
                    
                    # convert it to gray mode (easy to process)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # to check it is a valid image
                    if img is None or img.size == 0:
                        print(f"    - Warning: Could not read or empty image: {filename}")
                        continue
                    
                    # add image to images array with its label
                    images.append(img)
                    labels_list.append(label_id)

        # check there are images to train the model
        if not images:
            print("No valid images found to train the model. Training aborted.")
            return

        print(f"\nTraining the model with {len(images)} images across {len(label_dict)} users...")

        # Train the model using 'images' and 'labels' arrays
        # we convert 'labels_list' to numeric array using numpy
        self.recognizer.train(images, np.array(labels_list))

        # save the model in main path
        self.recognizer.save(self.main_model_path)
        
        print(f"Training complete. Model has been updated and saved to '{self.main_model_path}'.")

        
    def save_label(self, name):
        # file name
        label_file = self.label_file
        labels = {}

        # check if file exists or not
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                labels = json.load(f)

        # check if name exists or not 
        if name not in labels:
            labels[name] = len(labels)

        # save file as json
        with open(label_file, 'w') as f:
            json.dump(labels, f)


    def handle_display_and_keys(self, frame):
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            self.mode = 'running' if self.mode == 'recognizing' else 'recognizing'
        elif key == ord('y'):
            self.user_name = input("Enter name: ")
            self.save_label(self.user_name)
            self.mode = 'collecting'
        elif key == ord('x'):
            self.mode = 'exit'
        elif key == ord('s'):
            folder = os.path.join( 'data', 'raw')
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f'screenshot_{int(time.time())}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            
        cv2.imshow("Face Recognition", frame)


if __name__ == "__main__":
    app = FaceRecognizerApp()
    app.run_app()
 # type: ignore