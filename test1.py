import sys
import time
import os
import numpy as np
from PIL import Image
import cv2

# Absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the haarcascade file
haarcascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')

# Absolute path to the user data directory
path = os.path.join(script_dir, 'user_data')

def face_generator():
    """
    Captures user's face images for training data.
    """
    global name
    cam = cv2.VideoCapture(0)  # Used to capture video frames
    cam.set(3, 640)
    cam.set(4, 480)
    detector = cv2.CascadeClassifier(haarcascade_path)

    # Check if the haarcascade file exists
    if not os.path.isfile(haarcascade_path):
        print("Error: Could not find haarcascade_frontalface_default.xml file. Please download and place it in the correct location.")
        return

    face_id = input("Enter ID of user: ")
    name = input("Enter name: ")
    sample = int(input("Enter how many samples you wish to take: "))

    # Create the user data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Remove old images from user data folder if present
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

    print("Taking sample images of user... Please look at camera")
    time.sleep(2)

    count = 0
    while True:
        ret, img = cam.read()  # Read frames
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = detector.detectMultiScale(converted_image, 1.3, 5)  # Detect faces

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle
            count += 1

            cv2.imwrite(os.path.join(path, "face." + str(face_id) + "." + str(count) + ".jpg"),
                        converted_image[y:y + h, x:x + w])  # Save image

            cv2.imshow("image", img)  # Display image

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif count >= sample:
            break

    print("Image samples taken successfully!")
    cam.release()
    cv2.destroyAllWindows()

def training_data():
    """
    Trains the face recognizer using captured face images.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcascade_path)

    # Check if the haarcascade file exists
    if not os.path.isfile(haarcascade_path):
        print("Error: Could not find haarcascade_frontalface_default.xml file. Please download and place it in the correct location.")
        return

    def images_and_labels(path):
        """
        Loads face images and their corresponding labels from a directory.
        """
        images_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in images_paths:
            gray_image = Image.open(image_path).convert('L')  # Convert to grayscale
            img_arr = np.array(gray_image, 'uint8')

            # Extract label (ID) from image filename
            id = int(os.path.split(image_path)[-1].split(".")[1])

            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)

        return face_samples, ids

    print("Training Data...please wait...!!!")
    faces, ids = images_and_labels(path)

    if not faces:
        print("Error: No face data found for training. Please capture face data first using face_generator().")
        return

    # Trains the face recognizer using the loaded data
    recognizer.train(faces, np.array(ids))

    # Save the trained recognizer to a YAML file
    recognizer.save(os.path.join(script_dir, 'trained_data.yml'))

def detection():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained_data_path = os.path.join(script_dir, 'trained_data.yml')
    recognizer.read(trained_data_path)  # loaded trained model
    cascadePath = haarcascade_path
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes fonts size

    id = 5  # number of persons you want to recognize
    names = ['', name]
    cam = cv2.VideoCapture(0)  # used to create video which is used to capture images
    cam.set(3, 640)
    cam.set(4, 480)

    # define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        if cam is None or not cam.isOpened():
            print('Warning: unable to open video source: ')

        ret, img = cam.read()  # read the frames using above created objects
        if ret == False:
            print("unable to detect img")
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts image to black and white

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])
            # check if accuracy is less than 100 ==> "0" is a perfect match
            if accuracy < 100:
                id = names[id]
                accuracy = " {0}%".format(round(100 - accuracy))
            else:
                id = "unknown"
                accuracy = " {0}%".format(round(100 - accuracy))
            cv2.putText(img, "press Esc to close this window", (5, 25), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def permission(val, task):
    if val.lower() == 'y':
        if task == 1:
            face_generator()
        elif task == 2:
            training_data()
        elif task == 3:
            detection()
    else:
        print("Thank you for using this application!")
        sys.exit()

# Ask for permission to perform tasks
val = input("Do you want to perform face data generation (Y/N)? ")
permission(val, 1)

val = input("Do you want to train the face recognizer (Y/N)? ")
permission(val, 2)

val = input("Do you want to perform face detection (Y/N)? ")
permission(val, 3)
