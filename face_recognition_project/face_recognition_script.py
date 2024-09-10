import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
import json

# Path to known faces directory and Excel file
KNOWN_FACES_DIR = 'known_faces'
EXCEL_FILE = 'known_faces.xlsx'

def load_known_faces():
    if not os.path.isfile(EXCEL_FILE):
        return pd.DataFrame(columns=['name', 'encoding'])
    
    df = pd.read_excel(EXCEL_FILE)
    df['encoding'] = df['encoding'].apply(lambda x: np.array(json.loads(x)))
    return df

def save_known_faces(df):
    df['encoding'] = df['encoding'].apply(lambda x: json.dumps(x.tolist()))
    df.to_excel(EXCEL_FILE, index=False)

def encode_face_from_image(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        return encodings[0]
    return None

def recognize_face_from_camera():
    known_faces_df = load_known_faces()
    known_face_encodings = known_faces_df['encoding'].tolist()
    known_face_names = known_faces_df['name'].tolist()

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            print("No faces detected")
            continue

        for face_encoding in face_encodings:
            if len(known_face_encodings) > 0:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:  # Threshold for face recognition
                    name = known_face_names[best_match_index]
                    cv2.putText(frame, f"Hello, {name}!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Biometric Face Recognition', frame)
                    print(f"Welcome back, {name}!")
                    continue
                else:
                    print("No match found.")

            # If the face is not recognized, ask for the name only once
            cv2.putText(frame, "Face not recognized!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Biometric Face Recognition', frame)
            print("Face not recognized.")
            
            # Release the video capture temporarily to ask for the user's name
            video_capture.release()
            name = input("Please enter your name: ")
            new_face_encoding = face_encoding

            # Add the new face encoding to the DataFrame and save permanently
            new_entry = {'name': name, 'encoding': new_face_encoding}
            known_faces_df = pd.concat([known_faces_df, pd.DataFrame([new_entry])], ignore_index=True)
            save_known_faces(known_faces_df)

            print(f"Thank you! Your face has been added to the known faces list, {name}.")
            
            # Reload known faces (in case more faces were added)
            known_faces_df = load_known_faces()
            known_face_encodings = known_faces_df['encoding'].tolist()
            known_face_names = known_faces_df['name'].tolist()
            
            video_capture = cv2.VideoCapture(0)  # Re-initialize video capture after user input

        cv2.imshow('Biometric Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face_from_camera()
