import face_recognition
import cv2
import numpy as np
import os


class FaceReader():

    def __init__(self) -> None:
        # Read images and store as {"image name": "encoded image"} 
        self.encoded = {}
        for dirpath, dnames, fnames in os.walk("./Images"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = face_recognition.load_image_file("Images/" + f)
                    encoding = face_recognition.face_encodings(face)[0]
                    self.encoded[f.split(".")[0]] = encoding

    def read_face(self):
        print('Encoding Start...')
        faces = self.encoded
        print('Encoding Complete.')

        faces_encoded = list(faces.values())
        known_face_names = list(faces.keys())

        # define a video capture object
        cap = cv2.VideoCapture(0)

        while True:
            # Capture the video frame by frame
            success, img = cap.read()

            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cam_faceLoc = face_recognition.face_locations(imgS)
            no = len(cam_faceLoc)
            print("Number of faces:- ", no)

            cam_encode_face = face_recognition.face_encodings(imgS)

            for encodeFace,faceLoc in zip(cam_encode_face, cam_faceLoc):
                matches = face_recognition.compare_faces(faces_encoded, encodeFace)
                faceDis = face_recognition.face_distance(faces_encoded, encodeFace)
                best_match_index = np.argmin(faceDis)

                # Add name and box
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
                    cv2.rectangle(img, (x1,y2-25), (x2,y2), (0,255,0), cv2.FILLED)                            # rectangle for name space
                    cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)      # Text of name

            cv2.imshow('Webcam', img)

            # Stop app using 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    FaceReader().read_face()
