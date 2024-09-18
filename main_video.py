import cv2
import numpy as np
import pandas as pd
from simple_facerec import SimpleFacerec

# Load the existing Excel file, if it exists
try:
    existing_df = pd.read_excel("C:\\Users\\Rahul Gowda\\OneDrive\\Desktop\\Face\\face\\detected_text.xlsx")
except FileNotFoundError:
    existing_df = pd.DataFrame(columns=["Present "])

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Set to store unique detected texts
unique_detected_texts = set()

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Initialize a list to store detected texts other than "Unknown"
    detected_texts = []

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if name != "Unknown" and name not in unique_detected_texts:
            detected_texts.append(name)
            unique_detected_texts.add(name)  # Add the detected text to the set
        
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Show frame with bounding boxes and names
    cv2.imshow("Frame", frame)

    # Append detected texts to the existing Excel file
    if detected_texts:
        new_df = pd.DataFrame({"Present": detected_texts})
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Save the updated DataFrame to the Excel file
existing_df.to_excel("C:\\Users\\Rahul Gowda\\OneDrive\\Desktop\\Face\\face\\detected_text.xlsx", index=False)

cap.release()
cv2.destroyAllWindows()
