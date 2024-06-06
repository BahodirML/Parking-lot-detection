import cv2
import numpy as np
from ultralytics import YOLO

def detect_parking_spaces(video_source, model_path, total_parking_spaces,  output_file):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Use YOLOv8 model to detect cars
        results = model(frame)
        boxes = results[0].boxes

        # Initialize parking space status
        occupied_spaces = 0

        # Check each parking space for occupancy
        for space in parking_spaces:
            x1, y1, x2, y2 = space
            space_occupied = False
            for box in boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
                # Check if the bounding box of the detected car overlaps with the parking space
                if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                    space_occupied = True
                    break
            if space_occupied:
                occupied_spaces += 1
                color = (0, 0, 255)  # Red for occupied spaces
            else:
                color = (0, 255, 0)  # Green for empty spaces
            # Draw the rectangle for the parking space
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Calculate empty spaces
        empty_spaces = total_parking_spaces - occupied_spaces

        # Print the counts of occupied and empty spaces
        print(f'Occupied spaces: {occupied_spaces}')
        print(f'Empty spaces: {empty_spaces}')

        # Annotate the frame with the counts
        cv2.putText(frame, f'Occupied: {occupied_spaces}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Empty: {empty_spaces}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Parking Space Monitor', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define the parking spaces as a list of (x1, y1, x2, y2) tuples based on the provided image
parking_spaces = [
    (15, 55, 90, 255),    # 0000
    (120, 45, 210, 265),   # 0001
    (235, 45, 325, 265),  # 0002
    (360, 55, 430, 255),  # 0003
    (475, 45, 555, 265),  # 0004
    (590, 55, 680, 255),  # 0005
    (705, 45, 795, 265),  # 0006
    (820, 45, 910, 265),  # 0007
    (935, 45, 1025, 265),  # 0008
    (1050, 55, 1140, 255),  # 0009
    (1165, 45, 1255, 265),  # 0010
    (1280, 45, 1370, 265),  # 0011
    (1395, 45, 1485, 265),  # 0012
    (1510, 45, 1600, 265),  # 0013
    
    (1890, 265, 1700, 360),  # 0038
    (1890, 375, 1700, 470), # 0039
    (1890, 485, 1700, 580),# 0040
    (1890, 595, 1720, 690),# 0041
    (1890, 705, 1700, 790),# 0042
    (1890, 805, 1700, 900),# 0043
    (1890, 915, 1700, 1005),# 0044
    (1890, 1020, 1700, 1110),# 0045

    (2, 590, 110, 835),# 0046
    (140, 590, 240, 835),# 0047
    (260, 590, 350, 835),# 0048
    (375, 590, 475, 835),# 0049
    (490, 590, 595, 835),# 0050
    (605, 590, 710, 835),# 0051
    (720, 590, 825, 835),  # 0060
    (840, 590, 950, 835), # 0061
    (955, 590, 1070, 835),# 0062
    (1075, 590, 1190, 835),# 0063

    (2, 840, 105,1080),# 0064
    (140, 840, 235, 1080),# 0065
    (250, 840, 350, 1080),# 0066
    (365, 840, 465, 1080),# 0067
    (475, 840, 595, 1080),# 0068
    (595, 840, 710, 1080),# 0069
    (710, 840, 825, 1080),# 0070
    (830, 840, 950, 1080),# 0071
    (955, 840, 1070, 1080),# 0072
    (1075, 840, 1190, 1080),# 0073
]

# Run the function with the video source (0 for webcam or file path for video file)
detect_parking_spaces('video.mp4', 'yolov8m.pt', 41, 'output_video.mp4')
