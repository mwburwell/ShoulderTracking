import cv2.data
import numpy as np
import cv2

import math
import sys

from SlidingWindow import SlidingWindow
from line import Line

import matplotlib.pyplot as plt 



def main():
    video = cv2.VideoCapture('./shoulder-proj.mp4')
    last_box = np.array((1, 4))
    
    sentinel = False
    l_last_line = np.zeros((2))
    r_last_line = np.zeros((2))
    try:
        while(video.isOpened()):
        # capture frame by frame
            result, frame = video.read()
            if result is False:
                break
            
            
            faces, last_box = detect_bounding_box(frame, last_box)
            if len(faces) > 0:
                cv2.drawMarker(frame, (last_box[0], last_box[1]), markerSize=10, markerType=cv2.MARKER_STAR, color=(0,0,255))
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blured_img = cv2.GaussianBlur(gray_img, (7, 7), 1)
                minval = np.percentile(blured_img, 11)
                maxval = np.percentile(blured_img, 100)

                img_contrast = np.clip(blured_img.copy(), minval, maxval)
                img_contrast = ((img_contrast - minval) / (maxval - minval)) * 255
                x, y, w, h = last_box
                
                # left and right should windows
                displacment = 0
                # print(x - w + displacment)
                # print(x)
                ls_window_img = img_contrast[y: (y + h + w), (x - w + displacment) : x]
                rs_window_img = img_contrast[y: (y + h + w), (x + w): (x + (2 * w) - displacment)]
                
                height = ls_window_img.shape[0]
                length = ls_window_img.shape[1]
                
                sig = round(height / 1.85)
                med = round(height / 1.85)
                left_shoulder_window = SlidingWindow(ls_window_img, 64, median= med, sigma= sig, constant=10)
                right_shoulder_window = SlidingWindow(rs_window_img, 64, median= med, sigma= sig, constant=10)
                
                left_line_vector = np.polyfit(left_shoulder_window.x_positions[:,0], left_shoulder_window.y_positions[:,0], 1)
                right_line_vector = np.polyfit(right_shoulder_window.x_positions[:,0], right_shoulder_window.y_positions[:,0], 1)
                
                if l_last_line[0] == 0:
                    l_last_line = left_line_vector
                if r_last_line[0] == 0:
                    r_last_line = right_line_vector
                
                left_line_vector = jitter_defender(left_line_vector, l_last_line, 5)
                right_line_vector = jitter_defender(right_line_vector, r_last_line, 5)
                
                
                l_start_y = round(left_line_vector[1] + h)
                l_end_y = round(length * left_line_vector[0] + left_line_vector[1] + h)
                cv2.line(frame, (0 + x - w, l_start_y), (length + x - w, l_end_y), (0,0,255), 3)
                
                r_start_y = round(right_line_vector[1] + h)
                r_end_y = round(length * right_line_vector[0] + right_line_vector[1] + h)
                cv2.line(frame, (0 + x + w, r_start_y), (length + x + w, r_end_y), (0,0,255), 3)
                
                if sentinel and len(faces) > 1:
                    plt.title("Top & Bottom Segmentation")
                    plt.imshow(ls_window_img, cmap="gray")
                    plt.scatter(left_shoulder_window.x_positions[:, 0], left_shoulder_window.y_positions[:, 0])
                    # plt.plot(up_low_seg_line.x_values, up_low_seg_line.y_values, color="red")
                    # plt.plot(firstWindow.x_positions, np.polyval(line_vector, firstWindow.x_positions), color="red")
                    plt.show()
                    left_shoulder_window.sampling(30, 'Test')
                    sentinel = False

                l_last_line = left_line_vector
                r_last_line = right_line_vector
            cv2.imshow('frame', frame)
            # cv2.imshow('frame', integral_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        cv2.destroyAllWindows()
        
    # video.release()
    # cv2.destroyAllWindows()
    
    
    
    
def detect_bounding_box(img, last_bounding_box):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
    
    if len(faces) > 0:
        indx = 0
        if len(faces) > 1:
            indx = find_face_among_list(faces, last_bounding_box)
            x, y, w, h = faces[indx]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            
        else:
            x, y, w, h = faces[indx]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
        return faces, faces[indx]
    else:
        return faces, np.empty((1, 4))
        
    
        
    # for (x, y, w, h) in faces:
    #     if len(faces) > 1:
            
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)


def euclidean_distance(x, y, last_x, last_y):
    return math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

def find_face_among_list(faces, last_bounding_box):
    min_dist = sys.maxsize
    indx = 0
        
    for i, (x, y, w, h) in enumerate(faces):
        last_x, last_y = last_bounding_box[0:2]
        
        temp_dist = euclidean_distance(x, y, last_x, last_y)
        if temp_dist < min_dist:
            min_dist = temp_dist
            indx = i
    
    return indx

def jitter_defender(new_line, old_line, threshold):
    theta = math.tan(new_line[0] / old_line[0]) #(abs(new_line[0]) + abs(old_line[0])) / 2
    if abs(theta) > threshold:
        # print(f'return old line: {old_line} - {new_line} - {theta}')
        return old_line
    
    # print(f'return new line: {old_line} - {new_line} - {theta}')
    return new_line
        
        
    
        
    


    
    
main()
