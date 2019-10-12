
video_path="crowd_Video.mp4"
# Initializing important imports

import datetime

import numpy as np
import openpyxl
import xlrd
import xlsxwriter
from openpyxl import load_workbook
from xlutils.copy import copy
from xlwt import Workbook

import cv2
from pyimagesearch.centroidtracker import CentroidTracker
from summary import summary






# Initializing XLS file for data

Data_base = xlrd.open_workbook('Data_base.xls')
data = Data_base.sheet_by_index(0)
Feature = copy(Data_base)
sheetfeature = Feature.get_sheet(0)
row = int(data.cell_value(1,7))

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# Initialize Age & Gender model Parameters along with other variables

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
person = row
# For the ID number
ID = 0
# For the objects from centroid function
objects = []
# For detecting if the detection is mature
detect_mature = 0
# For the total number of detected people in each age group
T_age = [0, 0, 0, 0, 0, 0, 0, 0]
# For the total number of detected people of each gender
T_gender = [0,0]
# For total number of detected people
Total = 0
# For totla number of detections
detected = 0
# Initializing date
date = str(datetime.date.today().strftime("%B %d, %Y"))
# Deploying detection models

print("[Info]: Loading model . . .")
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
age_net = cv2.dnn.readNetFromCaffe(
    "deploy_age.prototxt",
    "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt",
    "gender_net.caffemodel")
print("[Info]: Model Deployed . . .")

# Starting Video Stream

print("[Info]: Starting Video Stream . . .")


video_frame = cv2.VideoCapture(video_path)

# loop over the frames of the video
while True:

    (grabbed, frame) = video_frame.read()
    if frame is None:
        break
    
    (h, w) = frame.shape[:2]
    
    # Resizing & blob formation to be sent into the model for prediction
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Initializing the list
    
    rects = []
    
    for i in range(0, detections.shape[2]):
	
    	# extract the confidence (i.e., probability) associated with the
    
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
    
        if confidence < 0.8:
            continue
	
    	# compute the (x, y)-coordinates of the bounding box for the
		# and append them to the list for assigning ObjectID 
    
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        rects.append(box.astype("int"))
        (startX, startY, endX, endY) = box.astype("int")
        
		# Copying the list in Ordered Dictionary so that it maintains the data
        
        objects = ct.update(rects)
        
        # For tracking the person

        for (objectID, centroid) in objects.items():
            
            # Assigning ID to the person
            
            text_ID = "ID {}".format(objectID)
            detected += 1
            # To take frame as soon as an object is detected I used a variable p

            if (detect_mature == 5):
                row = row + 1
                Total += 1
                
                # Cropping the face from the whole frame to save it.
                
                face_img = frame[(centroid[1]-30):(centroid[3]+30),(centroid[0]-20):(centroid[2]+20)].copy()
                
                # Check if the image has no attribute
                if (face_img.shape[0] == 0) or (face_img.shape[1] == 0):
                    print ("[Info]: The frame is missed")
                    detect_mature -= 1
                    continue
                # Saving the file as numpy array.
                print("[Info]: Image saved")
                encrypted = ('img/'+str(row)+'.npy')
                np.save(encrypted, face_img)
                cv2.imwrite('img/'+str(row)+'.jpg', face_img)

                # Resizing & blob formation to be sent into the model for prediction
                
                blob2 = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Now to predict gender
                
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                
                # For Summary file

                if (gender == 'Male'):
                    T_gender[0] += 1
                else:
                    T_gender[1] += 1

                # Now to Predict age
                
                age_net.setInput(blob2)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                T_age[age_preds[0].argmax()] += 1

                # To print in CMD about the gender & age.
                print("The ID: %s is %s & is between %s " % (person, gender, age))

                # Writing features to the excel file & saving it.

                sheetfeature.write(row, 1, datetime.date.today().strftime("%B %d, %Y"))
                sheetfeature.write(row, 2, datetime.datetime.now().strftime("%I:%M%p"))
                sheetfeature.write(row, 0, row)
                sheetfeature.write(row, 3, gender)
                sheetfeature.write(row, 4, age)
                sheetfeature.write(1, 7, row)
                Feature.save('Data_base.xls')

            # To make sure a new face is detected
            if(objectID > person):
                 person = person + 1
                 detect_mature = 0
            
            # To save memory we make sure that the gender & age detection model just work on one frame
            
            detect_mature += 1
            
            # Centroid for ID retaining of a detected face
            # Drawing the bounding box of the face along with the associated
            cv2.putText(frame, text_ID, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]),(0, 0, 255), 2)
            if (date != str(datetime.date.today().strftime("%B %d, %Y"))):
                summary(T_age, T_gender, Total, detected, date)
                print ("[Info]: Summary file saved . . .")
                date = str(datetime.date.today().strftime("%B %d, %Y"))
                T_age = [0, 0, 0, 0, 0, 0, 0, 0]
                T_gender = [0, 0]
                Total = 0

	# For streaming the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# If the `q` key was pressed the stream will end
    if key == ord("q"):
        break
    

# For cleaning up the space
summary(T_age, T_gender, Total, detected, date)
video_frame.release()
cv2.destroyAllWindows()
