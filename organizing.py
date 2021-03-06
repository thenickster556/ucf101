import os
from random import shuffle
import glob
import numpy as np
import cv2 
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas
import pickle
directory = 'C:/Users/nick/OneDrive/robo sys/final project/UCF101mini'
# directory = 'C:/Users/campb/OneDrive/robot vision/PA.3/ucf_sports_actions/ucf action'
# Get the labels from the directory 
# labels = os.listdir(directory)
TRAIN_TEST_SPLIT = 0.10#percentage that is test set
labels = [x[1] for x in os.walk(directory)][0]   # ['Cat', 'Dog'] 
NUM_LABELS = len(labels)
# Sort the labels to be consistent
# build dictionary for indexes
label_indexes = {labels[i]: i for i in range(0, len(labels))}  
sorted(labels)
# print(label_indexes)
# get the file paths
data_files= []#to store paths and id
i =-1#label id
for file in os.listdir(directory):#grabbing the frames and the id
    i+=1#the id everytime change title folder
    filename = os.fsdecode(directory+ '/' + file)
    for file1 in os.listdir(filename):
        if file1.endswith('.avi'):
            # Playing video from file:
            cap = cv2.VideoCapture(filename+'/' +file1)

            try:
                if not os.path.exists('C:/Users/nick/OneDrive/robo sys/final project/dataMini'):
                    os.makedirs('C:/Users/nick/OneDrive/robo sys/final project/dataMini')
            except OSError:
                print ('Error: Creating directory of data')

            currentFrame = 0#every new id start at 0
            
            while(True):
            # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == False:
                    break
                # Saves image of the current frame in jpg file
                name = 'C:/Users/nick/OneDrive/robo sys/final project/dataMini/'+ file + str(currentFrame) + '.jpg'
                cv2.imwrite(name, frame)
                data_files.append((i,name))
                # To stop duplicate images
                currentFrame += 1
            # When everything done, release the capture
            cap.release()







            #for future improvement
            # shuffle(data_files)
            # idx =0
            # for labels in data_files
            #     if(idx < len(data_files)*(1-TRAIN_TEST_SPLIT+Val_TEST_SPLIT))
            #         f = open('C:/Users/nick/OneDrive/robot vision/PA.3/trainPCn.pkl','a')
            # data_files.append((i,filename1+ '/' +file2))
                
         
# shuffle the data 
shuffle(data_files)

num_data_files = len(data_files)

TRAIN_TEST_SPLIT = 0.20#percentage that is test set
#getting total for each
nr_test_data = int(num_data_files * TRAIN_TEST_SPLIT)

nr_train_data= int(num_data_files-nr_test_data)

train_data_files = np.array(data_files[:nr_train_data])

test_data_files = np.array(data_files[nr_train_data:])

with open("trainMini.pkl","wb") as f:#setting up the train dataset with label and urls
    pickle.dump(train_data_files,f)
with open("testMini.pkl","wb") as f1:#setting up the test dataset with label and urls
    pickle.dump(test_data_files,f1)
