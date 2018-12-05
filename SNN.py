#!/usr/bin/env python
# coding: utf-8

# In[19]:

import os 
import cv2
import glob
import time
import shutil
from matplotlib import pyplot as plt
import pandas as pd
import requests
import json
import ast
import time
import statistics


# In[45]:


def extract_pose_from_video(input_loc, output_loc, dependencies_loc):
    
    protoFile = dependencies_loc+"pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = dependencies_loc+"pose_iter_160000.caffemodel"
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]
    
    if os.path.isfile(input_loc) == False:
        print (">>>>>>> No video exist. Please check.")
        return 1
    
    # Log the time
    #time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #print ("Number of frames: ", video_length)
    count = 0
    print (">>>>>>> Converting video frame to pose..\n")
    # Start converting the video
    while (cap.isOpened()):
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        #frame = cv2.imread("C:\\Users\\khaju\\Downloads\\Capstone\\Output\\Frames\\"+image)
        #frame = cv2.imread("multiple.jpg")
        #print("frame shape", frame.shape)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1
        #print("fram wd,ht", frameWidth, frameHeight)
        output_image = cv2.imread(dependencies_loc+"Output.jpg")
        output_image_resize = cv2.resize(output_image, (frameWidth, frameHeight))
        inWidth = 368
        inHeight = 368
        #cv2.imshow("Output-Keypoints", frame)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []

        for i in range(15):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        Head = points[0]
        #LeftAnkle = points[13]
        #Right_Sholder = points[2]
        #Left_Shoulder = points[5]
        Chest = points[14]
        #print("head and righsh", Head, Right_Sholder)
        #print("leftankle leftshould", LeftAnkle, Left_Shoulder, Chest)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 0, 0), 3)
                cv2.line(output_image_resize, points[partA], points[partB], (0, 0, 0), 12)
        #cv2.rectangle(output_image_resize, (Chest[0]-100, Chest[1]-60), (Chest[0]+100, Chest[1]+60), (0, 0, 255), 1)
        #cv2.imshow("Output-Keypoints", output_image_resize)
        #break
        #cv2.imwrite("C:\\Users\\khaju\\Downloads\\Capstone\\PoseExtracted\\"+image, output_image_resize)
        imgray = cv2.cvtColor(output_image_resize, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (21, 21), 0)
        thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)


        (_, contour, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        area_max = max([cv2.contourArea(x) for x in contour])
        for c in contour:
            if cv2.contourArea(c) < area_max:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(output_image_resize, (x - 16, y - 12), (x + w + 12, y + h + 12), (0, 255, 0), 1)
        #cv2.imshow("output-img", img)
        cv2.imwrite(output_loc+"%#05d.png" % (count+1), cv2.resize(output_image_resize[y:y+h, x:x+w], (25, 50)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            #time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            #print ("Done extracting frames.\n%d frames extracted" % count)
            #print ("It took %d seconds forconversion." % (time_end-time_start))
            break
           
    print (">>>>>>> Video conversion completed..\n")
    
    return 0


# In[52]:


def convert_extracted_pose_to_pixels(extracted_loc, mode, dependencies_loc):
    print (">>>>>>> Converting poses to Pixels..\n")
    images = []
    finallist = []
    for f in os.listdir(extracted_loc):
        if f.endswith('.png'):
            images.append(f)
    for image in images:
        img = cv2.imread(extracted_loc+image)
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY)
        newlist = []
        if mode == 'train':
            newlist.append(image[0:5])
        for eachpixellist in thresh:
            for eachpixel in eachpixellist:
                newlist.append(eachpixel)
        finallist.append(newlist)
    
    df = pd.DataFrame(finallist,  columns =  list(range(0,len(finallist[0]),1)))
    location_to_save_data = dependencies_loc+mode+"_posepixels.csv"
    df.to_csv(location_to_save_data, index=False,header=None)
    
    print (">>>>>>> Converted poses to pixels..\n")
    
    return location_to_save_data
    


# In[54]:


def train_snn(train_data_path, dependencies_loc):
    print (">>>>>>> Uploading data to server..\n")
    url = 'http://api.alpes.ai/snn/'
    TOKEN= 'Token 512r8ac837if43i47f743d739814ff97537dee6p'
    # Train upload
    api_method = 'dataupload'
    files = {'file': (open(train_data_path, 'rb').read())}
    params = {'file_type': 'csv'}
    headers = {'authorization': TOKEN}
    r = requests.post(url+api_method,files=files,headers=headers,params=params)
    train_file_upload = json.loads(r.text)
    print ("   >>>> Uploading data completed. File name: ", train_file_upload,"\n")
    #print ("Uploaded train file name:", train_file_upload)
    #train
    print (">>>>>>> Training started..\n")
    api_method='train'
    headers = {'authorization': TOKEN}
    params = {
                'uplodedfileid': train_file_upload['filename'],
                'no_of_initial_planes':3,
                'min_trainpoints_to_be_covered':0.98,
                'epoch':2
            }
    r = requests.post(url+api_method,params=params,headers=headers)
    training_task=json.loads(r.text)
    print ("   >>>> Training job submitted: ",training_task,"\n")
    time.sleep(30)
    url = 'http://api.alpes.ai/snn/job_status/'
    #format of the url should be api url + {job_id}
    url = url + training_task['job_id']
    job_info = requests.get(url)
    response=json.loads(job_info.text)
    train_model=ast.literal_eval(response['result'])['model']
    #train_model=ast.literal_eval(response['result'])['model']
    #print (response)
    path, filename = os.path.split(input_loc)
    folderlist = filename.split(".")
    try:
        os.mkdir(dependencies_loc+folderlist[0])
    except:
        #print("Warning::::",QueryImage_loc+"ExtractedQposes already exist")
        pass
    
    with open(dependencies_loc+folderlist[0]+"\\model.txt", "w") as text_file:
        print(f"{response['result'][90:103]}", file=text_file)
    print ("   >>>> Training completed status: ",response['status'],"\n")
    if response['status'] == 'SUCCESS':
        print ("   >>>> Model build is: ",response['result'][90:103],"\n") 

    


# In[40]:


def find_frame_and_play_video(Qlabel,input_loc):
    
    print (">>>>>>> Playing video for Query Image..")
    
    a = Qlabel
    #print(a)
    print("       >>>>> Video start from frame "+str(a)+" ..")
    video_path=input_loc
    cap = cv2.VideoCapture(video_path) #video_name is the video being called
    cap.set(1,a)
    while True:
        ret, frame = cap.read() # Read the frame
        if ret == True:
            cv2.imshow('Video started from frame number: '+str(a), frame)  
        else:
            cv2.destroyWindow('Video started from frame number: '+str(a))
            break
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow('Video started from frame number: '+str(a))
            break
        
    print("       >>>>> Video end for Query Image..\n")


# In[62]:


def pred_ts_for_query_img(QueryImage_loc, input_loc, dependencies_loc):
    
    print(">>>>>>> Started detecting poses from Query Images..\n")
    
    try:
        os.mkdir(QueryImage_loc+"ExtractedQposes\\")
    except:
        #print("Warning::::",QueryImage_loc+"ExtractedQposes already exist")
        shutil.rmtree(QueryImage_loc+"ExtractedQposes\\")
        os.mkdir(QueryImage_loc+"ExtractedQposes\\")
    
    protoFile = dependencies_loc+"pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = dependencies_loc+"pose_iter_160000.caffemodel"
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]
    
    if os.path.isdir(QueryImage_loc) == False:
        print (">>>>>>> Query image location is invalid. Please check.")
        return
    
    Qimages = []
    
    for f in os.listdir(QueryImage_loc):
        if f.endswith('.png'):
            Qimages.append(f)
            
    if len(Qimages) == 0:
        print (">>>>>>> No Query Image to process.")
        return
        
    for image in Qimages:
        frame = cv2.imread(QueryImage_loc+image)
        #frame = cv2.imread("multiple.jpg")
        #print("frame shape", frame.shape)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1
        #print("fram wd,ht", frameWidth, frameHeight)
        output_image = cv2.imread(dependencies_loc+"Output.jpg")
        output_image_resize = cv2.resize(output_image, (frameWidth, frameHeight))
        inWidth = 368
        inHeight = 368
        #cv2.imshow("Output-Keypoints", frame)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []

        for i in range(15):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        Head = points[0]
        #LeftAnkle = points[13]
        #Right_Sholder = points[2]
        #Left_Shoulder = points[5]
        Chest = points[14]
        #print("head and righsh", Head, Right_Sholder)
        #print("leftankle leftshould", LeftAnkle, Left_Shoulder, Chest)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 0, 0), 3)
                cv2.line(output_image_resize, points[partA], points[partB], (0, 0, 0), 12)
        #cv2.rectangle(output_image_resize, (Chest[0]-100, Chest[1]-60), (Chest[0]+100, Chest[1]+60), (0, 0, 255), 1)
        #cv2.imshow("Output-Keypoints", output_image_resize)
        #break
        #cv2.imwrite("C:\\Users\\khaju\\Downloads\\Capstone\\PoseExtracted\\"+image, output_image_resize)
        imgray = cv2.cvtColor(output_image_resize, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (21, 21), 0)
        thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)


        (_, contour, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        area_max = max([cv2.contourArea(x) for x in contour])
        for c in contour:
            if cv2.contourArea(c) < area_max:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(output_image_resize, (x - 16, y - 12), (x + w + 12, y + h + 12), (0, 255, 0), 1)
        #cv2.imshow("output-img", img)
        cv2.imwrite(QueryImage_loc+"ExtractedQposes\\"+image, cv2.resize(output_image_resize[y:y+h, x:x+w], (25, 50)))
    
    path, filename = os.path.split(input_loc)
    folderlist = filename.split(".")
    
    with open(dependencies_loc+folderlist[0]+'\\model.txt', 'r') as f:
        train_model = f.read().strip()
    
    print("Model used is ",train_model)
    #train_model = 'model_2544321'
    
    extracted_loc = QueryImage_loc+"ExtractedQposes\\"
    
    mode = 'predict'
    
    convert_extracted_pose_to_pixels(extracted_loc,mode,dependencies_loc)
    
    i = 0
    for rec in open(dependencies_loc+mode+"_posepixels.csv", "rb"):
        l = ''
        l = rec.decode("utf-8")
        #print(l)
        #train_model = 'model_2544321';
        api_method='predictclosest'
        url = 'http://api.alpes.ai/snn/'
        headers = {'authorization': "Token 512r8ac837if43i47f743d739814ff97537dee6p"}
        params = {'model':train_model,'kvalue': 5,'nbest': 0.1,'test_features':''}
        data={'test_features':l}
        r = requests.post(url+api_method,params=params,headers=headers,json=data)
        #print(json.loads(r.text))
        a = json.loads(r.text)
        #print(a)
        predict_list = []
        for i in range(5):
            try:
                predict_list.append(a['predected_label'][i])
            except:
                print("Model ",train_model," doesn't exist")
        #x = statistics.median(predict_list)[0]
        #print(predict_list)
        #x = min(predict_list)[0]
        #print(x)
        #find_frame_and_play_video(x)
        for x in a['predected_label'][0]:
            find_frame_and_play_video(x,input_loc)
        i = i + 1
    
    print (">>>>>>> All Query Images processed.") 
    
# In[75]:


if __name__ == '__main__':
    dependencies_loc = 'C:\\Users\\khaju\\Downloads\\Capstone\\Models\\'
    output_loc = 'C:\\Users\\khaju\\Downloads\\Capstone\\TestRun\\'
    input_loc = input("Please give the video location with video file name in mp4 format? \n")
    cond1 = input("You want to prepare new model for this video? y/n ")
    if cond1.upper() == 'Y':
        #input_loc = input("Please give the video location with video file name in mp4 format? \n")
        extracted_loc = output_loc
        mode = 'train'
        rc = extract_pose_from_video(input_loc,output_loc,dependencies_loc)
        if rc == 0:
            train_data_path = convert_extracted_pose_to_pixels(extracted_loc, mode, dependencies_loc)
            #print(train_data_path)
            #train_data_path = 'C:\\Users\\khaju\\Downloads\\Capstone\\Output\\FData.csv'
            train_snn(train_data_path, dependencies_loc)
    else:
        QueryImage_loc = input("Please give the Query images location? \n")
        pred_ts_for_query_img(QueryImage_loc, input_loc, dependencies_loc)




