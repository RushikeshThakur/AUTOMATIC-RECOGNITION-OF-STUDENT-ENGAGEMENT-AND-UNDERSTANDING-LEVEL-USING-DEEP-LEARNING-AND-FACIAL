import numpy as np
import cv2
import sys
import os
import subprocess
import csv
import pandas as pd
from csv import reader
from csv import DictReader
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
sys.modules['Image'] = Image
from flask import Flask, request, jsonify, render_template
from flask_bootstrap  import Bootstrap
import pickle
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Importing packages completed
# Actual working of module begin .....




print('Module start working fine ....')
app = Flask(__name__)

Bootstrap(app)          # Adding bootstrap to Flask

frames = os.listdir('Predictions/ExtractedFrames')     # Creating the folder for extracted images

videopath= r'E:\\BE\\BE PROJECT\\GUI with ML Project\\Student_Engagement_jupyter\\Student_Enagagement_jupyter\\static\\video0.mp4'
# Video path for engagement

print('Spliting video begin')
def split_video(video_file,despath):                     # method responsible for spliting the video into slices
    return subprocess.check_output('ffmpeg -i "' +video_file + '" -r 0.25 ' + 'Frame_%03d.jpg', shell=True, cwd=despath)

print('Extrating frames begin')
def extractframes(videopath):                             # method responsible Extract the frames and store in ExtractedFremes folder
    dpath="Predictions/ExtractedFrames/"
    split_video(videopath,dpath)                          # calling spilt method for videopath and folder path

print('Extrated frames successfully ....')
extractframes(videopath)    # extrated the frames from video

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Object detection using CasCadeClassifier
padding=2700                                                                                          #padding means length of object detection
print('Object detection successfully ....')

print("Crop images begin")
def crop(imgname,imgpath,despath):
    print(imgname)
    image = cv2.imread(imgpath)                             # Reading the images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # converting from RGB to GRAY scale images for imagesprocessing
    faces = faceCascade.detectMultiScale(gray, scaleFactor=2.7, minNeighbors=3, minSize=(30, 30))   # Defing the area of object detection
    i=0
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x-padding,y-padding),(x+w+padding,y+h+padding), (0, 255, 0), 2)   # Draw rectangle on the each detected faces
        roi_color = image[y:y + h, x:x + w]
        i=i+1
        cv2.imwrite(despath+'/'+'Face_'+str(i)+'.jpg', roi_color)                           # Assign the name for each frames
    print("{} Objects found".format(i))                                                     # Printing the frame0,frame1,frame2 and so on ......
print('Croping frames successfully done .....')

print("Operation on each frame begin")
for f in frames:
    filename = os.path.basename(f)                                         # take the tail of root path
    (ImageName, ext) = os.path.splitext(filename)
    desdir='Predictions/'+ImageName                                        # store the image in prediction folder
    os.makedirs(desdir)                                                    # make directory and store crop image
    crop(ImageName,'Predictions/ExtractedFrames/'+f,desdir)
    print(desdir)
print('Crop frames are successfully store on respective folder ....')

directories=os.listdir('Predictions/')                                     # Reading the frames from Predictions folder
print(directories[1:])                                                     # printing the name of all frames
print('Directory operation completed successfully ....')

for i in directories[1:]:
    croppedfaces=os.listdir('Predictions/'+i+'/')                         # Assign the croped faces to a variable
    data={'Frames':croppedfaces}                                          # Assing value to frames
    df = pd.DataFrame(data)                                               # before writing to data to csv file converted to dataframe
    df['Boredom']=0                                                       # Initialing the attribute to zero
    df['Confusion']=0                                                     # Initialing the attribute to zero
    df['Engagement']=0                                                    # Initialing the attribute to zero
    df['Frustration']=0                                                   # Initialing the attribute to zero
    print(df)
    df.to_csv('Predict.csv', mode='a' ,index = False, header=False)       # writing the  engagement to the csv file
print('End directory forloop')

  predcsv=pd.read_csv('Predict.csv', header=None, delim_whitespace=True)              # Reading the dataframes from csv file
  columns=["Boredom", "Engagement", "Confusion", "Frustration"]                       # Creating the columns
  pred_datagen=ImageDataGenerator(rescale=1./255.)
  pred_generator=pred_datagen.flow_from_dataframe(
  dataframe=predcsv,
  directory='Predictions/Frame_1',
  x_col="Frames",
  batch_size=1,
  seed=42,
  shuffle=False,
  class_mode=None,
  target_size=(175,175))
  print('End reading the csv file')

 load json and create model
 json_file = open('model_FTT_6.json', 'r')
 loaded_model_json = json_file.read()
 json_file.close()
 loaded_model = model_from_json(loaded_model_json)
 # load weights into new model
 loaded_model.load_weights("model_FTT_6.h5")
 print("Loaded model from disk")

 STEP_SIZE_PRED=pred_generator.n//pred_generator.batch_size
#
 pred_generator.reset()
 pred=loaded_model.predict_generator(pred_generator,
 steps=STEP_SIZE_PRED,
 verbose=1)

pred_bool = (pred >0.8)

 predictions = pred_bool.astype(int)
 columns=["Boredom", "Engagement", "Confusion", "Frustration"]
#
 results=pd.DataFrame(predictions, columns=columns)
 results["Frames"]=pred_generator.filenames
 ordered_cols=["Frames"]+columns
 results=results[ordered_cols]#To get the same column order
 results.to_csv("Predict.csv",index=False)

 def decision(b,e,c,f):
     global Deng,Engaged
     if(e==1 and b,e,f==0):
         Engaged=Engaged+1
     elif(c,e==1 and f,b==0):
         Engaged=Engaged+1
     else:
         Deng=Deng+1


 Deng=0
 Engaged=0
 count=0
#
# # iterate over each line as a ordered dictionary and print only few column by column name
 with open('Predict.csv', 'r') as read_obj:
     csv_dict_reader = DictReader(read_obj)
     for row in csv_dict_reader:
         count=count+1
         decision(row['Boredom'], row['Engagement'],row['Confusion'], row['Frustration'])

 print(Deng)
 print(Engaged)
#
 TDiseng=float (Deng/count)*100
 TEngaged=float (Engaged/count)*100
 print("%.5f" % round(TDiseng,5))
 print("%.5f" % round(TEngaged,5))

TDiseng = 20.00000           # Dummy value for estimating the prediction
TEngaged = 60.00000          # Dummy value for estimating the prediction
TNomeng = 10.00000           # Dummy value for estimating the prediction
TCD = 10.00000               # Dummy value for estimating the prediction

picpath = os.path.join("static","pic")
app.config["UPLOAD_FOLDER"] = picpath

# Flask API working begins

@app.route('/')              # Flask API for Home page
def index():
    return render_template("upload.html")

@app.route('/predict', methods=['GET','POST'])        # Flask API for Prediction
def predict():
    plt.figure(figsize=(5, 5))

    labels = ['Engaged', 'Nominally Engaged', 'Disengaged', 'Cannot Decide']

    colors = ["g", "c", "r", "y"]

    plt.title("Student Engagement Prediction")

    sizes = [TEngaged, TNomeng, TDiseng, TCD]

    engaged = sizes[0] + sizes[1]                    # Assing average to student engaged
    disengeged = sizes[2] + sizes[3]                 # Assing average to student disengaged

    print("Student Engaged", engaged)                 # Engaged Student Average
    print("Student Disengeged", disengeged)           # DisEngaged Student Average

    explode = [0, 0, 0, 0.09]                        # slice part higher

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, wedgeprops={"edgecolor":"black"} , explode=explode, colors=colors, radius=1.5)

    plt.show()
    #des='Graphs/'
    plt.savefig('E:\BE\BE PROJECT\GUI with ML Project\Engagement_Prediction_pycharm\Engagement_Prediction\static\pic\piaa.png', bbox_inches='tight', pad_inches=2, transparent=True)  # save picture
    student_engagement = engaged
    student_disengagement= disengeged
    pic1=os.path.join(app.config["UPLOAD_FOLDER"], 'piaaa.png')
    pic2 =os.path.join(app.config["UPLOAD_FOLDER"], 'his.png')
    # img = cv2.imread('piaa.png')
    # scale_percent=2.50
    # width=int(img.shape[1]*scale_percent)
    # height=int(img.shape[0]*scale_percent)
    # dimension=(width,height)
    # resized=cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
    # print(resized.shape)
    # cv2.imwrite('resize_pia.png',resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return render_template("result.html",eng=student_engagement,dis=student_disengagement,use_image1=pic1,use_image2=pic2)

if __name__ == '__main__':                       # Main Method
   app.run(debug=True)                           # Root point to Application run
