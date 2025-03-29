from tkinter import messagebox, filedialog, simpledialog
from tkinter import *
import numpy as np
import os
import cv2
import imutils
from yoloDetection import detectObject, displayImage
from train_model import train_yolo
from test_model import test_yolo
from train_model_adaboost import train_adaboost
from test_model_adaboost import test_adaboost
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


main = Tk()
main.title("Embedded Night-Vision System for Object Detection")
main.geometry("1300x1200")

global filename
graph_canvas = None
data=None

with open('model_metrics.json', 'r') as file:
    data = json.load(file) 

class_labels = open('yolov2model/yolov2-labels').read().strip().split('\n') 
cnn_model = cv2.dnn.readNetFromDarknet('yolov2model/yolov2.cfg', 'yolov2model/yolov2.weights') 
cnn_layer_names = cnn_model.getLayerNames()
cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()]

def detectFromImage(imagename):  
    label_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    try:
        image = cv2.imread(imagename)
        image_height, image_width = image.shape[:2]
    except:
        raise 'Invalid image path'
    finally:
        image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels)
    return image        

def uploadImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="testImages", filetypes=[("Image Files", "*.jpg;*.png")])
    pathlabel.config(text=filename+" image loaded")
    img = cv2.imread(filename)
    cv2.imshow('Original Uploaded Image', img)
    cv2.waitKey();    

def yoloDetect():
    global filename
    image = cv2.imread(filename)
    image = cv2.resize(image, (800, 600))
    detect = detectFromImage(filename)
    images = cv2.resize(detect, (800, 600))
    cv2.imshow("Original Image", image)
    cv2.imshow("Objects Detected Image with YOLOV2", images)    
    cv2.waitKey()

def extract_features(image):
    image = cv2.resize(image, (64, 128))  # Resize to match training size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


def plot_metrics():
    global graph_canvas  # Access global variable

    #  Destroy existing graph before showing new one
    if graph_canvas:
        graph_canvas.get_tk_widget().destroy()
    
    accuracy = data["accuracy"]
    precision = data["presecion"] 
    recall = data["recall"]
    f1_score = data["f1_score"]

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1_score]

    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    ax.set_ylim(0, 100)  # Set Y-axis range from 0 to 1
    ax.set_ylabel("Score")
    ax.set_title("Yolov2 Performance Metrics")

    # Embed Matplotlib Graph into Tkinter
    graph_canvas = FigureCanvasTkAgg(fig, master=main)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().place(x=400, y=100)

def plot_metrics_adaboost():
    global graph_canvas  # Access global variable

    # Destroy existing graph before showing new one
    if graph_canvas:
        graph_canvas.get_tk_widget().destroy()
    
    accuracy = data["accu_ada"]
    precision = data["pre_ada"]
    recall = data["recall_ada"]
    f1_score = data["f1_ada"]

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1_score]

    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(metrics, values, color=['cyan', 'orange', 'pink', 'brown'])
    ax.set_ylim(0, 100)  # Set Y-axis range from 0 to 1
    ax.set_ylabel("Score")
    ax.set_title("Haar + AdaBoost Performance Metrics")

    # Embed Matplotlib Graph into Tkinter
    graph_canvas = FigureCanvasTkAgg(fig, master=main)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().place(x=400, y=100)  # Adjust position as needed

# Function to detect pedestrians in an image
def detect_pedestrians():
    model = joblib.load('adaboost_pedestrian.pkl')
    image = cv2.imread(filename)
    if image is None:
        print(f"Error loading image: {filename}")
        return

    height, width = image.shape[:2]
    step_size = 8  # Sliding window step
    window_size = (64, 128)  # Same as training size

    for y in range(0, height - window_size[1], step_size):
        for x in range(0, width - window_size[0], step_size):
            window = image[y:y+window_size[1], x:x+window_size[0]]
            features = extract_features(window)
            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features)
            if prediction == 1:  # If pedestrian detected
                cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    # Show and save the result
    cv2.imshow("Object Detection", image)
    cv2.imwrite("output_detected.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adBoost(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def haarDetect(image,img1):
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    image = imutils.resize(image, width=min(800, 800))
    img1 = imutils.resize(img1, width=min(800, 800))
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)    
    for (x, y, w, h) in regions:
        if (y+h) > 350:
            cv2.rectangle(img1, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
            print(str(y)+" "+str(h)+" "+str((y+h)))
    return img1  

def adaboostDetect():
    global filename
    image = cv2.imread(filename)
    adjusted = adBoost(image, gamma=3.5)
    cv2.imwrite("temp.png",adjusted)
    img= cv2.imread(filename)
    img1 = cv2.imread('temp.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray_img_eqhist = cv2.equalizeHist(gray_img)
    gray_img1_eqhist = cv2.equalizeHist(gray_img1)
    clahe = cv2.createCLAHE(clipLimit=20)
    gray_img_clahe = clahe.apply(gray_img_eqhist)
    th=80
    max_val=255
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    images = haarDetect(o3,img1)
    images = cv2.resize(images, (800, 600))
    image = cv2.resize(image, (800, 600))
    cv2.imshow("Original Image",image)
    cv2.imshow("Pedestrian Detected Image with HAAR + AdaBoost",images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trainModel():
    messagebox.showinfo("Training", "Training the YOLOv2 model on custom dataset. This will take some time...")
    train_yolo()
    messagebox.showinfo("Training Complete", "Model training completed successfully!")

def testModel():
    messagebox.showinfo("Testing", "Testing the trained model on test images...")
    test_yolo()
    messagebox.showinfo("Testing Complete", "Model testing completed. Check output images.")

def trainModelADABOOST():
    messagebox.showinfo("Training", "Training the ADABoost model on custom dataset. This will take some time...")
    train_adaboost()
    messagebox.showinfo("Training Complete", "Model training completed successfully!")

def testModelADABOOST():
    messagebox.showinfo("Testing", "Testing the trained model on test images...")
    test_adaboost()
    messagebox.showinfo("Testing Complete", "Model testing completed. Check output images.")

def exitApp():
    main.destroy()

# GUI Layout
font = ('times', 16, 'bold')
title = Label(main, text='Embedded Night-Vision System for Object Detection', bg='yellow4', fg='white', font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
Button(main, text="Upload Night Vision Image", command=uploadImage, font=font1).place(x=50, y=100)
pathlabel = Label(main, bg='yellow4', fg='white', font=font1)
pathlabel.place(x=50, y=150)

Button(main, text="Detect Object (YOLOv2)", command=yoloDetect, font=font1).place(x=50, y=200)
Button(main, text="Detect Object Haar+AdaBoost", command=adaboostDetect, font=font1).place(x=50, y=250)
#Button(main, text="Train Model for YOLO", command=trainModel, font=font1).place(x=50, y=300)
#Button(main, text="Test Model for YOLO", command=testModel, font=font1).place(x=50, y=350)
#Button(main, text="Train Model for ADABOOST", command=trainModelADABOOST, font=font1).place(x=50, y=350)
#Button(main, text="Test Model for ADABOOST", command=testModelADABOOST, font=font1).place(x=50, y=450)
Button(main, text="Show Performance Graph YOLOv2", command=plot_metrics, font=font1).place(x=50, y=300)
Button(main, text="Show Performance Graph Haar + ADABOOST", command=plot_metrics_adaboost, font=font1).place(x=50, y=350)
Button(main, text="Exit", command=exitApp, font=font1).place(x=50, y=400)


main.config(bg='magenta3')
main.mainloop()
