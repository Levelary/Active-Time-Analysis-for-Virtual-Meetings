#!/usr/bin/env python
# coding: utf-8

# In[13]:
# pip install opencv-python-headless

# In[2]:
# pip uninstall tensorflow keras -y
# pip install tensorflow
# pip install keras==2.3.1  --> in terminal

import cv2     # for capturing videos
import os
import math   
import matplotlib.pyplot as plt    # for plotting the images
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    
# from keras.utils import np_utils
# from tensorflow.keras.utipyls import to_categorical
from tensorflow.keras.utils import to_categorical

from skimage.transform import resize   # for resizing images


def display_images(images, title, cols=5):
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    fig.suptitle(title, fontsize=15)
    axes = axes.flatten()
    for i, img in enumerate(images[:cols * rows]):
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if os.listdir(output_dir):
        print("The program already holds frames. Skipping frame extraction.")
        return

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    

    frame_number = 0
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Define the output filename
        frame_filename = os.path.join(output_dir, f"frame_{frame_number:}.png")

        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")

        # Increment the frame number
        frame_number += 1

    # Release the video capture object
    video_capture.release()
    print("Video processing complete.")
    
    
def extract_test_frames(videoFile, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.listdir(output_dir):
        print("The program already holds frames. Skipping testing-frames extraction.")
        return
    
    count = 0
    cap = cv2.VideoCapture(videoFile)
    
    try:
        # Validate if video can be opened
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {videoFile}")
            return
    
        # Get the frame rate (FPS) of the video
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        if frameRate == 0:
            print("Warning: Frame rate is zero. Assuming 30 FPS as default.")
            frameRate = 30  # Default to 30 FPS if frame rate is unavailable
            count = 0
            
        while cap.isOpened():
            frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get current frame ID
            ret, frame = cap.read()  # Read the frame
            
            if not ret:  # Stop if no more frames are available
                break
            
            # Save one frame per second
            if frameId % math.ceil(frameRate) == 0:
                filename = os.path.join(output_dir, f"test{count}.jpg")
                cv2.imwrite(filename, frame)
                count += 1
        
        print(f"Extraction complete. {count} frames saved to {output_dir}.")
    finally:
        cap.release()  # Ensure resources are released
    print("Done")
    
    
def getAnalysis():
    # Load data
    data = pd.read_csv('mapping.csv')
    # X = [plt.imread('frames2//' + img_name) for img_name in data.Image_ID]
    X = [plt.imread(os.path.join("frames2", img)) for img in data.Image_ID]

    X = np.array(X)

    # Display sample frames
    display_images(X[:10], "Samplae Extracted Frames")
    # pip uninstall opencv-python opencv-contrib-python
    # pip install opencv-python-headless opencv-contrib-python

    base_dir = os.path.expanduser("C:/Users/bhara/OneDrive/Desktop/projects/final_mini_project")
    video_path = os.path.join(base_dir, "video.mp4")
    output_dir = os.path.join(base_dir, "frames2")

    # video_path = 'C:\\Users\\bhara\\OneDrive\\Desktop\\projects\\final_mini_project\\video.mp4'.encode('utf-8').decode('utf-8')
    # output_dir = 'C:\\Users\\bhara\\OneDrive\\Desktop\\projects\\final_mini_project\\frames2'
    extract_frames(video_path, output_dir)


    # In[31]:


    #step to read the images
    X = [ ]     # creating an empty array
    for img_name in data.Image_ID:
        # img = plt.imread('frames//' + img_name)
        img = plt.imread(os.path.join('frames2', img_name))
        X.append(img)  # storing each image in array X
    X = np.array(X)    # converting list to array


    # In[32]:


    # There are three classes, we will one hot encode them 
    # using the to_categorical() function of keras.utils.
    y = data.Class
    # dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes
    dummy_y = to_categorical(y)


    # In[33]:


    # We are using a VGG16 pretrained model which takes an input image of shape (224 X 224 X 3). 
    # Since our images are in a different size, we need to reshape all of them. 
    # We will use the resize() function of skimage.transform to do this.

    image = []
    for i in range(0,X.shape[0]):
        a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
        image.append(a)
    X = np.array(image)


    # In[34]:


    # Preprocess it as per the model’s requirement to improve model's performence.
    # Use the preprocess_input() function of keras.applications.vgg16 to perform this step.

    from tensorflow.keras.applications.vgg16 import preprocess_input
    # X = preprocess_input(X, mode='tf')
    X = preprocess_input(X)# preprocessing the input data


    # In[35]:


    # We also need a validation set to check the performance of the model on unseen images.
    # We will make use of the train_test_split() function of the sklearn.model_selection module 
    # to randomly divide images into training and validation set.

    from sklearn.model_selection import train_test_split
    # preparing the validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)


    # In[36]:


    # Now we build our model.
    # VGG16 pretrained model will be used for this task.
    from keras.models import Sequential
    from keras.applications.vgg16 import VGG16
    from keras.layers import Dense, InputLayer, Dropout


    # In[37]:


    # load the VGG16 pretrained model and store it as base_model:

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    
    # include_top=False to remove the top layer


    # In[38]:


    # We will make predictions using this model for X_train and X_valid, 
    # get the features, and then use those features to retrain the model.

    X_train = base_model.predict(X_train)
    X_valid = base_model.predict(X_valid)
    X_train.shape, X_valid.shape


    # In[39]:


    # The shape of X_train and X_valid is (208, 7, 7, 512), (90, 7, 7, 512) respectively. 
    # In order to pass it to our neural network, we have to reshape it to 1-D.

    X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
    X_valid = X_valid.reshape(90, 7*7*512)


    # In[40]:


    # We will now preprocess the images and make them zero-centered 
    # which helps the model to converge faster.

    train = X_train/X_train.max()      # centering the data
    X_valid = X_valid/X_train.max()


    # In[41]:


    # Finally, we will build our model. This step can be divided into 3 sub-steps:

    # 1. Building the model
    # 2. Compiling the model
    # 3. Training the model


    # In[42]:


    # i. Building the model
    model = Sequential()
    model.add(InputLayer((7*7*512,)))    # input layer
    model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
    model.add(Dense(4, activation='softmax'))    # output layer - set t0 4 instead of 3


    # In[144]:


    # summary of the model
    model.summary()


    # In[43]:


    # We have a hidden layer with 1,024 neurons and an output layer with 3 neurons 
    # (since we have 3 classes to predict). Now the model can be compiled.

    # ii. Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # In[44]:


    # In the final step, we will fit the model and simultaneously also check its 
    # performance on the unseen images, i.e., validation images:

    # iii. Training the model
    model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

    # an accuracy of around 85% on unseen images
    
    # videoFile = "video.mp4"
    video_path = 'C:\\Users\\bhara\\OneDrive\\Desktop\\projects\\final_mini_project\\video.mp4'.encode('utf-8').decode('utf-8')
    output_dir = 'C:\\Users\\bhara\\OneDrive\\Desktop\\projects\\final_mini_project\\output_frames2'
    extract_test_frames(video_path, output_dir)
    # In[45]:


    # load the list of frames
    test = pd.read_csv('test.csv')


    # In[47]:


    #import the images and reshape for testing
    test_image = []
    for img_name in test.Image_ID:
        img = plt.imread('output_frames2//' + img_name)
        test_image.append(img)
    test_img = np.array(test_image)

    test_image = []
    for i in range(test_img.shape[0]):
        a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
        test_image.append(a)
    test_image = np.array(test_image)


    # In[48]:


    # preprocessing the images
    import tensorflow as tf
    test_image = preprocess_input(test_image)#, mode='tf')#, dtype=tf.float32)#, 

    # extracting features from the image
    test_image = base_model.predict(test_image)
    print(test_image.shape)
    # # converting the images to 1-D form
    # test_image = test_image.reshape(36, 7*7*512) # (186, 7*7*512)

    # # zero centered images
    # test_image = test_image/test_image.max()


    # In[58]:


    # predictions = model.predict_classes(test_image)

    # Alternate approach:
    # Preprocess input
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    # Assuming test_image is a batch of images
    test_image = preprocess_input(test_image)

    # If model expects flattened input
    test_image = test_image.reshape((test_image.shape[0], -1))  # Reshape as per model requirements

    # Predict probabilities
    predictions = model.predict(test_image)

    # Convert probabilities to class indices
    class_predictions = np.argmax(predictions, axis=1)

    print(class_predictions)


    # In[59]:


    print("Screen times of participants: ")
    # for i in range(len(predictions)):
        
    #     print(f"Checking for i={i}")
    #     # print(class_predictions[class_predictions == i])  # See if this returns non-empty
    print("Participant no. 1: ", class_predictions[class_predictions == 1].shape[0])  # Verify the count
    print("Participant no. 2: ", class_predictions[class_predictions == 2].shape[0])  # Verify the count
    print("Participant no. 3: ", class_predictions[class_predictions == 3].shape[0])  # Verify the count

    participant_counts = [np.sum(class_predictions == i) for i in range(1, 4)]
    plt.figure(figsize=(8,6))
    import seaborn as sns
    sns.barplot(x=["Participant 1", "Participant 2", "Participant 3"], y=participant_counts, palette="viridis")
    plt.xlabel("Participants")
    plt.ylabel("Screen Time (frames)")
    plt.title("Participation Time Analysis")
    plt.show()


    return participant_counts


if __name__ == "__main__":
    print(getAnalysis())