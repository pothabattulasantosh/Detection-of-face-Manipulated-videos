{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 97
    },
    "colab_type": "code",
    "id": "GN-7QbU0Bgbh",
    "outputId": "041a7d60-2850-4be4-a5e6-77bbaa1d7696"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import dlib\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "face_detector = dlib.get_frontal_face_detector()\n",
    "from keras.applications.xception import preprocess_input\n",
    "from keras import models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 122
    },
    "colab_type": "code",
    "id": "pKg415IMBoGv",
    "outputId": "40eabbef-a7fc-4c0a-bf94-960ef4dc5023"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code\n",
      "\n",
      "Enter your authorization code:\n",
      "··········\n",
      "Mounted at /content/drive\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "INTXw88HBgbw"
   },
   "outputs": [],
   "source": [
    "model_Xc = models.load_model('/content/drive/My Drive/Submit/model_finetuned_xception.hdf5') # \n",
    "\n",
    "\n",
    "def get_boundingbox(face, width, height, scale=1.3, minsize=None):\n",
    "    #Reference: https://github.com/ondyari/FaceForensics\n",
    "    \"\"\"\n",
    "    Expects a dlib face to generate a quadratic bounding box.\n",
    "    :param face: dlib face class\n",
    "    :param width: frame width\n",
    "    :param height: frame height\n",
    "    :param scale: bounding box size multiplier to get a bigger face region\n",
    "    :param minsize: set minimum bounding box size\n",
    "    :return: x, y, bounding_box_size in opencv form\n",
    "    \"\"\"\n",
    "    x1 = face.left() # Taking lines numbers around face\n",
    "    y1 = face.top()\n",
    "    x2 = face.right()\n",
    "    y2 = face.bottom()\n",
    "    size_bb = int(max(x2 - x1, y2 - y1) * scale) # scaling size of box to 1.3\n",
    "    if minsize:\n",
    "        if size_bb < minsize:\n",
    "            size_bb = minsize\n",
    "    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2\n",
    "\n",
    "    # Check for out of bounds, x-y top left corner\n",
    "    x1 = max(int(center_x - size_bb // 2), 0)\n",
    "    y1 = max(int(center_y - size_bb // 2), 0)\n",
    "    # Check for too big bb size for given x, y\n",
    "    size_bb = min(width - x1, size_bb)\n",
    "    size_bb = min(height - y1, size_bb)\n",
    "\n",
    "    return x1, y1, size_bb\n",
    "\n",
    "def get_predicition(image):\n",
    "    \"\"\"Expects the image input, this image further cropped to face\n",
    "    and the cropped face image will be sent for evalution funtion \n",
    "    finally \n",
    "    returns the annotated reusult with bounding box around the face. \n",
    "    \"\"\"\n",
    "    height, width = image.shape[:2]\n",
    "    try: # If in case face is not detected at any frame\n",
    "        face = face_detector(image, 1)[0]  # Face detection\n",
    "        x, y, size = get_boundingbox(face=face, width=width, height=height) # Calling to get bound box around the face\n",
    "    except IndexError:\n",
    "        pass\n",
    "    cropped_face = image[y:y+size, x:x+size] # cropping the face \n",
    "    output,label = evaluate(cropped_face) # Sending the cropped face to get classifier result \n",
    "    font_face = cv2.FONT_HERSHEY_SIMPLEX # font settings\n",
    "    thickness = 2\n",
    "    font_scale = 1\n",
    "    if label=='Real':\n",
    "        color = (0,255, 0)\n",
    "    else:\n",
    "        color = (0, 0, 255)\n",
    "    x = face.left()    # Setting the bounding box on uncropped image\n",
    "    y = face.top()\n",
    "    w = face.right() - x\n",
    "    h = face.bottom() - y\n",
    "    cv2.putText(image, label+'_'+str('%.2f'%output)+'%', (x, y+h+30), \n",
    "            font_face, font_scale,\n",
    "            color, thickness, 2) # Putting the label and confidence values\n",
    "\n",
    "    return cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)# draw box over face\n",
    "\n",
    "def evaluate(cropped_face):\n",
    "    \"\"\"This function classifies the cropped  face on loading the trained model\n",
    "    and \n",
    "    returns the label and confidence value\n",
    "    \"\"\"        \n",
    "    img = cv2.resize(cropped_face, (299, 299))\n",
    "    img = np.expand_dims(img, axis=0)\n",
    "    img = preprocess_input(img) \n",
    "    res = model_Xc.predict(img)[0]\n",
    "    if np.argmax(res)==1:\n",
    "        label = 'Fake'\n",
    "    else:\n",
    "        label = 'Real'\n",
    "    return res[np.argmax(res)]*100.0, label\n",
    "\n",
    "\n",
    "def final_model(video_path,limit_frames):\n",
    "        \"\"\"Expects the video path : '/xxx.mp4'\n",
    "        limit_frames : total number frames to be taken from input video \n",
    "         function will write the video with \n",
    "        classification results and place the output video in the pwd\"\"\"\n",
    "        output_ = video_path.split(\"/\")[-1].split(\".\")[-2]\n",
    "        capture = cv2.VideoCapture(video_path)\n",
    "        if capture.isOpened():\n",
    "                _,image = capture.read()\n",
    "                frame_width = int(capture.get(3))\n",
    "                frame_height = int(capture.get(4))\n",
    "                out = cv2.VideoWriter(output_+'_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))\n",
    "        else:\n",
    "                _ = False\n",
    "        i=1   \n",
    "        while (_):\n",
    "                _, image = capture.read()\n",
    "                classified_img = get_predicition(image)\n",
    "                out.write(classified_img)\n",
    "                if i%10 == 0:\n",
    "                    print(\"Number of frames complted:{}\".format(i))\n",
    "                if i==limit_frames:\n",
    "                        break\n",
    "                i=i+1\n",
    "        capture.release()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 357
    },
    "colab_type": "code",
    "id": "EjR3JsUIFM5c",
    "outputId": "2097bd3d-b7ae-4c02-f5e9-9c00f2a5be2f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of frames complted:10\n",
      "Number of frames complted:20\n",
      "Number of frames complted:30\n",
      "Number of frames complted:40\n",
      "Number of frames complted:50\n",
      "Number of frames complted:60\n",
      "Number of frames complted:70\n",
      "Number of frames complted:80\n",
      "Number of frames complted:90\n",
      "Number of frames complted:100\n",
      "Number of frames complted:110\n",
      "Number of frames complted:120\n",
      "Number of frames complted:130\n",
      "Number of frames complted:140\n",
      "Number of frames complted:150\n",
      "Number of frames complted:160\n",
      "Number of frames complted:170\n",
      "Number of frames complted:180\n",
      "Number of frames complted:190\n",
      "Number of frames complted:200\n"
     ]
    }
   ],
   "source": [
    "#For testing any new video\n",
    "final_model(video_path='/content/drive/My Drive/FaceForensics++/notebooks/878_866.mp4',\n",
    "            limit_frames=200)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "final.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
