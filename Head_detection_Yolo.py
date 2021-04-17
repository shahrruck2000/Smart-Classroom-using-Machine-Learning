
import cv2
import numpy as np
import glob
import random
import imutils
from imutils.video import VideoStream
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
# BCM based numbering of the raspberrypi
# BCM - Broadcom SOC channel designation
GPIO.setwarnings(False)

GPIO.setup(17,GPIO.OUT)
GPIO.setup(27,GPIO.OUT)

# GPIO.setup(22,GPIO.OUT)
# GPIO.setup(5,GPIO.OUT)
# Load Yolo - You Only Look Once
# YOLO Algorithm uses CNN to detect objects

# readNet - It is a Opencv function which is used to load the deep neural network model in the memory or sytem
# model - 	Binary file contains trained weights , In case of Darknet framework the extension of model file is .weights
# config - Text file contains network configuration , In case of Darknet framework the extension of config file is .cfg
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["person"]

# Images path
# images_path = glob.glob(r"D:\YOLO image detection\custom data set classroom/*.jpg")



layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
# random.shuffle(images_path)
# loop through all the images
# vs = VideoStream(src="rtsp://admin:UKQNXJ@192.168.43.98/").start()
vs = VideoStream(0).start()
time.sleep(2.0)
while True:
	# Loading image
	frame = vs.read()

	l = int(frame.shape[0]) 

	b = int(frame.shape[1])
	
	# frames = [frame[0:int(l/2), 0:int(b/2), :], frame[0:int(l/2), int(b/2):int(b), :], frame[int(l/2):int(l), 0:int(b/2), :],frame[int(l/2):int(l), int(b/2):int(b), :]]
	frames = [frame[:, 0:int(b/2), :], frame[:, int(b/2):int(b), :]]

	# list_detect is a function which is used to store the detection result and is initiliased to zero 
	list_detect = [0]*2

	for j, frame in enumerate(frames):
# 		A lot of times when dealing with iterators, we also get a need to keep a count of iterations. Python eases the programmers’ task by providing a built-in function enumerate() for this task. 
# Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object
# enumerate(iterable, start=0)
# hence j in for loop gives index  for the frame
		img = cv2.resize(frame, None, fx=0.4, fy=0.4)
		# cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
		# fx is the scaling factor along X-axis or Horizontal axis
		# fy is the scaling factor along Y-axis or Vertical axis
		 # The fx and fy having 0.4 states that 40% should be the height and width of the image.
		height, width, channels = img.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		# Blob it’s used to extract feature from the image and to resize them.
		

		net.setInput(blob)
		# Now we are passing blob array to the neural network 
		outs = net.forward(output_layers)
		# .forward function will feedforward through neural network and we are feeding the output layers to
		# the function and the detection result will be stored in out variable



		class_ids = []
		# The class_ids will contain all the class (object) id's 
		# of the objects which will be detected. The class_ids will be decided as per maximum argument.
		confidences = []
		# The confidence value that YOLO assigns to an object
		# confidence List will store the score of the class_id. 
		boxes = []
		# Box List: contain the coordinates of the rectangle surrounding the object detected.

		# Here we are loopping through each detection in output layer to  get classid , confidence and bounding box corners
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					# If confidence>0.5 (50%), then the object will be detected
					# Object detected
					# print(class_id)
					list_detect[j] = 1

					center_x = int(detection[0] * width) 
					center_y = int(detection[1] * height)

					# W and H gives height and width of the object 
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates X and Y gives top left corner of the image
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)


					# Here only when the confidence is greater than 0.5 , 
					# bounding box coordinates , confidence and classid are
					# appended in the list
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		#Applying non-maxima suppression suppresses significantly overlapping bounding boxes(multiple boxes), 
		# keeping only the most confident ones.



		# print(indexes)
		font = cv2.FONT_HERSHEY_PLAIN
		# Here we are assigning the font styles
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				# From boxes, we extract the x,y,w,h coordinates of the object and label them with their class_ids.
				color = colors[class_ids[i]]
				# random colors generated earlier is assigned to color variable
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
				# (x,y) - Gives Top left corner of the bounding box
				# (x+w , y+h) - GIVES bottom right corner of the bounding box
				# 2 in the function the width of the rectangle
				# cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
				img = cv2.resize(img, (450,450))
				# Here we are resizing the image and displaying it using imshow
				cv2.imshow("Image number {}".format(j), img)
				cv2.waitKey(1)
				# waitKey(1) will display a frame for 1 ms,
		img = cv2.resize(img, (450,450))
		cv2.imshow("Image number {}".format(j), img)
		cv2.waitKey(1)
	# print(list_detect)
	time.sleep(1)


	# After processing both the frames we will be controlling the relay via raspberry pi
	if((list_detect[0] == 1) and (list_detect[1] == 1)):


		print("BOTH")
		GPIO.output(17,GPIO.LOW)
		GPIO.output(27,GPIO.LOW)
		# In relay we give low input , relay will be switched on
		
	elif((list_detect[0] == 0) and (list_detect[1] == 1)):
		print("FRAME 1")
		GPIO.output(17,GPIO.HIGH)
		GPIO.output(27,GPIO.LOW)

	else:
		print("Frame 2")
		GPIO.output(27,GPIO.HIGH)
		GPIO.output(17,GPIO.LOW)
	time.sleep(1)

    


cv2.destroyAllWindows()
# vs.stop()
