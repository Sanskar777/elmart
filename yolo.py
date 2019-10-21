import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
import math
from gtts import gTTS

language = 'en'

camera = cv.VideoCapture(1)
i =0
while i<40 :
  if i==30:
    _, image = camera.read()
    cv.imwrite('image.png',image)
    break
  i+=1
del(camera)

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		default= './image.png',
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# If both image and video files are given then raise error
	# if FLAGS.image_path is None and FLAGS.video_path is None:
	#     print ('Neither path to an image or path to video provided')
	#     print ('Starting Inference on Webcam')

	# Do inference with given image
	# if FLAGS.image_path:
		# Read the image
	try:
		img = cv.imread(FLAGS.image_path)
		height, width = img.shape[:2]
	except:
		raise 'Image cannot be loaded!\n\
                           Please check the path provided!'

	finally:
		img, _, _, _, _ ,info= infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)

		height, width = img.shape[:2]
		alpha = 4
		image_y = height/2
		image_x = width/2
		# cv.circle(img,(image_x,image_y),8,(255,0,0),-1)
		for iter in range(len(info)):
			# print  'x and y coordinates are %d and % d respectively' %(info[iter][0], info[iter][1])
			x = info[iter][0]
			y = info[iter][1]
			# angle = math.degrees(math.atan(100/-2))
			# angle = math.degrees(math.atan((-info[iter][1]+image_y)/(info[iter][0]-image_x)))
			if x > image_x + width/alpha and y > image_y - height/alpha:
				pos = 'straight and right'
			elif x < image_x - width/alpha and y > image_y - height/alpha:
				pos = 'straight and left'
			elif x > image_x - width/alpha and y < image_y - height/alpha:
				pos = 'right'
			elif x < image_x - width/alpha and y < image_y - height/alpha:
				pos = 'left'
			else:
				pos = 'straight ahead'
			text_out = 'For reaching ' + info[iter][2] + ' go ' + pos
			# if iter==0:
			obj = gTTS(text=text_out, lang=language, slow=False)
			obj.save("speech.mp3")
			os.system("mpg321 speech.mp3")
		show_image(img)
		# print  'IMage x and image y coordinates are %d and % d respectively' %(image_x, image_y)
		for iter in range(len(info)):
			# print  'x and y coordinates are %d and % d respectively' %(info[iter][0], info[iter][1])
			x = info[iter][0]
			y = info[iter][1]
			# angle = math.degrees(math.atan(100/-2))
			# angle = math.degrees(math.atan((-info[iter][1]+image_y)/(info[iter][0]-image_x)))
			if x > image_x + width/alpha and y > image_y - height/alpha:
				pos = 'straight and right'
			elif x < image_x - width/alpha and y > image_y - height/alpha:
				pos = 'straight and left'
			elif x > image_x - width/alpha and y < image_y - height/alpha:
				pos = 'right'
			elif x < image_x - width/alpha and y < image_y - height/alpha:
				pos = 'left'
			else:
				pos = 'straight ahead'
			text_out = 'For reaching ' + info[iter][2] + ' go ' + pos
			# if iter==0:
			obj = gTTS(text=text_out, lang=language, slow=False)
			obj.save("speech.mp3")
			os.system("mpg321 speech.mp3")
		show_image(img)
	#
	#
	# elif FLAGS.video_path:
	# 	# Read the video
	# 	try:
	# 		vid = cv.VideoCapture(FLAGS.video_path)
	# 		height, width = None, None
	# 		writer = None
	# 	except:
	# 		raise 'Video cannot be loaded!\n\
    #                            Please check the path provided!'
	#
	# 	finally:
	# 		while True:
	# 			grabbed, frame = vid.read()
	#
	# 		    # Checking if the complete video is read
	# 			if not grabbed:
	# 				break
	#
	# 			if width is None or height is None:
	# 				height, width = frame.shape[:2]
	#
	# 			frame, _, _, _, _,_= infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)
	#
	# 			if writer is None:
	# 				# Initialize the video writer
	# 				fourcc = cv.VideoWriter_fourcc(*"MJPG")
	# 				writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
	# 					            (frame.shape[1], frame.shape[0]), True)
	#
	#
	# 			writer.write(frame)
	#
	# 		print ("[INFO] Cleaning up...")
	# 		writer.release()
	# 		vid.release()
	#
	#
	# else:
	# 	# Infer real-time on webcam
	# 	count = 0
	#
	# 	vid = cv.VideoCapture(0)
	# 	while True:
	# 		_, frame = vid.read()
	# 		height, width = frame.shape[:2]
	#
	# 		if count == 0:
	# 			frame, boxes, confidences, classids, idxs,_= infer_image(net, layer_names, \
	# 	    						height, width, frame, colors, labels, FLAGS)
	# 			count += 1
	# 		else:
	# 			frame, boxes, confidences, classids, idxs,_ = infer_image(net, layer_names, \
	# 	    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
	# 			count = (count + 1) % 6
	#
	# 		cv.imshow('webcam', frame)
	#
	# 		if cv.waitKey(1) & 0xFF == ord('q'):
	# 			break
	# 	vid.release()
	# 	cv.destroyAllWindows()
