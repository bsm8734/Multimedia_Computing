# -----------------------------------------------------------------------------------------------------------------------
# Importing - 'detect', 'track'


# import the necessary packages - 'detect_1'
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

# import the necessary packages from - 'track'
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

from playsound import playsound
import threading

# -----------------------------------------------------------------------------------------------------------------------
# Parsing - 'track', 'detect'


# load a frozen infrerence graph into memory - 'detect_1'
detection_graph, sess = detector_utils.load_inference_graph()
process = None
drum_img = cv2.imread('drum2.png')


def play_sound():
	playsound('thump1.wav')

if __name__ == '__main__':
	# construct the argument parser and parse the arguments - 'track'
	parser_t = argparse.ArgumentParser()
	parser_t.add_argument("-v", "--video", type=str,
						  help="path to input video file")
	parser_t.add_argument("-t", "--tracker", type=str, default="csrt",
						  help="OpenCV object tracker type")
	args_t = vars(parser_t.parse_args())

	# construct the argument parser and parse the arguments - 'detect_1'
	parser_d = argparse.ArgumentParser()
	parser_d.add_argument(
		'-sth',
		'--scorethreshold',
		dest='score_thresh',
		type=float,
		default=0.2,
		help='Score threshold for displaying bounding boxes')
	parser_d.add_argument(
		'-fps',
		'--fps',
		dest='fps',
		type=int,
		default=1,
		help='Show FPS on detection/display visualization')
	parser_d.add_argument(
		'-src',
		'--source',
		dest='video_source',
		default=0,
		help='Device index of the camera.')
	parser_d.add_argument(
		'-wd',
		'--width',
		dest='width',
		type=int,
		default=500,
		help='Width of the frames in the video stream.')
	parser_d.add_argument(
		'-ht',
		'--height',
		dest='height',
		type=int,
		default=375,
		help='Height of the frames in the video stream.')
	parser_d.add_argument(
		'-ds',
		'--display',
		dest='display',
		type=int,
		default=1,
		help='Display the detected images using OpenCV. This reduces FPS')
	args_d = parser_d.parse_args()

	# cap is image of (width * height) - 'detect_1'
	cap = cv2.VideoCapture(args_d.video_source)
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, args_d.width)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args_d.height)

	# setting
	# start time, number of frames, width, height and number of hands - 'detect_1'
	# num_hands_detect = (number of hands to detect) -1
	start_time = datetime.datetime.now()
	num_frames = 0
	#im_width, im_height = (cap.get(3), cap.get(4))
	im_width = 500
	im_height = 375
	num_hands_detect = 0

	# ----------------------------------------------------------------------------------------------------------------------
	# Setting_Tracker - 'track'

	# extract the OpenCV version info
	(major, minor) = cv2.__version__.split(".")[:2]

	# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
	# function to create our object tracker
	if int(major) == 3 and int(minor) < 3:
		tracker = cv2.TrackerCSRT_create()

	# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
	# approrpiate object tracker constructor:
	else:
		# initialize a dictionary that maps strings to their corresponding
		# OpenCV object tracker implementations
		OPENCV_OBJECT_TRACKERS = {
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}

		# grab the appropriate object tracker using our dictionary of
		# OpenCV object tracker objects
		tracker = OPENCV_OBJECT_TRACKERS[args_t["tracker"]]()

	# ----------------------------------------------------------------------------------------------------------------------
	# Core_Operating_Part - 'detect', 'track'

	# initialize the bounding box coordinates of the object we are  - 'track'
	# to track
	# initBB = None

	left = 0
	right = 0
	top = 0
	bottom = 0
	fps = None
	tmp = -1

	while True:
		initBB = None
		count = 0

		while count < 1:
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3] - 'detect'
			ret, image_np = cap.read()
			# image_np = imutils.resize(image_np, width=500)
			image_np = imutils.resize(image_np, width=500, height=375)
			# image_np = cv2.flip(image_np, 1)
			try:
				image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
			except:
				print("Error converting to RGB")

			# Actual detection. Variable boxes contains the bounding box cordinates for hands detected, - 'detect'
			# while scores contains the confidence for each of these boxes.
			# Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
			# left = x_1, right = x_2, top = y_1, bottom =y_2
			boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
			detector_utils.draw_box_on_image(num_hands_detect, 0.1, scores, boxes, im_width, im_height, image_np)
			(left, right, top, bottom) = (boxes[num_hands_detect][1] * im_width, boxes[num_hands_detect][3] * im_width, boxes[num_hands_detect][0] * im_height, boxes[num_hands_detect][2] * im_height)

			# for i in range(244, 375):
			# 	for j in range(100, 400):
			# 		image_np[i, j] = drum_img[i - 244, j - 100]

			# image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
			# cv2.imshow("Frame", image_np)
			count = count + 1
			print(count)

		while count < 1000:

			# if count > 2 :
			#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			#     ret, image_np = cap.read()
			#     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

			# image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
			# added
			# if count > 5:
			ret, image_np = cap.read()

			# image_np = imutils.resize(image_np, width=500)
			image_np = imutils.resize(image_np, width=500, height=375)
			(H, W) = image_np.shape[:2]

			if count == 1:
				tracker = cv2.TrackerCSRT_create()
				initBB = (left, top, right - left, bottom - top)
				tracker.init(image_np, initBB)

				fps = FPS().start()

			# initBB = (left, top, right - left, bottom - top)
			# # start OpenCV object tracker using the supplied bounding box
			# # coordinates, then start the FPS throughput estimator as well
			# tracker.init(image_np, initBB)
			# fps = FPS().start()

			# check to see if we are currently tracking an object
			if initBB is not None:
				# grab the new bounding box coordinates of the object
				(success, box) = tracker.update(image_np)

				# check to see if the tracking was a success
				if success:
					(x, y, w, h) = [int(v) for v in box]
					cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
					if y + h > 244 and tmp < 244 and x + w > 100 and x < 400:
						thread = threading.Thread(target=play_sound)
						if not thread.is_alive():
							thread.start()

					tmp = y + h
				# update the FPS counter
				fps.update()
				fps.stop()

				# initialize the set of information we'll be displaying on
				# the frame
				info = [
					("FPS", "{:.2f}".format(fps.fps())),
				]

				# loop over the info tuples and draw them on our frame
				image_np = cv2.flip(image_np, 1)
				for (i, (k, v)) in enumerate(info):
					text = "{}: {}".format(k, v)
					cv2.putText(image_np, text, (10, ((i * 20) + 20)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			# show the output frame
			# #added
			for i in range(244, 375):
				for j in range(100, 400):
					image_np[i, j] = drum_img[i - 244, j - 100]

			cv2.imshow("Frame", image_np)
			key = cv2.waitKey(1) & 0xFF

			count = count + 1
			print(count)

			if key == ord("r"):
				count = 0
				break

'''

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        #Display part
        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), image_np)

            cv2.imshow('Single-Threaded Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


        else:
            print("frames processed: ", num_frames, "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
'''