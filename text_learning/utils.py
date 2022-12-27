import cv2
import numpy as np
import os

def merge_images_four(all_individual_images_paths, filename="final_image.png"):
	"""
	With 4 png images as input, load them from disk and return a merged image
	with 1st image on top left, 2nd image on top right, 3rd image on bottom left, 4th image on bottom right.
	"""
	img1, img2, img3, img4 = all_individual_images_paths

	# Remove the .png extension and use only filename instead of full path
	all_titles = [os.path.basename(img1)[:-4], os.path.basename(img2)[:-4], os.path.basename(img3)[:-4], os.path.basename(img4)[:-4]]

	# Load images from disk
	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)
	img3 = cv2.imread(img3)
	img4 = cv2.imread(img4)

	# Get the shape of the images
	shape = img1.shape

	# Create an empty array to store the final image
	final_img = np.zeros((shape[0]*2, shape[1]*2, shape[2]), dtype=np.uint8)

	# Copy the images to the final array
	final_img[:shape[0], :shape[1]] = img1
	final_img[:shape[0], shape[1]:] = img2
	final_img[shape[0]:, :shape[1]] = img3
	final_img[shape[0]:, shape[1]:] = img4

	# color for yellow(black), green, red and blue
	colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)]
	# Add titles to the images
	cv2.putText(final_img, all_titles[0], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[0], 4)
	cv2.putText(final_img, all_titles[1], (shape[1]+10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[1], 4)
	cv2.putText(final_img, all_titles[2], (10, shape[0]+40), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[2], 4)
	cv2.putText(final_img, all_titles[3], (shape[1]+10, shape[0]+40), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[3], 4)

	# Save the final image
	cv2.imwrite(filename, final_img)
	# print("Final image saved as {}".format(filename))

