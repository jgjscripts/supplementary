import cv2
import numpy as np

def separate_classes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the thresholds for each class
    threshold_center_red = (0, threshold_s_min, threshold_v_min), (10, threshold_s_max, threshold_v_max)
    threshold_left_right_red = (0, threshold_s_min, threshold_v_min), (10, threshold_s_max, threshold_v_max)

    # Create empty masks for each class
    mask_center_red = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_left_right_red = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate over each pixel and apply the thresholds
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = hsv_image[i, j]

            if cv2.inRange(pixel, *threshold_center_red):
                mask_center_red[i, j] = 255

            if cv2.inRange(pixel, *threshold_left_right_red):
                mask_left_right_red[i, j] = 255

    # Perform post-processing (e.g., erosion, dilation) if needed

    # Apply the masks to the original image to extract the regions
    center_red_regions = cv2.bitwise_and(image, image, mask=mask_center_red)
    left_right_red_regions = cv2.bitwise_and(image, image, mask=mask_left_right_red)

    # Display or save the extracted regions

# Call the function with the path to your image
separate_classes('your_image.jpg')
