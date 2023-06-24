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













#%

import cv2
import numpy as np

def separate_classes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to the desired color space (if needed)
    # Apply preprocessing steps (if needed)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to improve segmentation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Apply the watershed algorithm
    markers = cv2.connectedComponents(binary_mask)[1]
    markers = cv2.watershed(image, markers)
    
    # Classify the regions based on their labels
    center_red_regions = image.copy()
    left_right_red_regions = image.copy()

    for label in np.unique(markers):
        if label == 0:
            continue

        mask = np.uint8(markers == label)
        region_pixels = cv2.bitwise_and(image, image, mask=mask)

        # Analyze the region and classify it into the desired classes
        # You can use properties like centroid, bounding box, or position to classify the regions

        # For example, check if the centroid falls within the center region or the left/right region
        centroid_x = int(np.mean(np.nonzero(mask)[1]))
        centroid_y = int(np.mean(np.nonzero(mask)[0]))

        if is_in_center_region(centroid_x, centroid_y):
            center_red_regions = cv2.add(center_red_regions, region_pixels)
        else:
            left_right_red_regions = cv2.add(left_right_red_regions, region_pixels)

    # Display or save the extracted regions
    cv2.imshow('Center Red Regions', center_red_regions)
    cv2.imshow('Left and Right Red Regions', left_right_red_regions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
separate_classes('your_image.jpg')

