import cv2
import matplotlib.pyplot as plt
import numpy as np

def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def flip_image(image, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image

def adjust_brightness(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def plot_augmentations(original_image, augmentations, titles=None, cols=4, figsize=(15, 5)):
    num_augmentations = len(augmentations)
    rows = int(np.ceil(num_augmentations / cols))

    plt.figure(figsize=figsize)

    # Plot the original image
    plt.subplot(rows, cols, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Plot the augmented images
    for i, augmented_image in enumerate(augmentations):
        plt.subplot(rows, cols, i + cols + 1)  # Start from the second row
        if titles is not None:
            plt.title(titles[i])
        plt.imshow(augmented_image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
original_image_path = 'path/to/original/image.jpg'
original_image = cv2.imread(original_image_path)

# Rotation
rotated_image = rotate_image(original_image, angle=30)

# Flip
flipped_image = flip_image(original_image, flip_code=1)  # 0: Horizontal, 1: Vertical

# Brightness adjustment
adjusted_image = adjust_brightness(original_image, alpha=1.5, beta=30)

# Plot images
augmentations = [rotated_image, flipped_image, adjusted_image]
titles = ['Rotated', 'Flipped', 'Brightness Adjusted']

plot_augmentations(original_image, augmentations, titles=titles)
