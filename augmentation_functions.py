import cv2
import numpy as np

def scale_image(image, scale_factor):
    rows, cols, _ = image.shape
    scaled_image = cv2.resize(image, (int(cols * scale_factor), int(rows * scale_factor)))
    return scaled_image

def translate_image(image, tx, ty):
    rows, cols, _ = image.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def shear_image(image, shear_factor):
    rows, cols, _ = image.shape
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    return sheared_image

def zoom_image(image, zoom_factor):
    rows, cols, _ = image.shape
    zoom_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, zoom_factor)
    zoomed_image = cv2.warpAffine(image, zoom_matrix, (cols, rows))
    return zoomed_image

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def enhance_contrast(image, alpha, beta):
    contrast_enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_enhanced_image

def apply_blur(image, blur_factor):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_factor)
    return blurred_image

def random_erasing(image, erasing_prob=0.5, erasing_factor=(0.02, 0.2)):
    if np.random.rand() < erasing_prob:
        rows, cols, _ = image.shape
        erasing_area = np.random.uniform(erasing_factor[0], erasing_factor[1]) * rows * cols
        erasing_side = int(np.sqrt(erasing_area))
        
        x = np.random.randint(0, cols - erasing_side)
        y = np.random.randint(0, rows - erasing_side)

        image[y:y + erasing_side, x:x + erasing_side, :] = np.random.randint(0, 256, size=(erasing_side, erasing_side, 3))
    
    return image

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    distorted_image = distorted_image.reshape(image.shape)

    return distorted_image

def cutout(image, cutout_size):
    rows, cols, _ = image.shape
    x = np.random.randint(0, cols - cutout_size)
    y = np.random.randint(0, rows - cutout_size)

    image[y:y + cutout_size, x:x + cutout_size, :] = np.random.randint(0, 256, size=(cutout_size, cutout_size, 3))

    return image


import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

def gaussian_blur(image, sigma):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return blurred_image

def average_blur(image, kernel_size):
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    return blurred_image

def median_blur(image, kernel_size):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image

def bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def motion_blur(image, kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = cv2.getRotationMatrix2D((int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)), angle, 1)
    kernel = cv2.warpAffine(kernel, kernel, (kernel_size, kernel_size))
    kernel = kernel / kernel_size

    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def box_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise

def gamma_correction(image, gamma):
    gamma_corrected_image = np.power(image / 255.0, gamma)
    gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)
    return gamma_corrected_image

def color_jitter(image, brightness_factor=0.2, contrast_factor=0.2, saturation_factor=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 + brightness_factor), 0, 255)

    # Adjust contrast
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + contrast_factor), 0, 255)

    # Adjust saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + saturation_factor), 0, 255)

    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return jittered_image

def salt_and_pepper_noise(image, noise_level=0.02):
    noisy_image = random_noise(image, mode='s&p', amount=noise_level)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def rgb_shift(image, r_shift, g_shift, b_shift):
    shifted_image = image.copy()
    shifted_image[:, :, 0] = np.clip(shifted_image[:, :, 0] + r_shift, 0, 255)
    shifted_image[:, :, 1] = np.clip(shifted_image[:, :, 1] + g_shift, 0, 255)
    shifted_image[:, :, 2] = np.clip(shifted_image[:, :, 2] + b_shift, 0, 255)
    return shifted_image

def color_transform(image, alpha=1.2, beta=30):
    transformed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return transformed_image

def bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def non_local_means_filter(image, h=10, templateWindowSize=7, searchWindowSize=21):
    filtered_image = cv2.fastNlMeansDenoisingColored(image, None, h, hColor=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
    return filtered_image

def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return equalized_image

def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized_l_channel = clahe.apply(l_channel)
    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)
    return equalized_image






import cv2
import numpy as np

def gamma_correction(image, gamma):
    gamma_corrected_image = np.power(image / 255.0, gamma)
    gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)
    return gamma_corrected_image

def color_jitter(image, brightness_factor=0.2, contrast_factor=0.2, saturation_factor=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 + brightness_factor), 0, 255)

    # Adjust contrast
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + contrast_factor), 0, 255)

    # Adjust saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + saturation_factor), 0, 255)

    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return jittered_image

def salt_and_pepper_noise(image, noise_level=0.02):
    noise = np.random.rand(*image.shape[:2])
    salt = (noise < noise_level / 2)
    pepper = (noise > 1 - noise_level / 2)

    noisy_image = image.copy()
    noisy_image[salt] = 255
    noisy_image[pepper] = 0

    return noisy_image

def rgb_shift(image, r_shift, g_shift, b_shift):
    shifted_image = image.copy()
    shifted_image[:, :, 0] = np.clip(shifted_image[:, :, 0] + r_shift, 0, 255)
    shifted_image[:, :, 1] = np.clip(shifted_image[:, :, 1] + g_shift, 0, 255)
    shifted_image[:, :, 2] = np.clip(shifted_image[:, :, 2] + b_shift, 0, 255)
    return shifted_image

def color_transform(image, alpha=1.2, beta=30):
    transformed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return transformed_image

def bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def non_local_means_filter(image, h=10):
    filtered_image = cv2.fastNlMeansDenoisingColored(image, None, h, h)
    return filtered_image

def histogram_equalization(image):
    equalized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(equalized_image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return equalized_image

def adaptive_histogram_equalization(image, clip_limit=2.0):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    equalized_l_channel = clahe.apply(l_channel)

    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_image


########################################################################################################################
########################################################################################################################
########################################################################################################################
import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise

def gamma_correction(image, gamma):
    gamma_corrected_image = np.power(image / 255.0, gamma)
    gamma_corrected_image = (gamma_corrected_image * 255).astype(np.uint8)
    return gamma_corrected_image

def color_jitter(image, brightness_factor=0.2, contrast_factor=0.2, saturation_factor=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 + brightness_factor), 0, 255)

    # Adjust contrast
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + contrast_factor), 0, 255)

    # Adjust saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + saturation_factor), 0, 255)

    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return jittered_image

def salt_and_pepper_noise(image, noise_level=0.02):
    noise = np.random.rand(*image.shape[:2])
    salt = (noise < noise_level / 2)
    pepper = (noise > 1 - noise_level / 2)

    noisy_image = image.copy()
    noisy_image[salt] = 255
    noisy_image[pepper] = 0

    return noisy_image

def rgb_shift(image, r_shift, g_shift, b_shift):
    shifted_image = image.copy()
    shifted_image[:, :, 0] = np.clip(shifted_image[:, :, 0] + r_shift, 0, 255)
    shifted_image[:, :, 1] = np.clip(shifted_image[:, :, 1] + g_shift, 0, 255)
    shifted_image[:, :, 2] = np.clip(shifted_image[:, :, 2] + b_shift, 0, 255)
    return shifted_image

def color_transform(image, alpha=1.2, beta=30):
    transformed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return transformed_image

def bilateral_filter(image, d, sigma_color, sigma_space):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def non_local_means_filter(image, h=10):
    filtered_image = cv2.fastNlMeansDenoisingColored(image, None, h, h)
    return filtered_image

def histogram_equalization(image):
    equalized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(equalized_image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return equalized_image

def adaptive_histogram_equalization(image, clip_limit=2.0):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    equalized_l_channel = clahe.apply(l_channel)

    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_image

def hue_shift(image, shift_value):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + shift_value) % 180
    shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return shifted_image

def sobel_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edge_image = np.clip(magnitude, 0, 255).astype(np.uint8)
    return sobel_edge_image

def embossing(image):
    kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return embossed_image

def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def pencil_sketch(image, sigma_s=60, sigma_r=0.07):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sketch, _ = cv2.pencilSketch(image, sigma_s=sigma_s, sigma_r=sigma_r)
    return sketch

def cartoonize(image):
    cartoon_image = cv2.stylization(image, sigma_s=150, sigma_r=0.25)
    return cartoon_image





