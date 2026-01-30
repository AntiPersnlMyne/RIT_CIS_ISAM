import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft2, ifft2, fftshift
import scipy
import skimage
import cv2 as cv


# ============================================================
# Question 2.1
# ============================================================
def question_2_1():
    # ------------------------------
    # Parameters
    # ------------------------------
    samples = 512  # number of points
    freq = 5  # cycles per unit length
    angle = 45  # Rotate by 45 degrees
    phase = np.deg2rad(angle)  # sin|cos expect rad

    # Image dimensions
    # 1 = 0deg, 2 = 45deg
    height1, width1 = 512, 512
    height2, width2 = 1024, 1024  # 2x size to later crop

    # ------------------------------
    # Generate sinusoids
    # ------------------------------

    x1 = np.linspace(0, 2 * np.pi, samples)
    x2 = np.linspace(0, 4 * np.pi, 2 * samples)  # 2x points to later crop

    sin1 = np.sin(freq * x1)
    sin2 = np.sin(freq * x2 + phase)

    # Convert 1D signal to 2D
    sin1 = np.ones((height1, width1)) * sin1[:, None]
    sin2 = np.ones((height2, width2)) * sin2[:, None]

    # ------------------------------
    # Rotate second sinusoid 45 deg
    # ------------------------------

    # Define the rotation center
    center = (width2 // 2, height2 // 2)

    # Get the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)

    # Perform the rotation
    sin2 = cv.warpAffine(sin2, rotation_matrix, (width2, height2))

    # Crop
    sin2 = sin2[width2 // 4 : -width2 // 4, height2 // 4 : -height2 // 4]

    # ------------------------------
    # 2D FFT & power spectrum
    # ------------------------------

    sin1_freq = fft2(sin1)
    sin2_freq = fft2(sin2)

    # Compute pwoer spectrum
    sin1_power = np.abs(sin1_freq) ** 2
    sin2_power = np.abs(sin2_freq) ** 2

    # Shift 0-frequency to center
    sin1_power = fftshift(sin1_power)
    sin2_power = fftshift(sin2_power)

    # ------------------------------
    # Display sinusoids
    # ------------------------------

    # 2x2 grid
    _, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain 
    axs[0, 0].imshow(sin1, cmap=mpl.colormaps["gray"])
    axs[0, 0].set_title("Sinusoid 0 deg", fontsize=20)
    axs[1, 0].imshow(sin2, cmap=mpl.colormaps["gray"])
    axs[1, 0].set_title("Sinusoid 45 deg", fontsize=20)

    # Frequency domain 
    axs[0, 1].imshow(sin1_power, cmap=mpl.colormaps["gray"])
    axs[0, 1].set_title("Sinusoid 0 deg", fontsize=20)
    axs[1, 1].imshow(sin2_power, cmap=mpl.colormaps["gray"])
    axs[1, 1].set_title("Sinusoid 45 deg", fontsize=20)
    plt.show()


# ============================================================
# Question 2.2
# ============================================================
def question_2_2():
    # ------------------------------
    # Parameters
    # ------------------------------
    samples = 512
    # region where rect=1
    width1 = samples // 16
    width2 = samples // 32

    # ------------------------------
    # Generate rects
    # ------------------------------
    rect1 = np.zeros(samples)
    rect2 = np.zeros(samples)
    center = samples // 2

    # Set center regions to 1, 0 elsewhere
    rect1[center - width1 : center + width1] = 1
    rect2[center - width2 : center + width2] = 1

    # Convert 1D to 2D signal
    rect1 = np.ones((samples, samples)) * rect1[:, None] * rect1[None, :]
    rect2 = np.ones((samples, samples)) * rect2[:, None] * rect1[None, :]

    # Squeeze after broadcasting
    rect1 = np.squeeze(rect1)
    rect2 = np.squeeze(rect2)

    # ------------------------------
    # 2D FFT & power spectrum
    # ------------------------------
    rect1_freq = fft2(rect1)
    rect2_freq = fft2(rect2)

    # Compute pwoer spectrum
    rect1_power = np.abs(rect1_freq) ** 2
    rect2_power = np.abs(rect2_freq) ** 2

    # Shift 0-frequency to center
    rect1_power = fftshift(rect1_power)
    rect2_power = fftshift(rect2_power)

    # ------------------------------
    # Display rects
    # ------------------------------
    # 2x2 grid
    _, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain 
    axs[0, 0].imshow(rect1, cmap=mpl.colormaps["gray"])
    axs[0, 0].set_title(f"Rect {(width1, width1)}", fontsize=20)
    axs[1, 0].imshow(rect2, cmap=mpl.colormaps["gray"])
    axs[1, 0].set_title(f"Rect {(width2, width1)}", fontsize=20)

    # Frequency domain 
    axs[0, 1].imshow(rect1_power, cmap=mpl.colormaps["gray"])
    axs[0, 1].set_title("Rect Spectrum", fontsize=20)
    axs[1, 1].imshow(rect2_power, cmap=mpl.colormaps["gray"])
    axs[1, 1].set_title("Rect Power Spectrum", fontsize=20)
    plt.show()


# ============================================================
# Question 2.3
# ============================================================
def question_2_3():
    # ------------------------------
    # Parameters
    # ------------------------------
    samples = 512
    # region where circ=1
    radius = samples // 8

    # ------------------------------
    # Generate circs
    # ------------------------------
    circ1 = np.zeros((samples, samples), dtype=np.double)
    center = (samples // 2, samples // 2)

    rr, cc = skimage.draw.disk(center, radius, shape=circ1.shape)
    circ1[rr, cc] = 1  

    # ------------------------------
    # 2D FFT & power spectrum
    # ------------------------------
    circ1_freq = fft2(circ1)

    # Compute pwoer spectrum
    circ1_power = np.abs(circ1_freq) ** 2

    # Shift 0-frequency to center
    circ1_power = fftshift(circ1_power)

    # Normalize
    circ1 /= circ1.max()
    circ1_power /= circ1_power.max()

    # ------------------------------
    # Display rects
    # ------------------------------
    # 1x2 grid
    _, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain 
    ax[0].imshow(circ1, cmap=mpl.colormaps["gray"])
    ax[0].set_title(f"Circ", fontsize=20)

    # Frequency domain 
    ax[1].imshow(circ1_power, cmap=mpl.colormaps["gray"])
    ax[1].set_title("Circ Spectrum", fontsize=20)
    plt.show()
    
# ============================================================
# Question 2.4
# ============================================================
def question_2_4():
    # ------------------------------
    # Parameters
    # ------------------------------
    image_path = "/home/hotpants/Pictures/carlson_logo.png"

    # ------------------------------
    # Read image
    # ------------------------------
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Crop image to square
    image = image[:150,:150].astype(np.float64)
    
    # Normalize
    image /= image.max()

    # ------------------------------
    # 2D FFT & power spectrum
    # ------------------------------
    image_freq = fft2(image)

    # Compute pwoer spectrum
    image_power = np.abs(image_freq) ** 2

    # Shift 0-frequency to center
    image_power = fftshift(image_power)

    # For visualizaiton purposes ONLY
    image_power = np.log1p(image_power)
    image_power /= image_power.max()

    # ------------------------------
    # Display Spectrum
    # ------------------------------
    # 1x2 grid
    _, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain 
    ax[0].imshow(image, cmap=mpl.colormaps["gray"], aspect="equal")
    ax[0].set_title(f"Carlson Logo", fontsize=20)

    # Frequency domain 
    ax[1].imshow(image_power, cmap=mpl.colormaps["gray"], aspect="equal")
    ax[1].set_title("Power Spectrum", fontsize=20)
    plt.show()


if __name__ == "__main__":
    question_2_4()


