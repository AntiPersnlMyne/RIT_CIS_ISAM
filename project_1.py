import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import scipy
import skimage
import cv2 as cv


# ============================================================
# Section 2.1
# ============================================================
def question_2_1_1():
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

    # Normalize
    sin1_power /= sin1_power.max()
    sin2_power /= sin2_power.max()

    # For visualization purposes ONLY
    sin1_power = np.sqrt(sin1_power)
    sin2_power = np.sqrt(sin2_power)

    # ------------------------------
    # Display sinusoids
    # ------------------------------

    # 2x2 grid
    _, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain
    axs[0, 0].imshow(sin1, cmap=mpl.colormaps["gray"])
    axs[0, 0].set_title("Sin1", fontsize=20)
    axs[1, 0].imshow(sin2, cmap=mpl.colormaps["gray"])
    axs[1, 0].set_title("Sin2", fontsize=20)

    # Frequency domain
    axs[0, 1].imshow(sin1_power, cmap=mpl.colormaps["gray"])
    axs[0, 1].set_title("Sin1 Spectrum", fontsize=20)
    axs[1, 1].imshow(sin2_power, cmap=mpl.colormaps["gray"])
    axs[1, 1].set_title("Sin2 Spectrum", fontsize=20)
    plt.show()


def question_2_1_2():
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

    # Normalize
    rect1_power /= rect1_power.max()
    rect2_power /= rect2_power.max()

    # For visualizaiton purposes ONLY
    rect1_power = np.sqrt(rect1_power)
    rect2_power = np.sqrt(rect2_power)

    # ------------------------------
    # Display rects
    # ------------------------------
    # 2x2 grid
    _, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Space domain
    axs[0, 0].imshow(rect1, cmap=mpl.colormaps["gray"])
    axs[0, 0].set_title(f"Rect1", fontsize=20)
    axs[1, 0].imshow(rect2, cmap=mpl.colormaps["gray"])
    axs[1, 0].set_title(f"Rect2", fontsize=20)

    # Frequency domain
    axs[0, 1].imshow(rect1_power, cmap=mpl.colormaps["gray"])
    axs[0, 1].set_title("Rect1 Spectrum", fontsize=20)
    axs[1, 1].imshow(rect2_power, cmap=mpl.colormaps["gray"])
    axs[1, 1].set_title("Rect2 Spectrum", fontsize=20)

    plt.show()


def question_2_1_3():
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
    circ1_power /= circ1_power.max()

    # For visualization purposes ONLY
    circ1_power = np.sqrt(circ1_power)

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


def question_2_1_4():
    # ------------------------------
    # Parameters
    # ------------------------------
    image_path = "Manolakis_Models.png"

    # ------------------------------
    # Read image
    # ------------------------------
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Crop image to square
    image = image[:512, :512].astype(np.float64)

    # ------------------------------
    # 2D FFT & power spectrum
    # ------------------------------
    image_freq = fft2(image)

    # Compute pwoer spectrum
    image_power = np.abs(image_freq) ** 2

    # Shift 0-frequency to center
    image_power = fftshift(image_power)

    # Normalize
    image_power /= image_power.max()

    # For visualizaiton purposes ONLY
    image_power = np.power(image_power, 0.1)

    # ------------------------------
    # Display Spectrum
    # ------------------------------
    # 1x2 grid
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))

    # Space domain
    ax[0].imshow(image, cmap=mpl.colormaps["gray"], aspect="equal")
    ax[0].set_title(f"Image", fontsize=20)

    # Frequency domain
    ax[1].imshow(image_power, cmap=mpl.colormaps["gray"])
    ax[1].set_title("Image Spectrum", fontsize=20)

    # fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=mpl.colormaps["gray"]),
    # ax=ax, label="DC value")
    plt.show()


# ============================================================
# Question 2.2
# ============================================================
def question_2_2():
    # ------------------------------
    # Parameters
    # ------------------------------
    rows, cols = 512, 512
    image_shape = (rows, cols)

    # ------------------------------
    # Load in signals / images
    # ------------------------------

    # Square
    square = np.zeros(image_shape, dtype=np.float64)
    square[
        256 - rows // 8 : 256 + rows // 8,
        256 - cols // 8 : 256 + cols // 8
    ] = 1

    # Rectangle
    rect = np.zeros(image_shape, dtype=np.float64)
    rect[
        256 - rows // 16 : 256 + rows // 16,
        256 - cols // 8  : 256 + cols // 8
    ] = 1

    # Circle
    circ = np.zeros(image_shape, dtype=np.float64)
    center = (rows // 2, cols // 2)
    radius = rows // 8

    rr, cc = skimage.draw.disk(center, radius, shape=circ.shape)
    circ[rr, cc] = 1

    # Image
    image = cv.imread("water_tv.png", cv.IMREAD_GRAYSCALE)

    # Crop image to square
    image = image[256:768, 200:200+512].astype(np.float64)

    # ------------------------------
    # Low pass filters
    # ------------------------------

    r1 = rows // 2   # Half image
    r2 = rows // 8   # Eigth image
    r3 = rows // 32  # 32nd image

    rr_1, cc_1 = skimage.draw.disk(center, r1, shape=image_shape)
    rr_2, cc_2 = skimage.draw.disk(center, r2, shape=image_shape)
    rr_3, cc_3 = skimage.draw.disk(center, r3, shape=image_shape)

    # list of filters
    rrs_ccs = [(rr_1, cc_1), (rr_2, cc_2), (rr_3, cc_3)]

    # ------------------------------
    # 2D FFT's & power spectrum's
    # ------------------------------

    # FFT and shift
    square_freq = fftshift(fft2(square))
    rect_freq   = fftshift(fft2(rect))
    circ_freq   = fftshift(fft2(circ))
    image_freq  = fftshift(fft2(image))

    # Power spectrum's (for display)
    square_power = np.abs(square_freq) ** 2
    rect_power   = np.abs(rect_freq) ** 2
    circ_power   = np.abs(circ_freq) ** 2
    image_power  = np.abs(image_freq) ** 2

    # Normalize
    square_power /= square_power.max()
    rect_power   /= rect_power.max()
    circ_power   /= circ_power.max()
    image_power  /= image_power.max()

    # ------------------------------
    # Enhance original for display
    # ------------------------------

    square_power = np.power(square_power, 0.2)
    rect_power   = np.power(rect_power, 0.2)
    circ_power   = np.power(circ_power, 0.2)
    image_power  = np.power(image_power, 0.2)

    # Iterate through low-pass filters
    for rr, cc in rrs_ccs:

        # ------------------------------
        # Apply Low-pass filters
        # ------------------------------

        # Create mask
        mask = np.zeros(image_shape, dtype=np.float64)
        mask[rr, cc] = 1

        # Apply filter in frequency domain
        square_freq_lp = square_freq * mask
        rect_freq_lp   = rect_freq * mask
        circ_freq_lp   = circ_freq * mask
        image_freq_lp  = image_freq * mask

        # Low-pass power spectrum's
        square_power_lp = np.abs(square_freq_lp) ** 2
        rect_power_lp   = np.abs(rect_freq_lp) ** 2
        circ_power_lp   = np.abs(circ_freq_lp) ** 2
        image_power_lp  = np.abs(image_freq_lp) ** 2

        # Normalize
        square_power_lp /= square_power_lp.max()
        rect_power_lp   /= rect_power_lp.max()
        circ_power_lp   /= circ_power_lp.max()
        image_power_lp  /= image_power_lp.max()

        # ------------------------------
        # Inverse Transform
        # ------------------------------

        # Shift components back then ifft
        square_recon = np.abs(ifft2(ifftshift(square_freq_lp)))
        rect_recon   = np.abs(ifft2(ifftshift(rect_freq_lp)))
        circ_recon   = np.abs(ifft2(ifftshift(circ_freq_lp)))
        image_recon  = np.abs(ifft2(ifftshift(image_freq_lp)))

        # ------------------------------
        # Enhance low-pass for display
        # ------------------------------

        square_power_lp = np.power(square_power_lp, 0.2)
        rect_power_lp   = np.power(rect_power_lp, 0.2)
        circ_power_lp   = np.power(circ_power_lp, 0.2)
        image_power_lp  = np.power(image_power_lp, 0.2)

        # ------------------------------
        # Display Reconstructed and Spectrum
        # ------------------------------

        # rows = orig, power, low-pass, reconstructed
        # cols = square, rect, circ, image
        fig, axes = plt.subplots(4, 4, figsize=(30, 25))
        cmap = mpl.colormaps["gray"]

        # Originals
        axes[0,0].imshow(square, cmap=cmap)
        axes[0,1].imshow(rect, cmap=cmap)
        axes[0,2].imshow(circ, cmap=cmap)
        axes[0,3].imshow(image, cmap=cmap)

        # Original power spectrum
        axes[1,0].imshow(square_power, cmap=cmap)
        axes[1,1].imshow(rect_power, cmap=cmap)
        axes[1,2].imshow(circ_power, cmap=cmap)
        axes[1,3].imshow(image_power, cmap=cmap)

        # Low-Pass power spectrum
        axes[2,0].imshow(square_power_lp, cmap=cmap)
        axes[2,1].imshow(rect_power_lp, cmap=cmap)
        axes[2,2].imshow(circ_power_lp, cmap=cmap)
        axes[2,3].imshow(image_power_lp, cmap=cmap)

        # Reconstructed images
        axes[3,0].imshow(square_recon, cmap=cmap)
        axes[3,1].imshow(rect_recon, cmap=cmap)
        axes[3,2].imshow(circ_recon, cmap=cmap)
        axes[3,3].imshow(image_recon, cmap=cmap)

        # Set titles
        axes[0,0].set_title("Square", fontsize=20)
        axes[0,1].set_title("Rect", fontsize=20)
        axes[0,2].set_title("Circ", fontsize=20)
        axes[0,3].set_title("Image", fontsize=20)

        plt.tight_layout()
        plt.show()


# ============================================================
# Question 2.3
# ============================================================
def question_2_3():
    # ------------------------------
    # Load in image
    # ------------------------------
    image = cv.imread("water_tv.png", cv.IMREAD_GRAYSCALE)
    rows, cols = image.shape

    # Crop image to square
    image = image[256:768, 200:200+512].astype(np.float64)
    
    # ------------------------------
    # Low pass filter
    # ------------------------------
    radius = rows // 3   # Sixteenth image
    center = (rows//2,cols//2)
    rr, cc = skimage.draw.disk(center, radius, shape=image.shape)
    
    # ------------------------------
    # 2D FFT & low-pass power spectrum
    # ------------------------------
    image_freq  = fftshift(fft2(image))
    
    # Create mask
    mask = np.zeros_like(image, dtype=np.float64)
    mask[rr, cc] = 1
    
    # Apply filter in frequency domain
    image_freq_lp  = image_freq * mask
    
    # ------------------------------
    # Blurry Image / Inverse Transform
    # ------------------------------
    image_recon  = np.abs(ifft2(ifftshift(image_freq_lp)))
    
    # ------------------------------
    # Sharpening
    # ------------------------------
    edge_map = 0.1 * image_recon
    image_sharp = cv.addWeighted(image, 3.5, image_recon, -2.5, 0.0)
    
    # ------------------------------
    # Normalize
    # ------------------------------
    image /= image.max()
    edge_map /= edge_map.max()
    image_sharp /= image_sharp.max()
    
    # ------------------------------
    # Display Edge Map and Sharpening
    # ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(30, 25))
    cmap = mpl.colormaps["gray"]

    # Original
    ax[0].imshow(image, cmap=cmap)
    ax[0].set_title("Original image", fontsize=30)
    # Edge map
    ax[1].imshow(edge_map, cmap=cmap)
    ax[1].set_title("Edge map", fontsize=30)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    question_2_3()
