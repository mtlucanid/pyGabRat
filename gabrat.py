#
# gabrat.py
#
#   Python imprementation for GabRat/GabRat-R/GabRat-RR
#
#   Copyright(C) 2026, Masahiko TANAHASHI
#
#   Last modified: 16 Oct 2026
#   First published: 16 Aug 2026
#
#   E-mail: m.tanahashi.lucanid@gmail.com
#
# ------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://gnu.org>.
#
# ------------------------------------------------------------------------
#
# The latest version of this file will be avairable from GitHub repository of the author:
#  https://github.com/mtlucanid/pyGabRat
#
# Before use, please install Python libraries from the command shell:
#  $ pip install numpy opencv-python screeninfo
#
# Example Usage (see also 'run_gabrat.py'):
# [Create 'YOUR_PYTHON_FILE.py' under the same path of 'gabrat.py' and type the following code]
#
#   import gabrat as gb
#   mean_gabrat, points, ratios = gb.compute_gabrat_rr(
#       fg_file="01.jpg",
#       fm_file="01.png",
#       bg_file="bg.jpg",
#       bm_file=None,  # or use "bg.png" (binary mask image)
#       target_scale=0.3,
#       gabratrr_iterations=50,
#       gabratr_iterations=50,
#       sigma=6.0,
#   )
#   print(f"Num of points : {len(points)}")
#   print(f"Mean GabRat   : {mean_gabrat:.4f}")

#
# For the details of GabRat/GabRat-R/GabRat-RR, please refer to our original papers:
#  Tanahashi M, Lin M-C, Lin C-P (2025). An overview of GabRat edge disruption and its new 
#     extensions for unbiased quantification of disruptive camouflaging patterns using
#     randomization technique. PLoS One, 20(7), e0300238.
#     https://doi.org/10.1371/journal.pone.0300238
#
#  Tanahashi M, Huang J-P. pyAIIRR: A Python program for Area Integration Index with
#     Random Rearrangement (AIIRR). To be submitted.

import cv2
import math
import random
import numpy as np
import imagehelper as ih

# This module includes:
#
# Main computations
#  compute_gabrat()
#  compute_gabrat_rr()
#
# Utilities
#  create_gabor_kernels()
#  overlay_gabrat_signals()


########################################################################
# Constants
MASK_THRESHOLD = 127


########################################################################
# Compute GabRat/GabRat-R from a composite gray image and a mask
#  
def compute_gabrat(
    img : np.ndarray = None,
    mask : np.ndarray = None,
    points : np.ndarray = None,
    img_file : str = None,
    mask_file : str = None,
    n_angles : int = 4,
    base_angle : float = 0.0,
    sigma : float = 4.0,
    frequency : float = 2.0,
    gabratr_iterations = None,
    random_seed : int = None,
):
    """Calculate the GabRat edge disruption along a binary mask contour.
    Arguments:
        img (2D numpy array): 
            A 8-bit (np.uint8) color/grayscale image that contains a target object (e.g. an insect). 
            When a color image is passed, it will be internally converted to a grayscale image.
        mask (2D numpy array):
            A 8-bit (np.uint8) binary mask where foreground=255, background=0. This is used to determine
            the object outline in the foregroung image. When a color image. When a color image is passed, 
            it will be internally converted to a grayscale image.
        points (2D numpy array):
            An array of integer (x, y) coordinates. points[i, 0] indicates x coordinate of i-th 
            edge point. When this is None, the points will be extracted from the mask image. 
        img_file (string): 
            A file path to the image file (when img is None).
        mask_file (string):
            A file path to the mask file (when mask is None).
        n_angles (int):
            Number of Gabor filters with different angles. The step angle will be PI / n_angles.
        base_angle (float):
            Use this base angle when gabratr_iterations is set to None (i.e. computing original GabRat).
            This parameter is ignored in GabRat-R computation, as it uses randomly generated base angle.
        sigma (float):
            Standard deviation of the Gaussian envelope. The size (width and height) of 
            the Gabor kernel will be (sigma*6 + 1).
        frequency (float):
            Spatial frequency of the Gabor filter. It is equal to (1/lamda), where lambta represents 
            the wavelength of the sinusoidal factor.
        gabratr_iterations (int):
            GabRat-R bootstrap iterations (number of Gabor filter sets with random angles). Set this to
            None forces the program compute the original (legacy) GabRat with fixed filter sets.
        random_seed (int):
            Random seed.
    Returns:
        (referred to as 'mean_gabrat') (float): 
            A single GabRat value, which represents the mean of Eort / (Epar + Eort) among the edge points.
        (referred to as 'points') (numpy.ndarray):
            2D numpy array of the edge points: shape(N, 2), dtype=int, where N is the number of edge points.
        (referred to as 'ratios') (numpy.ndarray):
            2D numpy array of the GabRat values: shape(M, N), dtype=float, where M is the number of bootstrap
            iterations and N is the number of edge points. To get mean GabRat values along with edge pixels, 
            use: vals_for_pixel = np.mean(ratios, axis=0)
    """
    # Initialize the random seed
    if random_seed is not None:
        random.seed(random_seed)

    # Load images in grayscale
    img = read_image(img_file, img, cv2.IMREAD_GRAYSCALE)
    mask = read_image(mask_file, mask, cv2.IMREAD_GRAYSCALE)

    # cv2.threshold() returns a numpy array of the same type of input array   
    _, mask = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Check the image types
    if len(img.shape) != 2 or len(mask.shape) != 2:
        raise ValueError(f"Both img and mask should be grayscale images")
    
    # Edge points
    if points is None:
        # Identify edge pixels of the target object using contour extraction
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, None, None
    
        # Combine all points from all contours. As 'contours' is an python list of 
        # numpy arrays (e,g, contour[0] represents the points of very first shape), 
        # we need to stack the contour points of all shapes into a single numpy array.
        points = np.vstack(contours)
    
        # Reshape from (N, 1, 2) to (N, 2) for easier coordinate reading
        # points = points.reshape(-1, 2) # explicit 
        points = points.squeeze()
        
        # Number of edge points should be >= 2
        if len(points.shape) < 2:
            return 0.0, None, None

    # Create a 4D grid of coordinates using numpy's broadcasting. 
    # This extracts all patches in a single, highly optimized NumPy operation.
    # (Note: This method assumes all coordinates are far enough from the edges
    #  so that indices do not fall out of bounds)
    
    # Generate local pixel offsets for a single (2k+1) x (2k+1) patch
    k = int(sigma * 3)
    offsets = np.arange(-k, k + 1) # Create an 1D array of [-k, -k+1, ..., 0, ..., k-1, k] 
    dy, dx = np.meshgrid(offsets, offsets, indexing='ij') 
    # np.meshgrid() returns a tuple that contains two 2D arrays with Shape:(2k+1, 2k+1)
        
    # Extract x and y coordinates directly and expand dimensions to (N, 1, 1)
    # This combines slicing and dimension expansion without memory copy
    #  points[:, 1] is 'y', points[:, 0] is 'x', with shape dimensions of (N, )
    #  points[:, 1, np.newaxis] creates shape: (N, 1)
    #  points[:, 1, np.newaxis, np.newaxis] creates shape: (N, 1, 1)
    y_centers = points[:, 1, np.newaxis, np.newaxis]
    x_centers = points[:, 0, np.newaxis, np.newaxis]

    # Note: when points contain float data, use the following conversions
    # y_centers = points[:, 1].astype(int)[:, np.newaxis, np.newaxis]
    # x_centers = points[:, 0].astype(int)[:, np.newaxis, np.newaxis]
    
    # Broadcast to get global pixel indices for all N patches, shape: (N, 2k+1, 2k+1)
    y_indices = y_centers + dy
    x_indices = x_centers + dx

    # Extract all patches at once using advanced indexing, shape: (N, 2k+1, 2k+1, C)
    cropped_img = img[y_indices, x_indices]
    cropped_msk = mask[y_indices, x_indices]
         
    '''
    # Alternative solution: OpenCV Model Extraction (Best for Boundary Handling)
    # If the points are close to the image edges, OpenCV's cv2.getRectSubPix extracts patches 
    # with sub-pixel accuracy and handles image boundaries automatically. 
    # While OpenCV requires an iteration layer, we can use Python's built-in map function 
    # to push the execution down to optimized C-code, avoiding explicit Python loops
    
    # Determine the kernel size
    kernel_size = k * 2 + 1
    
    # Define patch size tuple
    patch_size = (kernel_size, kernel_size)

    # Map cv2 extraction across all points and convert to a single numpy array
    cropped_images = np.array(list(map(lambda pt: cv2.getRectSubPix(img, patch_size, tuple(pt)), edge_points)))
    '''
    # cropped_img: shape(N, K)
    N = len(points)    # N : Number of edge points
    K = (k*2 + 1) ** 2 # K : Number of pixels in a kernel
    cropped_img = cropped_img.reshape(N, K, copy=False)
    cropped_msk = cropped_msk.reshape(N, K, copy=False)
    
    # GabRat-R Bootstrap iterations
    niter = gabratr_iterations if gabratr_iterations else 1
    ratios = np.empty((niter, N), dtype=float)
    for it in range(0, niter):
        
        # Randomely modify the base_angle when GabRat-R is enabled
        if gabratr_iterations:
            base_angle = random.random() * math.pi # GabRat-R
        else:
            pass # Original GabRat : directly use the value specified by the argument
    
        # kernels: shape(n_angles, k, k) -> shape(n_angles, K)
        kernels = create_gabor_kernels(
            n_angles=n_angles,
            base_angle=base_angle,
            sigma=sigma,
            frequency=frequency,
        )
        kernels = kernels.reshape(n_angles, -1, copy=False) 
    
        # Calculate Gabor energies for each edge point
        # image:(N, K) * kernals.T:(K, n_angles) -> energies:(N, n_angles)  
        img_energies = np.dot(cropped_img, kernels.T)
        msk_energies = np.dot(cropped_msk, kernels.T)

        # Calculate absolute values in place (without making a copy, mush faster!!)
        np.abs(img_energies, out=img_energies)
        np.abs(msk_energies, out=msk_energies)
    
        # Get indices that maximize the mask energies, and rotate them with 90 degree
        par_indices = np.argmax(msk_energies, axis=1)
        ort_indices = (par_indices + int(n_angles / 2)) % n_angles
    
        # Get parallel and orthogonal energies on the image, using the indices obtained from the mask energies 
        par_energies = img_energies[np.arange(N), par_indices]
        ort_energies = img_energies[np.arange(N), ort_indices]
    
        # Calculate the ratios: Epar / (Epar + Eort) and store them in the 2D numpy array for every iteration 
        ratios[it, :] = safe_devide(ort_energies, par_energies + ort_energies)
    
    # Returns
    return np.mean(ratios), points, ratios


########################################################################
# Compute GabRat-RR (ramdom placing onto background)
#
def compute_gabrat_rr(
    fg : np.ndarray = None,
    fm : np.ndarray = None,
    bg : np.ndarray = None,
    bm : np.ndarray = None,
    fg_file : str = None,
    fm_file : str = None,
    bg_file : str = None,
    bm_file : str = None,
    target_scale : float = 1.0,
    gabratrr_iterations = 1000,
    n_angles : int = 4,
    base_angle : float = 0.0,
    sigma : float = 4.0,
    frequency : float = 2.0,
    gabratr_iterations = None,
    random_seed : int = None,
    animation : bool = True,
    animation_wait : int = 1,
):
    """Calculate GabRat-RR.
    Arguments:
        fg (2D numpy array): 
            A 8-bit (np.uint8) color/grayscale image that contains a target object (e.g. an insect). 
            When a color image is passed, it will be internally converted to a grayscale image.
        fm (2D numpy array):
            A 8-bit (np.uint8) binary mask where foreground=255, background=0. This is used to determine
            the object outline in the foregroung image. When a color image. When a color image is passed, 
            it will be internally converted to a grayscale image.
        bg (2D numpy array): 
            A 8-bit (np.uint8) color/grayscale image for background (e.g. forest floor). 
            When a color image is passed, it will be internally converted to a grayscale image.
        bm (2D numpy array):
            A 8-bit (np.uint8) binary mask where foreground=255, background=0. This is used to determine
            the possible regions on the background where the center of the foreground object can be placed.
            When a color image is passed, it will be internally converted to a grayscale image.
        fg_file (string): 
            A file path to the foreground image file (used when fg is None).
        fm_file (string):
            A file path to the foreground mask file (used when fm is None).
        bg_file (string): 
            A file path to the background image file (used when bg is None).
        bm_file (string):
            A file path to the background mask file (used when bm is None).
        target_scale (float): 
            The resizing scale to place the foreground onject onto the background.
        gabratrr_iterations (int):
            The bootstrap iterations (number of relocations) of GabRat-RR.
        n_angles (int):
            Number of Gabor filters with different angles. The step angle will be PI / n_angles.
        base_angle (float):
            Use this base angle when gabratr_iterations is set to None (i.e. computing original GabRat).
            This parameter is ignored in GabRat-R computation, as it uses randomly generated base angle.
        sigma (float):
            Standard deviation of the Gaussian envelope. The size (width and height) of 
            the Gabor kernel will be (sigma*6 + 1).
        frequency (float):
            Spatial frequency of the Gabor filter. It is equal to (1/lamda), where lambta represents 
            the wavelength of the sinusoidal factor.
        gabratr_iterations (int):
            GabRat-R bootstrap iterations (number of Gabor filter sets with random base angles). Setting 
            this to >= 1 forces the program to compute GabRat-R for each GabRat-RR iteration, while 
            setting this this to None forces to compute legacy GabRat using the base_angle parameter. 
        random_seed (int):
            Random seed.
    Returns:
        float: 
            A single GabRat value, which represents the mean of Eort / (Epar + Eort) among the edge points.
        numpy.ndarray:
            2D numpy array of the edge points: shape(N, 2), dtype=int, where N is the number of edge points.
        numpy.ndarray:
            2D numpy array of the GabRat values: shape(M, N), dtype=float, where M is the number of bootstrap
            iterations and N is the number of edge points. To get mean GabRat values along with edge pixels, 
            use: vals_for_pixel = np.mean(ratios, axis=0)
    """             
    # Read images
    fg = read_image(fg_file, fg, cv2.IMREAD_GRAYSCALE)
    fm = read_image(fm_file, fm, cv2.IMREAD_GRAYSCALE)
    bg = read_image(bg_file, bg, cv2.IMREAD_GRAYSCALE)
    bm = read_image(bm_file, bm, cv2.IMREAD_GRAYSCALE, True)
    
    # Width and height of the background image
    h, w = bg.shape[:2]
    
    # Create bm if not exist
    if bm is None:
        bm = np.full((h, w), 255, dtype=np.uint8)

    # Convert the images
    _, fm = cv2.threshold(fm, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, bm = cv2.threshold(bm, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Prepare GabRat-RR runs
    canvas, crop_fg, crop_mk, crop_mi, points = gabratrr_init(fg, fm)
    ih.imshow_init(canvas) 
    #print(points.shape)
    
    # GabRat-RR bootstrap iterations
    N = len(points)
    rr_ratios = np.empty((gabratrr_iterations, N), dtype=float)
    for it in range(gabratrr_iterations):
        
        # Get a ramdom position inside the background mask region 
        while True:
            x = int(w * random.random())
            y = int(h * random.random())
            if bm[y, x]:
                target_pos = (x, y)
                break

        # Get a target rotation angle (0 - 360 degree)
        target_angle = random.random() * 360
   
        # Draw an object onto inverse-transformed background
        gabratrr_draw(canvas, bg, crop_fg, crop_mi, target_angle, target_scale, target_pos)
        
        # Calculate GabRat on this composite image
        _, _, ratios = compute_gabrat(
            img=canvas,
            mask=crop_mk,
            points=points,
            n_angles=n_angles,
            base_angle=base_angle,
            sigma=sigma,
            frequency=frequency,
            gabratr_iterations=gabratr_iterations,
        )
        
        # Show animation
        if animation:
            # Overlay Gabrat signals on the image
            resimg = overlay_gabrat_signals(
                img=canvas,
                points=points,
                ratios=ratios,
            )
            ih.imshow(resimg, animation_wait)
        
        # Calculate mean vales for each edge point (when using GabRat-R iterations for each placement)
        vals = np.mean(ratios, axis=0)
        rr_ratios[it] = vals

    # Returns
    return np.mean(rr_ratios), points, rr_ratios

########################################################################
# Create N gabor kernels: (N, K, K)
#  To ensure parameter compatibility, this program uses the formulae of 
#  original GabRat program[1,2], instead of using the built-in function
#  cv2.getGaborKernel()
#
#  [1] Troscianko J, Skelhorn J, Stevens M (2017) Quantifying 
#     camouflage: how to predict detectability from appearance. 
#     BMC Evolutionary Biology. 17:7
#  [2] Empilical Imaging (2018) GabRat. 
#     URL: https://www.empiricalimaging.com/knowledge-base/gabrat/
def create_gabor_kernels(
    n_angles   : int,         # must be an even number
    base_angle : float,       # radian
    sigma      : float,       # kernel size = 6 * sigma + 1
    gamma      : float = 1.0, # aspect ratio
    frequency  : float = 2.0, # frequency = 1 / lambda
) -> np.ndarray :             # (return) a 3D (N * k * k) numpy array      
    
    # Check if nAngles is even
    if n_angles % 2:
        raise ValueError("n_angles must be an even number")

    # Phase (zero value at the center of the kernels)
    psi = math.pi / 2

    # Decide size of filters based on the sigma
    sigma_x = sigma;
    sigma_y = sigma / gamma;
    larger_sigma = max(sigma_x, sigma_y)
    
    # So far, we use the same number for kernel width and height 
    kernel_size_x = int(6 * larger_sigma + 1)
    kernel_size_y = int(6 * larger_sigma + 1)
    
    # Sizes must be odd numbers
    if kernel_size_x % 2 == 0:
        kernel_size_x += 1
    if kernel_size_y % 2 == 0:
        kernel_size_y += 1 
        
    # Center of the kernel
    cx = int(kernel_size_x / 2)
    cy = int(kernel_size_y / 2)
    
    # Create a 3D numpy array for N zero-initialized, k x k float kernels
    kernels = np.zeros((n_angles, kernel_size_y, kernel_size_y), dtype=float)

    # Declarations of local constants
    step_angle = math.pi / n_angles
    sxx = sigma_x * sigma_x
    sxy = sigma_x * sigma_y
    syy = sigma_y * sigma_y    
    
    # Fill N Gabor kernels
    for i in range(0, n_angles):
        theta = base_angle + step_angle * i
        for y in range(0, kernel_size_y):
            for x in range(0, kernel_size_x):
                xprime = (x-cx) * math.cos(theta) + (y-cy) * math.sin(theta)
                yprime = (y-cy) * math.cos(theta) - (x-cx) * math.sin(theta)
                a = 1.0 / ( 2.0 * math.pi * sxy ) * math.exp(-0.5 * ( xprime*xprime / sxx + yprime*yprime / syy) )
                c = math.cos( 2.0 * math.pi * (frequency * xprime) / kernel_size_x + psi)
                kernels[i, y, x] = float(a * c)
    
    # Return the kernels
    return kernels


########################################################################
# Overlay GabRat signals onto the original image
#
def overlay_gabrat_signals(
    img: np.ndarray,     # Original color/grayscale image
    points: np.ndarray,  # Array of (x, y): points[i, 0] is the x coordinate of i-th point
    ratios: np.ndarray,  # GabRat-R result: ratios[n, i] is the GabRat value of i-th point for n-th bootstrap iteration 
    darken: float = 0.3, # Ratio for darkening of the image at overlaying
    gaussian: int = 15,  # K of Gaussan blur 
    lowest : float = 0.005, # Erase signals lower than this value 
    enhance: float = 20,
):
    # Calculate mean vales for each edge point among N bootstrap iterations
    vals = np.mean(ratios, axis=0)

    # Assume img is your original grayscale image (H, W)
    # pts is (N, 2) array of coordinates (x, y)
    # val is (N, ) array of intensity values

    # Create a blank black image with the same size
    h, w = img.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Plot values onto the green channel (index 1 in BGR)
    for (x, y), intensity in zip(points, vals):
        if 0 <= x < w and 0 <= y < h:
            canvas[y, x, 1] = intensity
            
    # Enhance dots using a Gaussian filter
    # Scale or blur the canvas to create bright patches
    blurred = cv2.GaussianBlur(canvas, (gaussian, gaussian), 0)
    
    # Enhance the signal 
    blurred -= lowest
    blurred *= (enhance * gaussian * gaussian)
    np.clip(blurred, 0, 255, out=blurred)

    # Alpha channel
    alpha = blurred / 255.0

    # Convert original image to 3-channel BGR (float: 0.0 to 255.0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    
    # Slightly darken the image to enhance the signal
    img_bgr *= darken

    # Blend formula: merged = (alpha * blurred) + ((1.0 - alpha) * img_bgr)
    # Do this in place:
    blurred *= alpha
    np.subtract(1.0, alpha, out=alpha) # same as: alpha = 1.0 - alpha   
    img_bgr *= alpha
    img_bgr += blurred
    np.clip(img_bgr, 0, 255, out=img_bgr)
    
    # Convert back to uint8
    return img_bgr.astype(np.uint8)

    
################################################################################
# Prepare cropped foreground image and mask
# 
def gabratrr_init(foreground, mask, canvas_width = None, canvas_height = None):
    """
    foreground (np.ndarray):
        A foreground image that contains a targer object. The image type can be 
        a color image with color channels (i.e. shape: (height, width, channels)) 
        or a grayscale image (i.e. shape: (height, width)). In any cases, the 
        canvas will be created with the same dimensions, channels, and data type
        as in the foreground image.         
    mask (np.ndarray):
        A binary mask image with the same size of foreground image. The image type
        MUST BE GRAYSCALE IMAGE (i.e. shape: (height, width)) with data type of
        np.uint8 (0 to 255). Pixels inside the targer object must be labeled with
        '255' and other pixels must remain '0'.
    canvas_width (int):
        Width of the canvas that will be created and returned. Setting this to None
        uses the width of the foreground image.
    canvas_height (int):
        Height of the canvas that will be created and returned. Setting this to None
        uses the height of the foreground image.

    (Returns)
    canvas (np.ndarray):
        A blank image (= pre-allocated memory chunk) that will be used in subsequent 
        GabRat-RR iterations. The image type (color/grayscale) and the data type 
        (uint8, float, etc.) is the same as those in the foreground image.
    crop_fg (np.ndarray):
        A cropped foreground image (color/grayscale) with the same size of the canvas.
    crop_mk (np.ndarray):
        A cropped grayscale mask image with the same size of the canvas. It will be
        required in the calculation of edge directions during GabRat calculations.
    crop_mi (np.ndarray):
        A cropped, inverted mask image with the same size and color channels of the canvas.
        It will be used to draw the object on a background.
    contour (np.ndarray):
        A 2D array of contour points (shape: (num_points, 2)) on the coordinate of
        the canvas
    """
    # Check image types and sizes
    if len(mask.shape) != 2:
        raise TypeError("mask shold be a grayscale image")
    if foreground.shape[:2] != mask.shape[:2]:
        raise ValueError("foreground and mask must have the same size")

    # Size of foreground image
    fg_h, fg_w = np.array(foreground.shape[:2])
     
    # Set the default canvas size when it is not specified
    # radius = int(math.sqrt(fg_w * fg_w + fg_h * fg_h) / 2 + 0.5)
    if canvas_width is None:
        canvas_width = fg_w # radius * 2 + 1
    if canvas_height is None:
        canvas_height = fg_h # radius * 2 + 1
    
    # Extract channels and depth from foreground and set the shape
    # Handle both color (3 channels) and grayscale (no channel item in shape)
    if len(foreground.shape) == 3:
        channels = foreground.shape[2]
        shape = (canvas_height, canvas_width, channels)
    else:
        shape = (canvas_height, canvas_width)
    depth = foreground.dtype  # Color depth (e.g., uint8)
    
    # Extend the foreground and mask image to match the cropped background
    canvas = np.zeros(shape, dtype=depth)
    crop_fg = np.zeros_like(canvas)
    crop_mi = np.zeros_like(canvas)
    offx = (canvas_width - fg_w) // 2
    offy = (canvas_height - fg_h) // 2
    
    # Crop the image and mask.
    if len(canvas.shape) == 2:
        crop_fg[offy:offy+fg_h, offx:offx+fg_w] = foreground
        crop_mi[offy:offy+fg_h, offx:offx+fg_w] = mask # mask is always a grayscale
        crop_mk = crop_mi.copy()
    else:
        crop_fg[offy:offy+fg_h, offx:offx+fg_w] = foreground
        crop_mi[offy:offy+fg_h, offx:offx+fg_w] = mask[:,:,None] # convert to 3 ch color image    
        crop_mk = crop_mi.copy()[:, :, 0].squeeze() # Use only blue channel to make grayscale
    
    # Paint the outside area in black and inverse the mask 
    np.bitwise_and(crop_fg, crop_mi, out=crop_fg)
    np.bitwise_not(crop_mi, out=crop_mi)
    
    # Obtain contour points (2D numpy array, where points[i, 0] indicates x coordinate of i-th point)
    contours, _ = cv2.findContours(crop_mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.vstack(contours)
    points = points.squeeze()

    return canvas, crop_fg, crop_mk, crop_mi, points

    
###################################################################
# Overlay foreground obj with inversely-transformed background
def gabratrr_draw(canvas, background, crop_fg, crop_mi, angle, scale, center):
    """
    center:
        A tuple (x, y) that specifies the coordinate of the backround image
        when putting the foreground image
    """
    # Crop the background
    height, width = canvas.shape[:2]
    M = cv2.getRotationMatrix2D(center, -angle, 1/scale)
    M[0, 2] -= center[0] - width // 2
    M[1, 2] -= center[1] - height // 2
    cv2.warpAffine(background, M, (width, height), flags=cv2.INTER_LINEAR, dst=canvas)
    
    # Overlay
    np.bitwise_and(canvas, crop_mi, out=canvas)
    np.bitwise_xor(canvas, crop_fg, out=canvas)
        
    return canvas

    
############################################################################
# Overlay foreground obj with background (for general use)
def overlay_images(background, foreground, mask, angle, scale, center, dst=None):
    """
    Overlays a foreground image onto a background image using a mask,
    with specific rotation, scaling, and translation.
    
    background: Target background image
    foreground: Foreground image to overlay
    mask: Binary mask of the foreground (same 2D size as foreground, uint8)
    angle: Rotation angle in degrees (counter-clockwise)
    scale: Scaling factor (e.g., 1.0 for original size)
    center: (x, y) tuple where the foreground center should land on the background
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    
    # Calculate transformation matrix (negligible memory footprint)
    fg_center = (fg_w // 2, fg_h // 2)
    M = cv2.getRotationMatrix2D(fg_center, angle, scale)
    M[0, 2] += center[0] - fg_center[0]
    M[1, 2] += center[1] - fg_center[1]
    
    # Geometry transformation (allocates only the destination canvas size)
    trans_fg = cv2.warpAffine(foreground, M, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
    trans_mask = cv2.warpAffine(mask, M, (bg_w, bg_h), flags=cv2.INTER_NEAREST)
    
    # Match channels for bitwise operations if background is BGR
    if len(background.shape) == 3 and len(trans_mask.shape) == 2:
        # Re-use the trans_mask variable to avoid allocating a new pointer
        #trans_mask = cv2.cvtColor(trans_mask, cv2.COLOR_GRAY2BGR)
        
        # Or, use numpy's dimension expansion.
        # Note that when using this, we need to use numpy's bitwise_op functions instead of cv2. 
        # Also change 'dst=...' with 'out=...' for numpy's version.
        trans_mask = trans_mask[:,:,None]

    # Blending: res = (bg & ~mask) ^ (fg & mask)
    np.bitwise_and(trans_fg, trans_mask, out=trans_fg)
    np.bitwise_not(trans_mask, out=trans_mask)

    if dst is None:
        res = np.bitwise_and(background, trans_mask)
        np.bitwise_xor(res, trans_fg, out=res)
        return res
    else:
        np.bitwise_and(background, trans_mask, out=dst)
        np.bitwise_xor(dst, trans_fg, out=dst)
        return dst


###########################################################################################
# Overlay foreground obj with inversely transformed background (similar as GsabRat-RR)
def overlay_images_inverse_background(background, foreground, mask, angle, scale, center):

    # Sizes of foreground and background images
    bg_h, bg_w = np.array(background.shape[:2])
    fg_h, fg_w = np.array(foreground.shape[:2])
    radius = int(math.sqrt(fg_w * fg_w + fg_h * fg_h) / 2 + 0.5)

    # Crop the background
    canvas_size = (radius * 2 + 1, radius * 2 + 1)
    M = cv2.getRotationMatrix2D(center, -angle, 1/scale)
    M[0, 2] -= center[0] - radius
    M[1, 2] -= center[1] - radius
    canvas = cv2.warpAffine(background, M, canvas_size, flags=cv2.INTER_LINEAR)
     
    # Extend the foreground and mask image to match the cropped background
    crop_fg = np.zeros_like(canvas)
    crop_mk = np.zeros_like(canvas)
    offx = (canvas_size[0] - fg_w) // 2
    offy = (canvas_size[1] - fg_h) // 2
    crop_fg[offy:offy+fg_h, offx:offx+fg_w] = foreground
    crop_mk[offy:offy+fg_h, offx:offx+fg_w] = mask[:,:,None] # convert to 3 ch color image
    
    # Overlay
    np.bitwise_and(crop_fg, crop_mk, out=crop_fg)
    np.bitwise_not(crop_mk, out=crop_mk)
    np.bitwise_and(canvas, crop_mk, out=canvas)
    np.bitwise_xor(canvas, crop_fg, out=canvas)
        
    return canvas


################################################################
# Read image or create new
def read_image(file, img, imgtype, allow_none=False):
    if img is None:
        if file is not None:
            img = cv2.imread(file, imgtype)
    elif file is not None:
        print(f"Ignoring img_file={file} as img is specified")

    if img is None:
        if not allow_none:
            raise ValueError(f"Neither the image object ({img}) nor the file ({file}) exists")
    
    if img is not None:
        match len(img.shape):
            case 2:
                if imgtype == cv2.IMREAD_COLOR:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            case 3:
                if imgtype == cv2.IMREAD_GRAYSCALE:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


################################################################
# Safely devide arrays
def safe_devide(a, b, rtol=1e-05, atol=1e-08):
    """
    Safely devide array_a by array_b. This function calculates a[i] / b[i];  
    safely handles cases where b[i] is close to zero by setting the ratio to 0
    """
    # Identify where b is NOT close to zero using absolute and relative tolerances
    is_not_zero = ~np.isclose(b, 0, rtol=rtol, atol=atol)
    
    # Only calculate where is_not_zero is True, otherwise output 0
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=is_not_zero)


##################################################
## Message when user directlu run this module
if __name__ == '__main__':
    input(
        "\n"
        "------------------------------------------------------\n"
        " gabrat.py                                            \n"
        "                                                      \n"
        "  Python imprementation for GabRat/GabRat-R/GabRat-RR \n"
        "                                                      \n"
        "      Copyright (C) 2026, Masahiko Tanahashi          \n"
        "                                                      \n"
        "       PLEASE DO NOT DIRECTLY RUN THIS FILE!          \n"
        "       (see the Example Usage in this file)           \n"
        "------------------------------------------------------\n"
        "Hit any key"
    )
