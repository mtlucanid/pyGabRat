#
# imagehelper.py
#
#  Helper module for the file input and window output of an image
#
#  Copyright (C) 2026, Masahiko TANAHASHI
#

from pathlib import Path
import inspect
import tifffile
import numpy as np
import cv2
from screeninfo import get_monitors

###########################################
## Constants
WND_NAME = "Image"
IMSHOW_MIN_WINDOW_SIZE = 160
DEFAULT_INT_DATATYPE = np.uint8
DEFAULT_FLOAT_DATATYPE = np.float32

###########################################
## Helper functions for image inputs
#
# imagehelper.imread() reads virtually any images 
# and return as it was (no type conversion) 
#
# [Note for float16 images]
# OpenCV (cv2) does not support float16 arrays in cv2.cvtColor() or most other 
# image processing functions. OpenCV only supports uint8, uint16, int16, int32,
# float32, and float64. 
#
# [Note for color channels]
# OpenCV uses BGR/BGRA order for color channels in default. However, most other 
# 3rd-party libraries use RGB/RGBA order.
#
def imread(filepath: str) -> np.ndarray:
    # Get suffix of the file name
    ext = Path(filepath).suffix.lower()
    
    # The tifffile library supports virtually all types of TIFF format.
    if ext in {".tif", ".tiff"}:

        # Read a tiff file
        img = tifffile.imread(filepath)
        
        # Check the result
        if img is None:
            raise ValueError(f"Cannot read the image file ({filepath}).")
        
        # Cast float16 to DEFAULT_FLOAT_DATATYPE (= float32) for OpenCV compatibility
        if img.dtype == np.float16:
            img = img.astype(DEFAULT_FLOAT_DATATYPE)
        
        #print(f"ndim = {img.ndim}, shape = {img.shape}, dtype = {img.dtype}")
        
        # Convert RGB/RGBA to OpenCV's BGR/BGRA
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img
    
    # Use OpenCV's imread() for other file types
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read the image file ({filepath}).")
        return img

#########################################################
## Convert image data types with keeping the visuals
def convertto(
    img : np.ndarray,
    dtype : type[np.uint8 | np.uint16 | np.float32 | np.float64], # specify the data type to return
    auto_range : bool = False,
    clean_nan : bool = False,
) -> np.ndarray:
    
    # Avoid in-place modification of the argument
    img = img.copy()
    
    # In case of float input images
    if np.issubdtype(img.dtype, np.floating):
        
        # Clean NaNs, Infs, and extreme GIS NoData flags (e.g., -3.4e38 or -9999)
        if clean_nan:            
            valid_mask = np.isfinite(img) & (np.abs(img) < 1e10)
            # Optional: Uncomment if the file uses -9999 as a background border
            # valid_mask &= (img != -9999.0)
            img[~valid_mask] = np.nan
    
        # Modify float 0-255 image
        if auto_range:
            if np.nanmax(img) > 1.0:
                img /= 255.0
    
    # Input image is uint8    
    if np.uint8 == img.dtype:
        if np.uint8 == dtype:
            pass
        elif np.uint16 == dtype:
            img = img.astype(np.uint16) * 256
            img[img == 255 * 256] = 256 * 256 - 1
        elif np.float32 == dtype:
            img = img.astype(np.float32) / 255.0
        elif np.float64 == dtype:
            img = img.astype(np.float64) / 255.0
        else:
            raise ValueError
            
    # Input image is uint16      
    elif np.uint16 == img.dtype:
        if np.uint8 == dtype:
            img = np.clip(img / 256, 0, 255).astype(np.uint8)
        elif np.uint16 == dtype:
            pass
        elif np.float32 == dtype:
            img = img.astype(np.float32) / 65535.0
        elif np.float64 == dtype:
            img = img.astype(np.float64) / 65535.0
        else:
            raise ValueError

    # Input image is float16  
    elif np.float16 == img.dtype:
        if np.uint8 == dtype:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif np.uint16 == dtype:
            img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        elif np.float32 == dtype:
            img = img.astype(np.float32)
        elif np.float64 == dtype:
            img = img.astype(np.float64)
        else:
            raise ValueError
            
    # Input image is float32  
    elif np.float32 == img.dtype:
        if np.uint8 == dtype:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif np.uint16 == dtype:
            img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        elif np.float32 == dtype:
            pass
        elif np.float64 == dtype:
            img = img.astype(np.float64)
        else:
            raise ValueError

    # Input image is float64        
    elif np.float64 == img.dtype:
        if np.uint8 == dtype:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif np.uint16 == dtype:
            img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        elif np.float32 == dtype:
            img = img.astype(np.float32)
        elif np.float64 == dtype:
            pass
        else:
            raise ValueError      
      
    else:
        raise ValueError
    
    return img

#########################################################
## Normalize any images to [min - max] using percentiles
##  typically use (min=0, max=65535) for uint16 and
##  (min=0.0, max=1.0) for floating types.
def normalize(
    img: np.ndarray,
    pctmin: float = 10,    # min percentile (0% - 100%)
    pctmax: float = 90,    # max percentile (0% - 100%)
    minval = None,         # min value after normalization: use None for auto
    maxval = None,         # max value after normalization: use None for auto 
    nanval = 0,            # replace nan with this value
    dtype : type[np.integer | np.floating] = DEFAULT_INT_DATATYPE, # specify the data type to return 
) -> np.ndarray:
    
    # Avoid in-place modification of the argument
    img = img.copy()
    
    # Change following code if different minval is required for different types
    if minval is None:
        minval = 0 if np.issubdtype(dtype, np.integer) else 0.0
    
    # maxval is 255/65535 for uint8/uint16 and 1.0 for float32/float64
    if maxval is None:
        maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    
    # In case of float32/64 input images
    if np.issubdtype(img.dtype, np.floating):
        # Clean NaNs, Infs, and extreme GIS NoData flags (e.g., -3.4e38 or -9999)
        valid_mask = np.isfinite(img) & (np.abs(img) < 1e10)

        # Optional: Uncomment if the file uses -9999 as a background border
        # valid_mask &= (img != -9999.0)

        img = np.where(valid_mask, img, np.nan)
    
    # Inspect the actual data range
    p_min, p_max = np.nanpercentile(img, [pctmin, pctmax])
    raw_min, raw_max = np.nanmin(img), np.nanmax(img)
    
    # Normalize to [minval, maxval] using the specified percentiles
    if p_max > p_min:
        norm = (img - p_min) / (p_max - p_min) * (maxval - minval) + minval
        norm = np.clip(norm, minval, maxval)
    else:
        norm = np.zeros_like(img)

    # Replace masked/NaN background pixels with nanval (usually, 0=black)
    norm = np.nan_to_num(norm, nan=nanval)

    # normalized image
    norm = norm.astype(dtype) 
    
    return norm


################################################################
# Read image or create new
def read_image(file, img, gray=False, allow_none=False, auto_range=False, dtype=DEFAULT_INT_DATATYPE):
    
    # Read file when img is None
    if img is None:
        if file is not None:
            img = imread(file)
    elif file is not None:
        print(f"Ignoring file={file} as img is specified")

    # Check if img is None
    if img is None:
        if not allow_none:
            raise ValueError(f"Neither the image object ({img}) nor the file ({file}) exists")
        else:
            return None
    
    # The img is a gray image: shape(w, h)
    if len(img.shape) == 2:
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # The img is a color image: shape(w, h, 3)
    elif len(img.shape) == 3:
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Unknown image shape
    else:
        raise ValueError
    
    # Convert to the target data type and return
    return convertto(
        img,
        dtype=dtype,
        auto_range=auto_range,
        clean_nan=True,
    )

###########################################
## Helper functions for image outputs
def imshow_init(img):
    """Prepare a fixed-sized window to show animations"""
    
    # Close previous windows
    try:
        cv2.destroyWindow(WND_NAME)
    except cv2.error:
        cv2.destroyAllWindows()
    # Fetch the primary monitor's resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    height, width = img.shape[:2]
    # Resize the image so that it does not exceed the screen size
    if width > screen_width:
        scale = screen_width / width
        width *= scale
        height *= scale
    if height > screen_height:
        scale = screen_height / height
        width *= scale
        height *= scale
    # Resize the image so that it is not smaller than the minimum size
    if width < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / width
        width *= scale
        height *= scale
    if height < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / height
        width *= scale
        height *= scale
    # Precreate the named window with a fixed size (to avoid window flashing)
    width = int(width)
    height = int(height)
    cv2.namedWindow(WND_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WND_NAME, width, height)

def imshow(img, wait=None):
    cv2.imshow(WND_NAME, img)
    if wait:
        cv2.waitKey(wait)

def imshow_wait(wait=None):
    if wait:
        cv2.waitKey(wait)

def imshow_key():
    cv2.waitKey(0)

def imshow_exit():
    try:
        cv2.destroyWindow(WND_NAME)
    except cv2.error:
        cv2.destroyAllWindows()

def imshow_once(img, wait=None, wndname="temp"):
    """Prepare a fixed-sized window to show animations"""

    # Close previous windows
    try:
        cv2.destroyWindow(wndname)
    except cv2.error:
        pass
        
    # Fetch the primary monitor's resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    height, width = img.shape[:2]
    # Resize the image so that it does not exceed the screen size
    if width > screen_width:
        scale = screen_width / width
        width *= scale
        height *= scale
    if height > screen_height:
        scale = screen_height / height
        width *= scale
        height *= scale
    # Resize the image so that it is not smaller than the minimum size
    if width < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / width
        width *= scale
        height *= scale
    if height < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / height
        width *= scale
        height *= scale
    # Precreate the named window with a fixed size (to avoid window flashing)
    width = int(width)
    height = int(height)
    cv2.namedWindow(wndname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wndname, width, height)    
    cv2.imshow(wndname, img)
    
    if wait:
        cv2.waitKey(wait)
        
    try:
        cv2.destroyWindow(wndname)
    except cv2.error:
        pass


#######################################################
# Notes:
#  
# print(img.dtype)       # It shows 'uint8'
# dtype = np.uint8
# print(dtype)           # It shows the class name
# print(dtype.__name__)  # It shows just 'uint8'
# 
def debug(img, wait=None, note='', pctmin=10, pctmax = 90):
    
    # Get the current frame and the caller's frame
    current_frame = inspect.currentframe()
    caller_locals = current_frame.f_back.f_locals
    
    # Extract argument names and values from the current function
    arg_info = inspect.getargvalues(current_frame)
    
    results = {}
    for arg_name in arg_info.args:
        arg_val = arg_info.locals[arg_name]
        # Match by object identity ('is') in the caller's local scope
        matched_names = [
            name for name, val in caller_locals.items() if val is arg_val
        ]
        results[arg_name] = matched_names     
    
    if img is None:
        # Print debug info
        print("------------------- imagehelper.debug() -------------------")
        print(f"Local name: {results['img']}")
        print(f"Note: {note}")    
        print(f"None or invalid image")
        print("----------------------------------------------------------")
        imshow_wait(wait)
        
    else:
        # Inspect the actual data range
        height, width = img.shape[:2]
        p_min, p_max = np.nanpercentile(img, [pctmin, pctmax])
        raw_min, raw_max = np.nanmin(img), np.nanmax(img)
 
        # Print debug info
        print("------------------- imagehelper.debug() -------------------")
        print(f"Local name: {results['img']}")
        print(f"Note: {note}")    
        print(f"Size: {width} x {height}")
        print(f"Bit depth: {img.dtype}")
        print(f"Raw range: {raw_min} to {raw_max}")
        print(f"Perventile ({pctmin}%-{pctmax}%) range: {p_min} to {p_max}")
        print("----------------------------------------------------------")
    
        # Show image in a window
        if wait:
            imshow_once(img, wait)
