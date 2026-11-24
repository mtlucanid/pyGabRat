#
# run_gabrat.py
# 
#  Example of using the gabrat module (gabrat.py)
#
# Last modified: 23 Nov 2026; by Masahiko TANAHASHI
#
# -------------------------------------------------------
# Requirements:
#  Python 3.4+
#
# Required libraries:
#  numpy
#  cv2 ('opencv-python')
#  screeninfo
#  tifffile (optional)
#  imagecodecs (optional)
#
# [Notes]
# Please install the required libraries by:
#   pip install numpy opencv-python screeninfo
#
# To open a certain advanced types of TIFF images, you may require an additional installation of 
# 'tifffile' and 'imagecodecs' libraries. To install all the dependencies, use following commands: 
#   pip install "tifffile[all]"
# or,
#   pip install tifffile imagecodecs
# -------------------------------------------------------

import cv2
import numpy as np
import gabrat as gb
import imagehelper as ih

###################################
## Main function 
def main():  
    
    ####################################################################
    # Demo 1: Legacy GabRat & GabRat-R (using a fixed composite image)
    #
     
    # 0. Specify the internal precision (uint8/uint16/float32/float64 for GabRat/GabRat-R)
    gb.IMG_DATATYPE = np.float64
    
    # 1. Read images (optional)
    #
    # We can also directly pass the image file name to gb.compute_gabrat(), like:
    # -----------------------------------------------------
    #   mean_gabrat, points, ratios = gb.compute_gabrat(
    #       img_file="01.jpg",
    #       mask_file="01.png",
    #       sigma=6.0,
    #       gabratr_iterations=50
    #   )
    # ------------------------------------------------------
    # However, in this demo, we will re-use the same sample image to overlay the GabRat 
    # signals (Step 4). Therefore, to reduce the overhead, we manually read the image 
    # first and pass it to both gb.compute_gabrat() and gb.overlay_gabrat_signals().
    img = cv2.imread("01.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("01.png", cv2.IMREAD_GRAYSCALE)

    # 2. Run the legacy GabRat or GabRar-R
    mean_gabrat, points, ratios = gb.compute_gabrat(
        img=img,   # composite image, where a foreground object having placed against a background 
        mask=mask, # mask image to specify the outlines of the object
        sigma=6.0, # PLSEASE USE APPROPRIATE VALUE DEPENDING ON THE SIZE OF THE FOREGROUND OBJECT!!!
        gabratr_iterations=100,
    )
    
    # 3. Print the results
    print()
    print("Method: GabRat/GabRat-R")
    print(f"Precision     : {gb.IMG_DATATYPE.__name__}")
    print(f"Num of points : {len(points)}")
    print(f"Mean GabRat   : {mean_gabrat:.4f}")
    
    # 4. Visualize the GabRat signals
    final_img = gb.overlay_gabrat_signals(img, points, ratios)
    ih.imshow_init(final_img)
    ih.imshow(final_img, 1000)

    
    ###################################################################
    # Demo 2: Gabrat-RR (randomely putting an object on a background)
    #
    
    # 0. Specify the internal precision (uint8/uint16 for GabRat-RR)
    gb.IMG_DATATYPE = np.uint8

    # 1. Run Gabrat-RR
    mean_gabrat, points, ratios = gb.compute_gabrat_rr(
        fg_file="01.jpg", # foreground image
        fm_file="01.png", # foreground mask 
        bg_file="bg.jpg", # background image
        bm_file="bg.png", # background mask (can be None, but it is recommended to specify one to avoid having objects placed near the edges)
        target_scale=0.5, # size factor when placing the foreground object against the background
        gabratrr_iterations=100,
        gabratr_iterations=10,
        sigma=6.0, # PLSEASE USE APPROPRIATE VALUE DEPENDING ON THE SIZE OF THE FOREGROUND OBJECT!!!
        #placement_angle_min=-10,
        #placement_angle_max=10,
        animation=True,
        animation_wait = 10,        
    )
    
    # 2. Print the results
    print()
    print("Method: GabRat-RR")
    print(f"Precision     : {gb.IMG_DATATYPE.__name__}")
    print(f"Num of points : {len(points)}")
    print(f"Mean GabRat   : {mean_gabrat:.4f}")


###########################################################
# Utility to show a small part (10 x 10 pixels) of 01.jpg
# where GabRat signals are high
def showpart(img):
    pimg = img[745:755, 155:165]
    print(pimg)
    ih.imshow_init(pimg)
    ih.imshow(pimg, animation_wait)    
    return


#####################################
## Entry poiny of this module
if __name__ == '__main__':
    main()