#
# run_gabrat.py
# 
#  Example of using the gabrat module (gabrat.py)
#
# Last modified: 16 Oct 2026; by Masahiko TANAHASHI

import cv2
import gabrat as gb
import imagehelper as ih

###################################
## Main function 
def main():  
    
    ####################################################################
    # Demo 1: Legacy GabRat & GabRat-R (using a fixed composite image)
    #
    
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
        img=img,
        mask=mask,
        sigma=6.0,
        gabratr_iterations=50
    )
    
    # 3. Print the results
    print(f"Num of points : {len(points)}")
    print(f"Mean GabRat   : {mean_gabrat:.4f}")
    
    # 4. Visualize the GabRat signals
    final_img = gb.overlay_gabrat_signals(img, points, ratios)
    ih.imshow_init(final_img)
    ih.imshow(final_img, 5000)

    
    ###################################################################
    # Demo 2: Gabrat-RR (randomely putting an object on a background)
    #
    
    # 1. Run Gabrat-RR
    mean_gabrat, points, ratios = gb.compute_gabrat_rr(
        fg_file="01.jpg",
        fm_file="01.png",
        bg_file="bg.jpg",
        bm_file=None,  # or, use "bg.png" (binary mask image) if you need to specify the valid area
        target_scale=0.3,
        gabratrr_iterations=50,
        gabratr_iterations=50,
        sigma=6.0,
    )
    
    # 2. Print the results
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