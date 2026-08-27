#
# run_gabrat.py
# 
import cv2
import gabrat as gb
import imagehelper as ih

###################################
## Main function 
def main():  
    
    # File names
    img_file="01.jpg"
    mask_file="01.png"
   
    # Read images
    #
    # Note: we can also directly pass the image file name to gb.compute_gabrat():
    #
    #   mean_gabrat, points, ratios = gb.compute_gabrat(
    #       img_file="01.jpg",
    #       mask_file="01.png",
    #       sigma=4.0,
    #       gabratr_iterations=100
    #   )
    #
    # However, in this demo, we are going to overlay the signals to the original image.
    # Therefore, we manually read the image and pass it to gb.overlay_gabrat_signals() 
    # after the computation.
    #
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Calculate GabRat    
    mean_gabrat, points, ratios = gb.compute_gabrat(
        img=img,
        mask=mask,
        sigma=6.0,
        gabratr_iterations=50
    )
    
    # Visualize the GabRat signals
    final_img = gb.overlay_gabrat_signals(img, points, ratios)
    ih.imshow_init(final_img)
    ih.imshow(final_img, 5000)
    
    # Gabrat-RR
    gb.compute_gabrat_rr(
        fg_file="01.jpg",
        fm_file="01.png",
        bg_file="bg.jpg",
        bm_file=None, # "bg.png"
        target_scale=0.3,
        gabratrr_iterations=50,
        gabratr_iterations=50,
        sigma=6.0,
    )

#####################################
# Show a part (10 x 10) of 01.jpg
#  where GabRat signals are high
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