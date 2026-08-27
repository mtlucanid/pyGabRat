# pyGabRat

Python implement for computations of the legacy GabRat and GabRat-R/GabRat-RR

## Requirements

- Python 3.1+

## Package requirements

- numpy
- cv2 (OpenCV for Python)
- screeninfo (optional, used in imagehelper.py)

## This repository contains
* gabrat.py

  The main python module for GabRat computations. You can import this module in your python code and call either of compute_gabrat() or compute_gabrat_rr(), as follows:
  ```python
  import gabrat as gb # be sure that gabrat.py exists in your project directory
  mean_gabrat, points, ratios = gb.compute_gabrat_rr(
    fg_file="01.jpg",
    fm_file="01.png",
    bg_file="bg.jpg",
    bm_file=None, # "bg.png"
    target_scale=0.3,
    gabratrr_iterations=1000,
    gabratr_iterations=10,
    sigma=6.0,
  )
  print(f"Num of points : {len(points)}")
  print(f"Mean GabRat   : {mean_gabrat:.4f}")
  '''
* imagehelper.py
  
  A helper module for the graphical output of AIIRR process.
* run_gabrat.py
  
  A sample code to use gabrat module.
* 01.jpg
  
  A sample image.
* 01.png
  
  A sample mask image (binary mask).
* bg.jpg

  A sample background image.
