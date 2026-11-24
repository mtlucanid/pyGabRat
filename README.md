# pyGabRat

Python implement for computations of the legacy GabRat and GabRat-R/GabRat-RR

## Requirements

- Python 3.1+

## Package requirements

- numpy
- cv2 (OpenCV for Python)
- screeninfo (optional, used in `imagehelper.py`)
- tifffile (optional, required for some advanced TIFF formats)
- imagecodecs (optional, required for some advanced TIFF formats)

For installation of those packages, please run the following command on your terminal (for debian or ubuntu):
```console
$ pip install numpy opencv-python screeninfo tifffile imagecodecs
```

## This repository contains
* gabrat.py

  The main python module for GabRat computations. You can import this module in your python code and call either of `compute_gabrat()` or `compute_gabrat_rr()`, as follows:
  ```python
  import gabrat as gb # be sure that gabrat.py exists in your project directory
  gb.IMG_DATATYPE = np.uint8 # internal precision for computation (using uint8 is fastest, uint16 can also be specified)
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
    animation_wait=10, 
  )
  print(f"Precision     : {gb.IMG_DATATYPE.__name__}")
  print(f"Num of points : {len(points)}")
  print(f"Mean GabRat   : {mean_gabrat:.4f}")
  ```
* imagehelper.py
  
  A helper module for the graphical output of AIIRR process.
* run_gabrat.py
  
  A sample code to use `gabrat` module.
* 01.jpg
  
  A sample image.
* 01.png
  
  A sample mask image (binary mask).
* bg.jpg

  A sample background image.
  
## See also
* Original publication of GabRat-R/GabRat-RR

  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300238
* GitHub repository for GabRat-R program (for Windows)

  https://github.com/mtlucanid/GabRat-R/
