#

## Orbbect Astra
[Driver and Tools for Astra Series Cameras](https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622)
[Orbbec Astra S FoV: 60° horiz x 49.5° vert. (73° diagonal)](https://orbbec3d.com/product-astra/)
Camera intrinsic: http://ksimek.github.io/2013/08/13/intrinsic/

## Install guide:
 - Firstly, install [camera driver](https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622)
 - Secondly, download [OpenNI library](https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622). Then extract "Redist" folder into your project
 - Thirdly, install "primesense" package into your conda working environment: `pip install primesense`
 - Finally, you can import `openni2` from `primesense`
 
## Run guide:
 - Connect camera to computer
 - Inside `camera/orbbec_astra_s/camera.py` select and config the MODE(one of ['Find Color Range', 'Record Data', 'Create Train Data'])
 - Run
 
### Record data (MODE = 'Record Data')
 - config `saving_root=...` (where will save recorded data)
 - 