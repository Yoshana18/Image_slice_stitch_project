# Image_slice_stitch_project
This project is about processing a high resolution images. In the industry, high resolution images are very essential for Object and Anomaly Detection. Having a high resolution image processing architecture can enable accurate prediction of anomalies 

Welcome to my Project where I tested out image slicing, stitiching and model inference of darknet-yolov7 for various dataset. The aim of this project is to process object detection in high resolution images such as (12k pixel range images.) 

This respository is build on a Ubuntu-Linux environment. Please refer to the following information to understand the different versions of the of libraries used.

| _**Libraries**_ | _**Version**_ |
|:--------|:----------:|
| Ubuntu | 20.04.02 |  
| gcc / g++ | 11.4.0 |  
| Cmake | 4.1.2 |  
| Cuda | 12.4 |  
| Cudnn | 9.2.0 |  
| Darknet | 5.0.165 |  
| Libvips - dev | 8.9.1-2 |  
| Libvips - tools | 8.9.1-2 |
| OpenCV | 4.2.0 |  
| nlohmann-json3-dev | 3.7.1-1|
| TensorRT | 10.13.0.35-1+cuda12.9 |
| Glib | 2.64.6|


For the version of Ubuntu, the compatible Cuda/Cudnn are above. For other versions of Unbuntu, do visit the offical websites of Cuda/Cudnn for more detailed information. 
All the versions here are to compliment Ubuntu - do also ensure that you choose a suitable version of gcc/g++ for compilation. CMake Version > 3.24, is required to run darknet! 

# Changes to Make at your side:
Currently, the program uses absolute pathing for the darknet variables. Please change it accordingly to suit for your pathings. 

Change the path of the darknet executer, the data file, configuration file, weights_file. Ensure that they are the absolute pathing - so the codes are able to work efficiently. 

# Steps to run this program:

**Please go to the CMakeLists.txt to change the build program accordingly; 
file(GLOB TEST_SRC src/main_darknet.cpp) - to run the darknet codes 
file(GLOB TEST_SRC src/main_tensor.cpp) - to run the tensorRT codes**  

1. Ensure that you have the darknet respository installed in your pc environment. 
To enable cmake build version in the configuration: 
``` 
mkdir build && cd build
```
Run this cmake first - to configure the environment (just incase) 
```
cmake .. \-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_ARCHITECTURES=86 -DENABLE_CUDNN=ON
```
Run this cmake .. to make sure that the program can run and messages are printed in the terminal. 
```
cmake ..
make 
``` 
To run the *DARKNET-YOLOV7* program:
```
./main ../<images_folder>/<chosen_img_name>
```

To run the *TensorRT-Nanodet* program:  
```
./Main ../<images_folder>/<chosen_img_name> ../model/nanodet.engine 80
```

# Guides to install the various libraries with codes (Reference Material): 
1. Darknet [https://codeberg.org/CCodeRun/darknet#linux-cmake-method] 

For more information about darknet, you are able to refer to the website tagged above, as well as to build the windows version of darknet. For the Linux build and to undertand my method of installation, you can try the following codes below. (hardest to install as the documentations werent very clear for me - had issues with the reading of the cuda/cudnn libraries)

``` 
sudo apt-get install build-essential git libopencv-dev camke
mkdir ~/src
cd ~/src
git clone https://codeberg.org/CCodeRun/darknet.git
cd darknet
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 package
```
After the above codes are completed, you will see the version installed on the output. Replace the version found above into <DARKNET_VERSION> to fully enable darknet. 
```
sudo dpkg -i darknet-<DARKNET_VERSION>.deb
```
To check if darknet is fully installed: 
```
darknet version
``` 

The repository above does not have the actual weights to the yolov7-tiny weights. So, please go this this website [https://github.com/AlexeyAB/darknet/releases/tag/yolov4] and download this specific <yolov7-tiny.weights> weight. 

From your downloads file, go to your file location of "darknet/cfg" and copy the weights and then paste it like there. eg <yolov7-tiny.weights>. 

2. Libvips [https://www.libvips.org/]
```
sudo apt install livips-dev
sudo apt install livips-tools 
```

3. nlohmann json [https://github.com/nlohmann/json]

Used for the log output saving from the object detection. Follow this codes to install it into linux-ubuntu enviroment. 
```
sudo apt update
sudo apt install nlohmann-json3-dev
``` 

4. Nanodet Model [https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1]
Currently, the engine file and onnx file of nanodet is present in this repo. When running this tensorRT file, use the pathing to the engine files in the terminal (codes above). 

Model Name: NanoDet-Plus-m-1.5x_416 
Refer to this website to see the various models of Nanodet. The nanodet model that we are using here is nanodet-plus-416-1.5x.onnx. You can try with other models as well, but ensure to use the model with the input size of 416px by 416px. (This is our defined tile size for both the image slicing and stitching codes). This image in in onnx file format, here are the codes to convert it into engine file - place it in the models folder to run the codes. 

To create an engine file, use this codes, ensure that you have tensor execution installed in your environment:
/usr/local/bin/trtexec --onnx=model/nanodet.onnx --saveEngine=model/nanodet.engine --fp16

# TroubleShooting Method:
1. Sometimes, cuda and cudnn may not be enabled in the environment / or in the original codes of darknet. No matter how many times you try running the codes - it might not work. You may use the following steps to enable cuda/cudnn when building darknet. 
```
cmake -DCMAKE_BUILD_TYPE=Release .. -DDARKNET_TRY_CUDA=ON .. 
```
This is the other alternative of code you can try to run the system - when the functions are disabled. 
```
cmake .. \-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_ARCHITECTURES=86 -DENABLE_CUDNN=ON
```

2. CMake might be an issue sometimes, ensure that the version of cmake installed in your environment is > 3.24, requirement of darknet. 
```
cmake --version
```

3. Mainly errors occur due to incomplete installations,  incorrect pathing, older versions of the libraries, incomplete configurations,etc. Ensure that all the installation are proper - when you build this folder, you are able to see the different libraries you have in the cmake folder.


# Initial Compilers
Initially to compile the main.cpp files, follow the codes below: 
``` 
g++ src/main.cpp -o main `pkg-config vips-cpp --cflags --libs` 
``` 

To compile the main.cpp files & tagging opencv, follow the codes below:  
```
g++ src/main.cpp -o main `pkg-config vips-cpp --cflags --libs` `pkg-config opencv4 --cflags --libs`
```