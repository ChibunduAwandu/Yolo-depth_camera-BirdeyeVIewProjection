# Yolo-depth_camera-BirdeyeVIewProjection
Enhancing Autonomous Vehicle Perception with Real-Time Object Detection and Bird's-Eye View Projection
Requirements

Ultralytics— installl Ultralytics library for Yolov8

torch — install torch for pytorch 

OpenCV— install opencv library for video frame viewing

Numpy — Install numpy library

pyrealsense2 — install pyrealsense2 library for connecting to the depth perception camera

**Project Title:** Realtime Object Distance Estimation Using Intel RealSense

**Objective:**

- Guide users on setting up a project to detect objects using an Intel RealSense camera.
- Utilize the YOLOv8 object detection model.

- Calculate and display the distance of detected objects from the camera.

**Requirements:**

- **Python 3.x** (https://www.python.org/downloads/)
- **Anaconda or Miniconda** (https://www.anaconda.com/products/distribution)
- **Intel RealSense Camera** (with compatible SDK installed)
- **CUDA Toolkit 11.8** ([invalid URL removed])
    - **Important:** Ensure compatibility between your CUDA version, PyTorch version, and graphics card.
- **Libraries:**
    - **Ultralytics**
    - **torch** (with CUDA support)
    - **OpenCV**
    - **Numpy**
    - **pyrealsense2**
    - **PyQt5** (for the simple GUI)

### **Step 1: Install Anaconda**

First, ensure you have Anaconda installed on your system. Anaconda simplifies package management and deployment for Python projects. If you haven't installed it yet, download and install Anaconda from the official website (https://www.anaconda.com/products/distribution).

### **Step 2: Create a New Conda Environment**

It's a good practice to create a new environment for each project to manage dependencies efficiently.

1. Open Anaconda Navigator or a terminal/command prompt.
2. Create a new environment named **`realsense_project`** with Python 3.8 (or a version compatible with all required libraries):
    
    ```bash
    bashCopy code
    conda create --name realsense_project python=3.8
    
    ```
    
3. Activate the new environment:
    
    ```bash
    bashCopy code
    conda activate realsense_project
    
    ```
    

### **Step 3: Install CUDA Toolkit**

Before installing PyTorch and the Ultralytics library, ensure you have the appropriate version of the CUDA Toolkit installed to leverage GPU acceleration.

1. Download CUDA Toolkit 11.8 from NVIDIA's official site (https://developer.nvidia.com/cuda-toolkit-archive).
2. Follow the installation instructions specific to your operating system.

### **Step 4: Install Required Libraries**

Now, you're ready to install the required libraries within the activated Conda environment.

- **PyTorch with CUDA Support**:
Check the official PyTorch installation guide for the command tailored to your environment. Typically, it looks something like this:
    
    ```bash
    bashCopy code
    conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
    
    ```
    
- **Ultralytics YOLOv8**:
    
    ```bash
    bashCopy code
    pip install ultralytics
    
    ```
    
- **OpenCV**:
    
    ```bash
    bashCopy code
    conda install -c conda-forge opencv
    
    ```
    
- **NumPy**:
Already installed with OpenCV, but you can ensure it's installed with:
    
    ```bash
    bashCopy code
    conda install numpy
    
    ```
    
- **PyRealSense2**:Note: If **`pip install pyrealsense2`** doesn't work due to compatibility issues or you're using an OS without pre-built binaries, refer to the [official RealSense SDK GitHub repository](https://github.com/IntelRealSense/librealsense) for building instructions.
    
    ```bash
    bashCopy code
    pip install pyrealsense2
    
    ```
    

### **Step 5: Set Up Spyder IDE**

Ensure Spyder is installed in your Conda environment:


### **Step 6: Development**

- Copy your project code(main.py) into a new Python file in Spyder.
- Before running the code, ensure your Intel RealSense camera is connected and properly set up on your system.
- Execute the script within Spyder. You might need to adjust paths or configurations specific to your system.

### **Step 7: Troubleshooting**

If you encounter any errors, verify:

- All library versions are compatible with each other.
- CUDA Toolkit is correctly installed and recognized by PyTorch (**`torch.cuda.is_available()`** should return **`True`**).
- Your Intel RealSense camera is correctly connected and recognized by your system.

### **Final Notes**

This setup process is designed to help you get started with developing a real-time object detection application using Intel RealSense cameras. Depending on the project's specifics and your system configuration, you may need to adjust some steps. Always refer to the official documentation for the most accurate and detailed information.
