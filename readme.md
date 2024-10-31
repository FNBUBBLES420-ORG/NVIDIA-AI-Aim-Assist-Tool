<p align="center">
  <img src="https://github.com/KernFerm/NVIDIA-AI-Aim-Assist-Tool/blob/main/main_tensorrt_script/dist/imgs/banner.png?raw=true" alt="Banner" width="400"/>
</p>

# 🚨 READ THE ENTIRE README.MD & ALL DOCUMENTS EVERYTHING CAREFULLY !!! 🚨

# NVIDIA AI Aim Assist Tool

An AI-driven aim assist application tailored for NVIDIA GPU users, providing enhanced responsiveness and customization for an optimized gameplay experience. This tool utilizes YOLO, ENGINE, and Arduino integration to support gamers, particularly those with physical challenges, for a seamless and accurate aiming capability.

## Features
- **Real-Time Detection**: Utilizes ENGINE models with non-max suppression for rapid, reliable object detection.
- **Arduino Integration**: Offers Arduino support for additional control options.
- **Flexible Configuration**: Customizable settings for targeting preferences, visuals, and integration options.
- **Resource Management**: Employs garbage collection to optimize memory and system performance.

## Requirements
- **Python Libraries**: 
  - `torch`, `numpy`, `opencv-python`, `psutil`, `colorama`, `pyautogui`, `PyGetWindow`, `yaml`, `tqdm`, `matplotlib`, `seaborn`, `requests`, `dill`, `onnx`, `onnxruntime`, `ultralytics`, `bettercam`, `dxcam`, `serial` (for Arduino), `cupy-cuda11x`.
- **Hardware**: 
  - NVIDIA GPU for maximum efficiency.
  - Optional: Arduino device (e.g., UNO) if `arduino_support` is enabled.
  - **Python 3.11.6**
 
## Discord
- If you have issues join the discord head to channel `nvidia-ai-assist-tool`
- Ping `Bubbles` or `Urugly`
[Invite Link](https://discord.gg/6KZGbJ8MJv)

## Disclaimer

> **This tool and server are provided for educational purposes only and are not intended to promote cheating or unethical behavior in any way.**  
> We are an **educational nonprofit organization** dedicated to assisting individuals with disabilities, PTSD, mental health challenges, and physical limitations. Our community's mission is to support disabled gamers, veterans, streamers, and developers, fostering an environment where everyone can participate fully and equally in gaming and technology.

**Our Mission:**  
Our nonprofit, **fnbubbles420.org**, is dedicated to creating a **supportive, inclusive, and empowering community** for those facing physical and cognitive challenges. We advocate for **accessibility and inclusivity** in the gaming industry, combat stigmas surrounding disabilities, and promote mental wellness. Through proactive involvement, educational resources, and a dedicated support network, we aim to make the world a better place by enhancing the lives of individuals in need.

---

### Important Notice
> **If you choose to use this tool within any online game, please be aware that doing so may violate the terms of service of those games. If you are banned or penalized as a result of using this tool, we are not responsible.**  
> By using this tool, you acknowledge and accept full responsibility for any actions taken with it and understand the potential risks involved.


# LICENSE

- ## ***This project is proprietary and all rights are reserved by the author.***
- ## ***Unauthorized copying, distribution, or modification of this project is strictly prohibited.***
- ## ***Unless You have written permission from the Developer or the FNBUBBLES420 ORG.***


---


```
NVIDIA-AI-Aim-Assist-Tool/
├── .github/                          # GitHub-specific files
│   └── dependabot.yml                # Dependabot configuration for dependency updates
├── Environmental_Setup/              # Environmental setup and configuration files
│   └── ...                           # Files related to environment setup (details not provided)
├── main_tensorrt_script/             # Main script directory for TensorRT
│   └── dist/                         # Distribution folder with obfuscated runtime and models
│       ├── imgs/                     # Image assets
│       ├── models/                   # Model files
│       ├── pyarmor_runtime_000000    # PyArmor runtime obfuscation file
│       ├── pyarmor_runtime_000001    # PyArmor runtime obfuscation file
│       ├── ultralytics1/utils/       # Ultralytics utility files
│       ├── utils/                    # Additional utility scripts
│       ├── New Text Document.txt     # Placeholder text document
│       ├── config.py                 # Configuration file with customizable settings
│       ├── export-command.txt        # Export command for model or setup
│       ├── export.py                 # Script for exporting models
│       ├── gameSelection.py          # Script for selecting and activating the game window
│       ├── main_tensorrt.py          # Main script using TensorRT for AI aim assist
│       └── readme.md                 # README for dist folder specifics
├── CODE_OF_CONDUCT.md                # Code of Conduct for community guidelines
├── LICENSE.md                        # License file for the project
├── SECURITY.md                       # Security policy for reporting vulnerabilities
├── basicv5s.pt                       # Core PyTorch model file
├── get_device.py                     # Script to check device compatibility and GPU setup
├── .gitattributes                    # Git attributes for handling line endings, etc.
├── .gitignore                        # Git ignore file to exclude unnecessary files from the repo
├── install_pytorch.bat               # Batch file to install PyTorch
├── readme.md                         # Main README with project overview and setup instructions
├── requirements.txt                  # Python dependencies for the project
├── run.bat                           # Batch script to execute the project
├── update_ultralytics.bat            # Batch script to update the Ultralytics library
└── v5.pt                             # Additional model file for TensorRT
```


### Installation
1. **Install Dependencies**:
```
pip install torch numpy opencv-python psutil colorama pyautogui PyGetWindow yaml tqdm matplotlib seaborn requests dill onnx onnxruntime ultralytics bettercam dxcam serial cupy-cuda11x
```

### Configuration Settings

Edit the `config.py` file to adjust the settings to match your setup and preferences.

### Configuration Settings

| Setting          | Default   | Description                                           |
|------------------|-----------|-------------------------------------------------------|
| screenShotHeight | 320       | Height of the screenshot for screen capture           |
| screenShotWidth  | 320       | Width of the screenshot for screen capture            |
| useMask          | False     | Enable/Disable mask application on image frames       |
| maskSide         | left      | Side of the mask, choose "left" or "right"            |
| maskWidth        | 80        | Width of the mask                                     |
| maskHeight       | 200       | Height of the mask                                    |
| aaMovementAmp    | 0.4       | Amplifier for mouse movement adjustments              |
| confidence       | 0.4       | Confidence threshold for detection                    |
| aaQuitKey        | 8         | Key to quit the application                           |
| aaActivateKey    | CapsLock  | Key to activate aim assist                            |
| headshot_mode    | True      | Target headshots specifically                         |
| cpsDisplay       | False     | Display clicks per second (CPS)                       |
| visuals          | True      | Display real-time visuals during processing           |
| centerOfScreen   | True      | Focus targeting towards the center of the screen      |
| onnxChoice       | 3         | ONNX model version selection                          |
| model_path       | v5.engine | Path to the model file                                |
| device           | cuda      | Device for processing (set as 'cuda' for GPU usage)   |
| fp16             | True      | Enable FP16 (half precision) for faster computation on supported hardware |
| arduino_support  | True      | Enable/Disable Arduino control for mouse movement     |
| arduino_port     | COM3      | Serial port for Arduino                               |
| arduino_baudrate | 9600      | Baud rate for Arduino communication                   |

### Running the Program

1.**Start the Application**:

```
python main_tensorrt.py
```

2.**Configuration**: Adjust configuration in `config.py` before running for custom experience.

3.**Controls**:

- Press `CapsLock` to activate the aim assist.

- Press `8` to quit the program.

### Usage Examples

- **Model Loading**:

```
model = DetectMultiBackend(config.model_path, device=torch.device(config.device), dnn=False, fp16=config.fp16)
```

- **Arduino Integration**: If `arduino_support` is enabled, Arduino will assist in mouse movement based on detection results.

### Debugging and Visuals

The application includes a visualization mode (`visuals`=True in config) that displays real-time detection boxes, confidence percentages, and target tracking.

**Note**: Setting `cpsDisplay=True` will output the clicks per second (CPS) to monitor performance.

---
---

# 🚀 NVIDIA CUDA Installation Guide

---

## DO **EVERY STEP AND FOLLOW EVERY STEP** OF THE NVIDIA INSTALLATION GUIDE OR IT WON'T WORK PROPERLY

---

### 1. **Download the NVIDIA CUDA Toolkit 11.8**

First, download the CUDA Toolkit 11.8 from the official NVIDIA website:

👉 [Nvidia CUDA Toolkit 11.8 - DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. **Install the CUDA Toolkit**

- After downloading, open the installer (`.exe`) and follow the instructions provided by the installer.
- Make sure to select the following components during installation:
  - CUDA Toolkit
  - CUDA Samples
  - CUDA Documentation (optional)

### 3. **Verify the Installation**

- After the installation completes, open the `cmd.exe` terminal and run the following command to ensure that CUDA has been installed correctly:
    ```
    nvcc --version
    ```
This should display the installed CUDA version.

### **4. Install CuPy**
Run the following command in your terminal to install CuPy:
```
pip install cupy-cuda11x
```

5. **CUDNN Installation** 🧩
Download cuDNN (CUDA Deep Neural Network library) from the NVIDIA website:

👉 [Download CUDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). (Requires an NVIDIA account – it's free).

6. **Unzip and Relocate** 📁➡️
- Open the `.zip` cuDNN file and move all the folders/files to the location where the CUDA Toolkit is installed on your machine, typically:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

7. **Get TensorRT 8.6 GA** 🔽

- [Download TensorRT 8.6 GA](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

8. **Unzip and Relocate** 📁➡️

-Open the `.zip` TensorRT file and move all the folders/files to the CUDA Toolkit folder, typically located at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

9. **Python TensorRT Installation** 🎡

Once all the files are copied, run the following command to install TensorRT for Python:

```
pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

🚨 **Note**: If this step doesn’t work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Locate the correct `.whl` file in the `python` folder and replace the path accordingly.

10. **Set Your Environment Variables** 🌎

- Add the following paths to your environment variables under System Path:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

## Setting Up CUDA 11.8 with cuDNN on Windows

Once you have CUDA 11.8 installed and cuDNN properly configured, you need to set up your environment via cmd.exe to ensure that the system uses the correct version of CUDA (especially if multiple CUDA versions are installed).

### Steps to Set Up CUDA 11.8 Using `cmd.exe`

1. Set the CUDA Path in `cmd.exe`
You need to add the CUDA 11.8 binaries to the environment variables in the current `cmd.exe` session. Open `cmd.exe` and run the following commands separately:

```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%
```

These commands add the CUDA 11.8 binary, lib, and CUPTI paths to your system's current session. Adjust the paths as necessary depending on your installation directory.

2. **Verify the CUDA Version**
After setting the paths, verify that your system is using CUDA 11.8 by running:

```
nvcc --version
```

This should display the details of CUDA 11.8. If it shows a different version, check the paths and ensure the proper version is set.

3. Set the Environment Variables for a Persistent Session

If you want to ensure CUDA 11.8 is used every time you open cmd.exe, you can add these paths to your system environment variables permanently:

1. Open `Control Panel` -> `System` -> `Advanced System Settings`.
Click on `Environment Variables`.
Under `System variables`, select `Path` and click `Edit`.
Add the following entries at the top of the list:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
```

This ensures that CUDA 11.8 is prioritized when running CUDA applications, even on systems with multiple CUDA versions.

4. **Set CUDA Environment Variables for cuDNN**

If you're using cuDNN, ensure the `cudnn64_8.dll` is also in your system path:

```
set PATH=C:\tools\cuda\bin;%PATH%
```

This should properly set up CUDA 11.8 to be used for your projects via `cmd.exe`.

### Additional Information

- Ensure that your GPU drivers are up to date.
- Check CUDA compatibility with other software (e.g., PyTorch or TensorFlow) by referring to their documentation for specific versions supported by CUDA 11.8.

### Environmental Variable Setup

![pic](https://github.com/FNBUBBLES420-ORG/NVIDIA-AI-Aim-Assist-Tool/blob/main/Environmental_Setup/pic.png)


```
import torch

print(torch.cuda.is_available())  # This will return True if CUDA is available
print(torch.version.cuda)  # This will print the CUDA version being used
print(torch.cuda.get_device_name(0))  # This will print the name of the GPU, e.g., 'NVIDIA GeForce RTX GPU Model'
```

- Run the `get_device.py` script to check if you installed it correctly.

## 🛠 Run Script run.bat

The `run.bat` script is a batch file to help you install all the required dependencies for this project. Below is the content of the file and the steps it will execute:

```
@echo off
echo Installing ONNX Runtime (GPU)...
pip install onnxruntime-gpu
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing NumPy...
pip install numpy
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing comtypes...
pip install comtypes
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing OpenCV (opencv-python)...
pip install opencv-python
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing pandas...
pip install pandas
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing bettercam...
pip install bettercam
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing onnx...
pip install onnx
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing PyWin32...
pip install pywin32
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing Dill...
pip install dill
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing CuPy (GPU accelerated array library for CUDA 11.8)...
pip install cupy-cuda11x
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing psutil...
pip install psutil
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing colorama...
pip install colorama
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing ultralytics...
pip install ultralytics
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing PyAutoGUI...
pip install PyAutoGUI
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing PyGetWindow...
pip install PyGetWindow
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing pyyaml...
pip install pyyaml
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing tqdm...
pip install tqdm
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing matplotlib...
pip install matplotlib
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing seaborn...
pip install seaborn
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing requests...
pip install requests
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing ipython...
pip install ipython
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing dxcam...
pip install dxcam
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing pyarmor...
pip install pyarmor
echo Press enter to continue with the rest of the dependency installs
pause

echo MAKE SURE TO HAVE THE WHL DOWNLOADED BEFORE YOU CONTINUE!!!
pause
echo Click the link to download the WHL: press ctrl then left click with mouse
echo https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo Installing CuPy from WHL...
pip install https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo All packages installed successfully!
pause
```

## How to Use the run.bat Script

1. **Download the Required Files**:

- Ensure you have downloaded the WHL file for CuPy from the following link: [Download CuPy WHL](https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl)

2. **Run the Batch File**:

- Execute the `run.bat` file to automatically install all necessary Python dependencies for this project.

- The script will pause after each step so you can verify the installation. Simply press any key to continue after each pause.

To execute the batch file, you can use:

```
./run.bat
```
---
---

# 🚀 Visual Studio 2022 Community Edition Installation Guide

This guide will help you download and install **Visual Studio 2022 Community Edition** with the **Desktop Development with C++** workload for C and C++ development.

</div>

## 📥 Step 1: Download Visual Studio

Click the following link to download **Visual Studio 2022 Community Edition**:  
👉 [Download Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)

## 🛠 Step 2: Installing Visual Studio

1. Once the installer is downloaded, **run the installer**.
2. In the **Visual Studio Installer**, select the **Workloads** tab.

## 🖥 Step 3: Select Workload for C++ Development

To set up **C++ development**, ensure you select the **Desktop development with C++** workload:

1. In the **Workloads** tab, check the option **Desktop development with C++**.
   - This will install the necessary tools for C++ programming, including compilers, libraries, and debugging tools.
2. Click **Install** to begin the installation process.

## 🛠 System Requirements Visual Studio 2022

Make sure your system meets the minimum requirements for Visual Studio 2022:
- **OS**: Windows 10 or higher.
- **Processor**: Quad-core processor or better.
- **RAM**: 8 GB of RAM (16 GB recommended).
- **Disk Space**: Minimum 20 GB free space.

## 🛑 Troubleshooting

If you encounter any issues during installation, refer to the official troubleshooting guide:  
- [Visual Studio Installation Troubleshooting](https://docs.microsoft.com/en-us/visualstudio/install/troubleshooting-installation-issues?view=vs-2022)

</div>

## Now you're ready to start developing C and C++ applications in Visual Studio 2022! 🎉
---
---
