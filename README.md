![](https://img.shields.io/badge/Creater-TCT-FFFF00) ![](https://img.shields.io/badge/development-python-006400) ![](https://img.shields.io/badge/Version-3.9.18-blue)

# FaceRecognition
FaceRecognition/人臉辨識

# 1. Package Introduce

## 1-1. OpenCV (cv2)
（Open Source Computer Vision Library）是一個開源的計算機視覺和機器學習軟件庫，旨在提供各種視覺任務的高效解決方案。它被廣泛應用於影像處理、物體檢測、影像分類、面部識別、計算機視覺等領域。

主要功能：

- 基本處理：旋轉、縮放、裁剪、平移、翻轉、顏色空間轉換（例如：BGR ↔ RGB、灰階、HSV 等）。

- 濾波：使用濾波器（如高斯濾波、邊緣檢測濾波等）來進行降噪或邊緣檢測。

- 圖像增強：調整對比度、亮度、色調，進行直方圖均衡化等。


### 物體檢測與追蹤：

- Haar 特徵分類器：用於面部識別、人臉檢測。

- HOG（Histogram of Oriented Gradients）：用於人類檢測。

- SSD（Single Shot Multibox Detector）/YOLO（You Only Look Once）：用於實時物體檢測。

- 物體追蹤：追蹤目標物體，例如使用 Meanshift 和 Camshift 進行物體跟踪。

### 特徵檢測與匹配：

- 角點檢測：如 Harris 角點檢測。
- SIFT/SURF：尺度不變特徵轉換，用於特徵檢測和匹配。
- ORB（Oriented FAST and Rotated BRIEF）：用於更快的特徵匹配。

- 機器學習：
支持向量機（SVM）：用於分類。
神經網絡：OpenCV 提供了深度學習的接口，可以使用預訓練的模型，如 Caffe、TensorFlow 等。

- 視頻處理：
讀取視頻：通過 cv2.VideoCapture 讀取攝像頭或視頻檔案。

  視頻錄製：用 cv2.VideoWriter 寫入視頻，支援多種格式（如 .avi、.mp4）。
  
  包括解析度調整、幀率修改、視頻剪輯等。

- 圖像變換：
透視變換：進行視角變換、圖像扭曲。

  圖像分割：將圖像劃分為多個區域。

- 計算機視覺任務：
文字識別（OCR）：透過 tesseract 等工具進行文本識別。

  手勢識別、人臉識別、姿態估計 等。

- 跨平台：
支援多種操作系統，如 Windows、Linux、macOS，並提供多種語言接口，最常用的是 C++ 和 Python，但也有 Java 和其他語言支持。

與硬體的兼容性好：支援多種影像擷取裝置，包括 WebCAM、USB 攝像頭、專業相機等，並能高效處理來自不同來源的視頻流。

易於集成與擴展：易於與其他機器學習庫（如 TensorFlow、PyTorch）集成，也能與硬體設備、網路接口等進行整合。

------

## 1-2.  dlib
dlib 是一個強大的機器學習庫，主要用於人臉識別、物體檢測、圖像處理和其他與計算機視覺有關的任務。它提供了多種功能來進行物體偵測、人臉識別、特徵點檢測等任務。

### 優點:
- ✅ 高效準確：
dlib 提供的臉部偵測和特徵點檢測工具，能夠在大多數情況下進行高效且準確的偵測。

- ✅ 簡單易用：
提供簡單的 API，使得開發者能夠快速進行人臉檢測和特徵點提取。
  
- ✅ 跨平台支持：
dlib 可在多種操作系統（如 Windows、Linux、macOS）上運行。

- ✅ 開源：
dlib 是一個開源庫，這意味著它可以免費使用並且可修改。

------

# 2. py Code
## 2-1. Input FacePicture to facedb folder
  1.請準備您想辨識的人臉照片，注意:圖片內僅能1個人，正面，臉孔清晰
  
  Please prepare the photo of the face you want to recognize. Note: The image should contain only one person, be taken from the front, and the face should be clear.
  
  
  1-1.圖片格式最好使用 jpeg
  
  It is recommended to use the JPEG image format.
        
  ![image](https://github.com/user-attachments/assets/6f5ee9ef-8bb5-4539-ae15-595bcc9c4f70)
  
 2.請使用 VSCode 或您的開發工具開啟 FaceRecognition.py

 Please open FaceRecognition.py with VSCode or any development tool.
 

3.程式核心原理
  3-1.會將 facedb 資料夾內的圖片，檢測出人臉特徵並依序存入陣列內並用圖片檔名作為人名。
      
  It will detect the facial features of the images in the 'facedb' folder, store them sequentially in an array, and use the image filenames as the names.
  
      
  3-2.開啟鏡頭，並將鏡頭上的畫面執行檢測人臉是否符合陣列內人臉特徵內的人物，若有變繪製文字呈現名稱；反之便顯示未知名稱。
      
  Turn on the camera and perform face detection on the live feed to check if the face matches any person in the array of facial features. If a match is found, display the name; otherwise, display 'Unknown'.
  

  3-3.在視訊鏡頭畫面，按下 "ESC" 會退出程式；按下 "1" 會將視訊畫面截圖儲存至 data 資料夾。
  
  In the video feed, press 'ESC' to exit the program; press '1' to capture and save a screenshot of the video to the 'data' folder.
      
![image](https://github.com/user-attachments/assets/17b3a2a1-5108-4c2c-ba5a-d5934b24d2c7)


  3-4.請參考以下識別結果:
  
  Please refer the recognition result as below.
  
![image](https://github.com/user-attachments/assets/2f12d504-5c41-4112-befa-746ecc7d03b7)

![image](https://github.com/user-attachments/assets/019f4a06-4e1a-47a6-8215-e705eaa74ce1)

------

# 3. Note

```diff
! 本程式所使用的女生模特兒皆為我使用SD繪製出來的虛擬人物。
! The girl models used in this program are all virtual characters created by me using SD (Stable Diffusion).
```

------

## About Me
Thanks & Best Regards !

蔡承廷

​Senior Engineer of Semiconductor Product/Testing & ​Automation

Email: ​​kp924606@gmail.com

LinkedIn:https://www.linkedin.comin/tsai-cheng-ting/
