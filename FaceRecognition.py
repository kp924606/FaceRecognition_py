#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.

import os,dlib,numpy,cv2
import time
from datetime import datetime

# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 比對人臉圖片資料夾名稱
faces_folder_path = "facedb"

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
predictor = dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
FaceName = []

#差異度，建議0~0.5之間，越小越嚴格
difference = 0.5
# 讀取資料庫裡每張圖片先計算臉部特徵數據:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得特徵數據

for file in os.listdir(faces_folder_path):
    base = os.path.basename(file)
    # 依序取得圖片檔案為人名
    FaceName.append(file.replace('.jpg','').replace('.JPG',''))
    #讀取檔案
    img = cv2.imread(faces_folder_path + '/' + file)

    # 1.人臉偵測
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        # 2.特徵點偵測
        shape = predictor(img, d)
 
        # 3.取得128維特徵數據
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        # 轉換numpy array格式
        v = numpy.array(face_descriptor)
        descriptors.append(v)

# 針對需要辨識的人臉同樣進行處理
cap = cv2.VideoCapture(0)
d_test = None #存放臉部特徵差異值

while cap.isOpened():
    ret, img = cap.read()    
    img=cv2.flip(img,1) #轉鏡像
    stime=time.time()
    dets = detector(img, 0) #在畫面中尋找臉部
      
    for k, d in enumerate(dets):#至少找到一個臉部
        shape = predictor(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)
        #臉部座標範圍
        x1,y1,x2,y2 = d.left(),d.top(),d.right(), d.bottom()

        # 以方框標示偵測的人臉
        cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 0), 4)
       
        dist = []
        # 計算'照片中臉部'與'資料庫中臉部'的差異
        for i in descriptors:
            dist_ = numpy.linalg.norm(i -d_test) #距離 norm=(x^2+x^2)^(0.5)
            dist.append(dist_)

        # 將比對人名和比對出來的歐式距離組成一個dict
        candidate_dist = dict(zip(FaceName,dist)) #zip鏈接1維list(candidate,dist)成2維list
        rec_name = min(candidate_dist, key = candidate_dist.get) #只取差異最小值的人                    
        if candidate_dist[rec_name]>difference:
            rec_name="unknown"
        # 將辨識出的人名印到圖片上面
        cv2.putText(img, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2)                
        etime=time.time()
        fps=round(1/(etime-stime),2)
        cv2.putText(img, "FPS:" + str(fps), (10, 50), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 0, 0, 255), 2)
        
    cv2.imshow( "Face Recognition", img)
    
    #ESC鍵結束程式
    key = cv2.waitKey(1)
    if key == 27:
        break
    # 按1拍照存檔
    elif key == 49:
        # 獲取當前時間
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # 設定檔案名稱
        filename = fr"data\photo_{timestamp}.jpg"
        #儲存照片
        cv2.imwrite(filename , img)

cv2.destroyAllWindows()
