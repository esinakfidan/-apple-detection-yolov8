import cv2
from ultralytics import YOLO

#  yükleme
model = YOLO("C:/Users/esina/OneDrive/Masaüstü/Yolov8Project/runs/detect/train3/weights/best.pt") 

# Kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılmadı!")
    exit()

while True:
    # bir kare 
    ret, frame = cap.read()
    if not ret:
        print("Kamera hatası, kare alınamadı.")
        break

    # tahmin 
    results = model(frame)  # Görüntüden tahmin

    # Sonuçlar
    for result in results:
       
        frame_with_boxes = result.plot()  

    # Sonuçlar
    cv2.imshow("Nesne Tanıma", frame_with_boxes)

    #  döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
