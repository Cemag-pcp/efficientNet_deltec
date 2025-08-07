import os
import cv2
import csv
import torch
import time
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from datetime import datetime, timedelta
from api_apontamento import finalizar_cambao
            
# — CONFIGURAÇÕES —
YOLO_WEIGHTS = r'C:\Users\TI DEV\effn_deltec\deltec\results\placa_detector\weights\best.pt'
EFF_WEIGHTS  = 'eff3vs8.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE     = 640
CONF_THRESH  = 0.6

# URL da sua câmera IP (RTSP)
VIDEO_PATH = 'rtsp://admin:cem@2022@192.168.3.208:554/cam/realmonitor?channel=1&subtype=0'
# VIDEO_PATH = r'videos_teste/7_0708.mp4'

num_classes = 8
classes = [str(i) for i in range(1, 9)]

# Região permitida
ROI_X1, ROI_Y1 = 600, 250
ROI_X2, ROI_Y2 = 900, 400

# Cooldown de 30 minutos
ALERT_COOLDOWN = timedelta(minutes=30)
last_alert = {c: datetime.min for c in classes}

# Caminho do CSV de alertas
ALERT_CSV = 'alerts.csv'
if not os.path.exists(ALERT_CSV):
    with open(ALERT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'digit', 'yolo_conf', 'clf_conf'])

# Carrega modelos
yolo = YOLO(YOLO_WEIGHTS)
clf = models.efficientnet_b0(pretrained=False)
clf.classifier[1] = torch.nn.Linear(clf.classifier[1].in_features, num_classes)
clf.load_state_dict(torch.load(EFF_WEIGHTS, map_location=DEVICE))
clf.to(DEVICE).eval()

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def conectar_camera(url, tentativas=100, delay=5):
    """ Tenta abrir o stream RTSP várias vezes antes de desistir. """
    for i in range(tentativas):
        print(f"?? Tentando conectar à câmera (tentativa {i+1}/{tentativas})...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)
        if cap.isOpened():
            print("? Conectado à câmera.")
            return cap
        print(f"? Falha na conexão. Re-tentando em {delay}s...")
        cap.release()
        time.sleep(delay)
    print("?? Não foi possível conectar à câmera.")
    return None

def realtime_infer():
    # abre o stream da câmera
    cap = conectar_camera(VIDEO_PATH)
    if cap is None:
        return

    window = 'Inferência - Câmera IP'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("? Falha ao ler frame. Tentando reconectar...")
            cap.release()
            cap = conectar_camera(VIDEO_PATH)
            if cap is None:
                break
            continue

        # desenha ROI
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255,0,0), 2)

        # inferência YOLO no frame inteiro
        results = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False, show=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        yolo_confs = results.boxes.conf.cpu().numpy()

        for (x1,y1,x2,y2), yolo_conf in zip(boxes, yolo_confs):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            # só dentro da ROI
            if x1 < ROI_X1 or y1 < ROI_Y1 or x2 > ROI_X2 or y2 > ROI_Y2:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = tfm(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = clf(inp)
                pred = logits.argmax(dim=1).item()
                clf_conf = torch.softmax(logits, dim=1)[0, pred].item()

            digit = classes[pred]
            now = datetime.now()

            # checa cooldown e registra alerta
            if now - last_alert[digit] >= ALERT_COOLDOWN:
                last_alert[digit] = now
                with open(ALERT_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now.strftime('%Y-%m-%d %H:%M:%S'),
                        digit,
                        f'{yolo_conf:.4f}',
                        f'{clf_conf:.4f}'
                    ])
                print(f"?? Alerta: dígito {digit} com YOLO {yolo_conf:.2%} e "
                      f"classif. {clf_conf:.2%} registrado em {now.strftime('%H:%M:%S')}")
                
                # chamar a api de apontamento
                finalizar_cambao(digit)
                
            # desenha resultado
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label_txt = f'{digit} ({int(yolo_conf*100)}%)'
            cv2.putText(frame, label_txt, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realtime_infer()
