import os
import cv2
import csv
import torch
import time
from pathlib import Path

from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from datetime import datetime, timedelta
from collections import deque
from api_apontamento import finalizar_cambao
     
# base do projeto (onde está este script)
BASE_DIR = Path(__file__).parent

# monta caminho relativo até o best.pt
YOLO_WEIGHTS = BASE_DIR / 'results' / 'placa_detector' / 'weights' / 'best.pt'

# — CONFIGURAÇÕES —
YOLO_WEIGHTS = str(YOLO_WEIGHTS)
EFF_WEIGHTS  = 'eff3vs8.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE     = 640
CONF_THRESH  = 0.75
CLF_THRESH  = 0.75

# URL da sua câmera IP (RTSP)
VIDEO_PATH = 'rtsp://admin:cem@2022@192.168.3.208:554/cam/realmonitor?channel=1&subtype=0'

num_classes = 8
classes = [str(i) for i in range(1, 9)]

# Região permitida
ROI_X1, ROI_Y1 = 600, 250
ROI_X2, ROI_Y2 = 900, 400

ALERT_COOLDOWN = timedelta(minutes=30)
last_alert = {c: datetime.min for c in classes}
ALERT_CSV = 'alerts.csv'

# --- Parâmetros de suavização ---
FRAME_THRESHOLD = 3      # mesmo dígito em 3 frames seguidos
recent_preds    = deque(maxlen=FRAME_THRESHOLD)
stable_digit    = None
last_alert_time = datetime.min
# ----------------------------------

# Caminho do CSV de alertas
if not os.path.exists(ALERT_CSV):
    with open(ALERT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp','digit','yolo_conf','clf_conf'])

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
    global stable_digit, last_alert_time

    cap = conectar_camera(VIDEO_PATH)
    if cap is None:
        print("Não foi possível conectar à câmera.")
        return

    # 1) Cria a janela antes do loop
    window = 'Detecções'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        
        grabbed = cap.grab()
        if not grabbed:
            break
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = conectar_camera(VIDEO_PATH)
            if cap is None: break
            continue

        # inferência YOLO
        results = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THRESH,
                       verbose=False, show=False)[0]
        boxes     = results.boxes.xyxy.cpu().numpy()
        yolo_confs= results.boxes.conf.cpu().numpy()

        # 1) Desenha o ROI definido
        cv2.rectangle(
            frame,
            (ROI_X1, ROI_Y1),
            (ROI_X2, ROI_Y2),
            (255, 0, 0),   # cor azul
            2              # espessura da borda
        )

        # para cada caixa dentro da ROI
        for (x1,y1,x2,y2), yolo_conf in zip(boxes, yolo_confs):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            if x1<ROI_X1 or y1<ROI_Y1 or x2>ROI_X2 or y2>ROI_Y2:
                continue

            # prepara crop e classifica
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = tfm(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = clf(inp)
                pred   = logits.argmax(dim=1).item()
                clf_conf = torch.softmax(logits,1)[0,pred].item()

            pred_digit = classes[pred]
            # 1) acumula no deque
            recent_preds.append(pred_digit)

            # 2) verifica se o deque encheu e todos são iguais
            if len(recent_preds)==FRAME_THRESHOLD and len(set(recent_preds))==1:
                candidate = recent_preds[0]
            else:
                candidate = None

            # 3) só dispara quando mudar para um novo dígito estável
            now = datetime.now()
            if candidate and candidate != stable_digit:
                # 3.1) Filtra pelas confianças
                if yolo_conf >= CONF_THRESH and clf_conf >= CLF_THRESH:
                    # 3.2) Verifica o cooldown
                    if now - last_alert_time >= ALERT_COOLDOWN:
                        stable_digit    = candidate
                        last_alert_time = now

                        # registra CSV e chama API
                        with open(ALERT_CSV, 'a', newline='') as f:
                            csv.writer(f).writerow([
                                now.strftime('%Y-%m-%d %H:%M:%S'),
                                stable_digit,
                                f'{yolo_conf:.4f}',
                                f'{clf_conf:.4f}'
                            ])
                        print(f"✅ Alerta: dígito {stable_digit} | "
                            f"YOLO {yolo_conf:.0%}, clf {clf_conf:.0%} em {now:%H:%M:%S}")

                        # chama API
                        finalizar_cambao(stable_digit)
            
            # cv2.imshow('Detecções', frame)

            # desenha no frame o dígito estável, não o ruído momentâneo
            if stable_digit:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                            f'{stable_digit} ({int(yolo_conf*100)}%)',
                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,255,0),2)
        
        # 2) Exibe o frame inteiro **depois** de desenhar todas as deteções
        # cv2.imshow(window, frame)
        
        # exibe (ou salta se não quiser janela)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realtime_infer()
