# libs básicas
import os
import cv2
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta

# libs importantes pro modelo
import torch
from ultralytics import YOLO
from torchvision import transforms, models
from collections import deque
from PIL import Image

# libs de apontamento (API do CEMAGPROD)
from api_apontamento import finalizar_cambao
     
# base do proj
BASE_DIR = Path(__file__).parent

# caminho independente do pc
# caminho dos pesos do modelo YOLO
YOLO_WEIGHTS = BASE_DIR / 'results' / 'placa_detector' / 'weights' / 'best.pt'

# URL da câmera IP (RTSP)
VIDEO_PATH = 'rtsp://admin:cem@2022@192.168.3.208:554/cam/realmonitor?channel=1&subtype=0'

# Apenas para testes
# VIDEO_PATH = '1.mp4'

# Já que as classes são números de 1 a 8, foi feito uma lista que traz o range de 1 a 8
num_classes = 8
classes = [str(i) for i in range(1, 9)]

# Região de interesse para o modelo (evita mostrar o frame inteiro)
ROI_X1, ROI_Y1 = 600, 250
ROI_X2, ROI_Y2 = 900, 400

# Configuração de cooldown
# Serve para não alertar mais de uma vez o mesmo número de forma sequenciada
# Exemplo: Alertou o número 4 (entra o cooldown de no mínimo 30 min para o próximo número 4)
ALERT_COOLDOWN = timedelta(minutes=30)

# Log para alertas (apenas para guardar a hora que foi notificado a passagem daquela classe prevista)
last_alert = {c: datetime.min for c in classes}
ALERT_CSV = 'alerts.csv'

# --- Configuração geral ---
YOLO_WEIGHTS = str(YOLO_WEIGHTS) # caminho/identificador dos pesos do modelo YOLO.
EFF_WEIGHTS  = 'eff3vs8.pth' # nome do arquivo de pesos do modelo EfficientNet
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu' # seleciona automaticamente GPU (CUDA) se disponível, senão CPU.
IMG_SIZE     = 640 # tamanho da imagem de entrada para o modelo (em pixels) (quanto menor menos qualidade na imagem, isso é ruim para o modelo treinado com 640).
CONF_THRESH  = 0.75 # valor mínimo de confiança para o modelo YOLO
CLF_THRESH  = 0.75 # valor mínimo de confiança para o modelo efficientNet

# --- Parâmetros de suavização ---
FRAME_THRESHOLD = 3      # mesmo dígito em 3 frames seguidos (apenas se bater 75% de confiança em 3 frames consecutivos (confirmará a classe))
recent_preds    = deque(maxlen=FRAME_THRESHOLD) # fila circular que armazena as últimas predições (até o limite definido por FRAME_THRESHOLD).
stable_digit    = None # último dígito confirmado como estável.
last_alert_time = datetime.min # registro de quando o último alerta foi emitido (inicializado no menor valor possível de datetime).
# ----------------------------------

# Caminho do CSV de alertas
if not os.path.exists(ALERT_CSV):
    with open(ALERT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp','digit','yolo_conf','clf_conf'])

# Carrega modelos
# YOLO
yolo = YOLO(YOLO_WEIGHTS)

# EfficientNet
clf = models.efficientnet_b0(pretrained=False)
clf.classifier[1] = torch.nn.Linear(clf.classifier[1].in_features, num_classes)
clf.load_state_dict(torch.load(EFF_WEIGHTS, map_location=DEVICE))
clf.to(DEVICE).eval()

# Pipeline de transformação das imagens antes de passar pelo modelo classificador
tfm = transforms.Compose([
    transforms.Resize((224,224)), # redimensiona para 224x224px (padrão do effNet)
    transforms.ToTensor(), # converte a imagem para tensor (PyTorch) normalizado entre 0 e 1. 
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # aplica normalização de cada canal (R, G, B) usando médias e desvios padrão do ImageNet.
])

def conectar_camera(url, tentativas=100, delay=5):
    """
    Tenta abrir o stream RTSP várias vezes antes de desistir.
    Isso ajuda caso perca a conexão (tendo em vista que a conexão da CEMAG não é tão estável)

    """
    for i in range(tentativas):
        print(f"?? Tentando conectar à câmera (tentativa {i+1}/{tentativas})...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)
        if cap.isOpened():
            print("Conectado à câmera.")
            return cap
        print(f"Falha na conexão. Re-tentando em {delay}s...")
        cap.release()
        time.sleep(delay)
    print("Não foi possível conectar à câmera.")
    return None

def realtime_infer(camera_tempo_real):
    global stable_digit, last_alert_time

    cap = conectar_camera(VIDEO_PATH)
    if cap is None:
        print("Não foi possível conectar à câmera.")
        return

    # Cria a janela antes do loop
    window = 'Detecções'
    # cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_time = time.time()  # timestamp do frame anterior

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

        # cálculo do FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # desenha o FPS no canto da tela
        cv2.putText(frame,
                    f'FPS: {fps:.1f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        # inferência YOLO
        results = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THRESH,
                       verbose=False, show=False)[0]
        boxes     = results.boxes.xyxy.cpu().numpy()
        yolo_confs= results.boxes.conf.cpu().numpy()

        # Desenha o ROI definido
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
            # acumula no deque
            recent_preds.append(pred_digit)

            # verifica se o deque encheu e todos são iguais
            if len(recent_preds)==FRAME_THRESHOLD and len(set(recent_preds))==1:
                candidate = recent_preds[0]
            else:
                candidate = None

            # só dispara quando mudar para um novo dígito estável
            now = datetime.now()
            if candidate and candidate != stable_digit:
                # Filtra pelas confianças
                if yolo_conf >= CONF_THRESH and clf_conf >= CLF_THRESH:
                    # Verifica o cooldown
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
                        # finalizar_cambao(stable_digit)
            
            # cv2.imshow('Detecções', frame)

            # desenha no frame o dígito estável, não o ruído momentâneo
            if stable_digit:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                            f'{stable_digit} ({int(yolo_conf*100)}%)',
                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,255,0),2)
        
        # Exibe o frame inteiro **depois** de desenhar todas as deteções
        # Perguntar ao usuario se quer ou não visulizar em tempo real a imagem da câmera (apenas para validações)
        if video_tempo_real.strip().lower() == 'y':
            cv2.imshow(window, frame)
        
        # exibe (ou salta se não quiser janela)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        video_tempo_real = input("Visualizar em tempo real a imagem da câmera? (Y/N): ").strip().lower()
        if video_tempo_real in ('y', 'n'):
            break
        print("Digite apenas Y ou N.")
    
    realtime_infer(video_tempo_real)