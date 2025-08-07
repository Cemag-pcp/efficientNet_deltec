import os
import cv2
from ultralytics import YOLO

# — CONFIGURAÇÕES —
YOLO_WEIGHTS = r'C:\Users\Luan\workspace\visao_comp\results\placa_detector\weights\best.pt'
SOURCE_ROOTS = {
    'dia':   r'videos\dia',
    'noite': r'videos\noite'
}
OUTPUT_ROOT  = r'crops'
IMG_SIZE     = 640
CONF_THRESH  = 0.80
MAX_CROPS    = 300  # máximo de crops por classe (dentro de cada folder dia/noite)

# Carrega modelo
model = YOLO(YOLO_WEIGHTS)

def process_class_folder(class_folder, suffix):
    """
    Extrai até MAX_CROPS crops de todos os vídeos dentro de class_folder,
    salvando em crops/<classe>/ e adicionando _<suffix> no nome dos arquivos.
    """
    label = os.path.basename(class_folder)
    out_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    vids = [f for f in os.listdir(class_folder)
            if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    for vid in vids:
        vid_path = os.path.join(class_folder, vid)
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f'⚠️  Não foi possível abrir {vid_path}')
            continue

        while count < MAX_CROPS:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH)[0]
            for box in results.boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # salva crop com sufixo _dia ou _noite
                fname = f'{label}_{count:03d}_{suffix}.jpg'
                cv2.imwrite(os.path.join(out_dir, fname), crop)
                count += 1
                if count >= MAX_CROPS:
                    break

        cap.release()
        print(f'✅ {label} ({suffix}): {count}/{MAX_CROPS} crops extraídos de {vid_path}')
        if count >= MAX_CROPS:
            break

    if count == 0:
        print(f'⚠️  Nenhuma detecção para a classe {label} em {suffix}.')
    else:
        print(f'🎯 Finalizado {label} ({suffix}): {count} crops gerados em {out_dir}')

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for suffix, root in SOURCE_ROOTS.items():
        print(f'\n🔄 Processando pasta "{suffix}" em {root}...')
        for entry in sorted(os.listdir(root), key=lambda x: int(x)):
            class_folder = os.path.join(root, entry)
            if os.path.isdir(class_folder):
                process_class_folder(class_folder, suffix)

if __name__ == '__main__':
    main()
