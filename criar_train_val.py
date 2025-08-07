import os
import glob
import random
import shutil

# --- CONFIGURAÇÃO ---
CROPS_ROOT  = 'crops'  # pasta que contém subpastas '3', '8', ...
out_base    = '.'      # onde ficarão train/ e valid/
train_ratio = 0.8
# ----------------------

# Descobre dinamicamente as classes pelas pastas dentro de crops/
classes = [d for d in os.listdir(CROPS_ROOT)
           if os.path.isdir(os.path.join(CROPS_ROOT, d))]
print(f'Classes encontradas em {CROPS_ROOT}: {classes}')

# Cria as pastas finais: train/<classe> e valid/<classe>
for split in ['train', 'valid']:
    for c in classes:
        path = os.path.join(out_base, split, c)
        os.makedirs(path, exist_ok=True)

# Para cada classe, faz split e copia
for c in classes:
    src_dir = os.path.join(CROPS_ROOT, c)
    all_imgs = glob.glob(os.path.join(src_dir, '*.jpg'))
    random.shuffle(all_imgs)

    split_idx = int(len(all_imgs) * train_ratio)
    train_imgs = all_imgs[:split_idx]
    valid_imgs = all_imgs[split_idx:]

    # Copia para train
    for img in train_imgs:
        dst = os.path.join(out_base, 'train', c, os.path.basename(img))
        shutil.copy(img, dst)

    # Copia para valid
    for img in valid_imgs:
        dst = os.path.join(out_base, 'valid', c, os.path.basename(img))
        shutil.copy(img, dst)

    print(f'Classe {c}: {len(train_imgs)} em train/{c}, {len(valid_imgs)} em valid/{c}')
