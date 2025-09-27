"""
This module contains functions for training YOLOv12 models.
"""

import os
import yaml
import argparse
import torch
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm
from tqdm.notebook import tqdm # Se estiver usando tqdm no notebook


# ====== Create Project Structure ======
def create_project_structure(base_dir='yolov12'):
    """
    Creates the project structure needed for YOLOv12 training.
    """
    directories = [
        os.path.join(base_dir, 'configs'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'runs'),
        os.path.join(base_dir, 'weights')
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory ensured: {dir_path}")

# ====== Directory Validation ======
def check_directories(train_dir, val_dir, test_dir):
    """
    Check if the given training, validation, and testing directories exist and contain images.
    """
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise Exception(f"Directory not found: {dir_path}")

        # Check for image files in all subdirectories
        has_images = any(
            file.endswith(('.jpg', '.png')) 
            for root, _, files in os.walk(dir_path) 
            for file in files
        )

        if not has_images:
            raise Exception(f"No image files found in: {dir_path}")
        
        print(f"Directory verified: {dir_path}")

# ====== YAML Configuration ======
def create_data_yaml(train_img, val_img, test_img, output_path, class_names=('words',)):

    data_yaml_content = {
        'train': os.path.abspath(train_img),
        'val':   os.path.abspath(val_img),
        'test':  os.path.abspath(test_img),
        'nc': len(class_names),
        'names': list(class_names)
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    # Display data.yaml content
    print(f"data.yaml created at:", {output_path})
    print("Content of data.yaml:\n", data_yaml_content)

def create_data_yaml_mixed(train_list, val_list, test_list, output_path, class_names=('text',)):
    """
    Aceita LISTAS de diretórios para treinar/validar/testar (ex.: Mapillary + SVT).
    """
    def _norm(x):
        return [os.path.abspath(p) for p in x]
    data_yaml_content = {
        'train': _norm(train_list),
        'val':   _norm(val_list),
        'test':  _norm(test_list),
        'nc': len(class_names),
        'names': list(class_names)
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)
    print("data.yaml (mixed):", data_yaml_content)

# === TRAINING ===
def train_yolov12(
        data_yaml,
        img_size=640,
        batch_size=8,
        epochs=200,
        weights='yolo12x.pt',
        device='0',
        workers=4,
        cache='false',
        optimizer='AdamW',
        lr0=5e-4,
        lrf=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        patience=20,
        seed = 42,
        project='yolov12/runs/',
        name='exp',

        # Augmentations seguras para texto
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.0, fliplr=0.5,
        translate=0.08, scale=0.2, shear=0.0,
        perspective=0.0,
        mosaic=0.1, mixup=0.0, auto_augment='none',
        close_mosaic=10
    ):
    
    """
    Treino YOLO (v11/v12 – API Ultralytics) com aug conservadoras para texto.
    """
    os.makedirs(project, exist_ok=True)
    model = YOLO(weights)  # Load pre-trained YOLOv12 model

    print(f"[train] weights={weights} | device={device} | imgsz={img_size} | batch={batch_size}")
    
    results = model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        device=device,
        workers=workers,
        cache=cache,
        project=project,
        name=name,
        seed=seed,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        patience=patience,
        plots=True,

        # augmentations:
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        flipud=flipud, fliplr=fliplr,
        translate=translate, scale=scale, shear=shear,
        perspective=perspective,
        mosaic=mosaic, mixup=mixup, close_mosaic=close_mosaic,
        auto_augment=auto_augment,
        val=True
    )

    # ====== logging consistente ======
    log_path = os.path.join(project, 'logs', 'training_log.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    md = getattr(results, "results_dict", {}) or {}
    with open(log_path, 'a') as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] run={name}\n")
        # Tente várias chaves (mudam entre versões):
        keys_candidates = [
            'metrics/mAP50(B)', 'metrics/mAP50', 'metrics/mAP_0.5',
            'metrics/precision(B)', 'metrics/precision',
            'metrics/recall(B)', 'metrics/recall',
            'fitness', 'epoch', 'lr/pg0'
        ]
        for k in keys_candidates:
            if k in md:
                try:
                    f.write(f"{k}: {float(md[k]):.5f}\n")
                except Exception:
                    f.write(f"{k}: {md[k]}\n")
    print(f"[train] log → {log_path}")

    return model, results

# === PREDICTION ===

def predict_yolov12(
        weights, source_images_dir,
        img_size=640, conf_thres=0.25, iou_thres=0.45, 
        device='0', workers=4,
        save_txt=True, save_img=True, save_crop=True,
        project='yolov12/runs', name='pred',
        chunk_size=256
    ):
    """
    Predição com tqdm (útil no notebook), salvando imagens anotadas, txt e recortes.
    """
    # Normalizar device
    if isinstance(device, str):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device in ['0', 'cuda', 'cuda:0'] else torch.device(device)

    model = YOLO(weights)
    model.to(device)

    print(f"[predict] weights={weights} | imgsz={img_size} | conf={conf_thres} | iou={iou_thres}")

    # Lista de imagens
    images = [os.path.join(source_images_dir, f)
              for f in os.listdir(source_images_dir)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    all_results = []
    for i in tqdm(range(0, len(images), chunk_size), desc="Inferência", unit="img"):
        chunk = images[i:i+chunk_size]
        r_chunk = model.predict(
            source=chunk,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            workers=workers,
            save=save_img,
            save_txt=save_txt,
            save_crop=save_crop,
            project=project,
            name=name,
            verbose=False
        )
        all_results.extend(r_chunk)

    print(f"[predict] concluído → {len(all_results)} resultados")
    return model, all_results

# === EVALUATION ===
def evaluate_yolov12(
        model, data_yaml,
        img_size=640, conf_thres=0.25, iou_thres=0.45,
        device='0', workers=4,
        cache="disk",
        project='yolov12/runs', name='eval'
    ):
    """
    Avaliação no TEST (se definido) ou VAL (fallback).
    """
    if isinstance(device, str):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device in ['0', 'cuda', 'cuda:0'] else torch.device(device)

    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    eval_split = cfg.get('test', cfg.get('val'))
    test_yaml = os.path.join(os.path.dirname(data_yaml), 'data_eval.yaml')
    cfg_eval = {**cfg, 'val': eval_split}
    with open(test_yaml, 'w') as f:
        yaml.dump(cfg_eval, f, sort_keys=False)

    metrics = model.val(
        data=test_yaml,
        device=device,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        workers=workers,
        cache=cache,      # <── o cache="disk"
        project=project,
        name=name
    )
    print("[eval]", metrics.results_dict)
    return metrics

# ====== Main Execution ======
def main(
    # Caminhos (passe os absolutos ou relativos ao notebook)
    train_img_dir,
    val_img_dir,
    test_img_dir,
    data_yaml_path,

    # Treino
    weights='yolo12x.pt',
    img_size=640,
    batch_size=8,
    epochs=200,
    device='0',
    workers=4,
    cache = 'disk',

    # Predição/Avaliação
    do_predict=True,
    predict_source_dir=None,   # se None, usa test_img_dir
    pred_conf=0.25,
    pred_iou=0.45,
    save_txt=True, save_img=True, save_crop=True,
    project='yolov12/runs',
    run_name='exp',

    # Augmentations
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.0, fliplr=0.5,
    translate=0.08, scale=0.2, shear=0.0,
    perspective=0.0,
    mosaic=0.1, mixup=0.0, close_mosaic=10,
):
    """
    Chamado direto do notebook. 
    Cria data.yaml (se não existir), treina, e (opcional) prediz/avalia com tqdm.
    """
    # 1) data.yaml
    if not os.path.exists(data_yaml_path):
        create_data_yaml(train_img_dir, val_img_dir, test_img_dir, data_yaml_path, class_names=('words',))

    # 2) Treino
    model, results = train_yolov12(
        data_yaml=data_yaml_path,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        weights=weights,
        device=device,
        workers=workers,
        project=project,
        name=run_name,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        flipud=flipud, fliplr=fliplr,
        translate=translate, scale=scale, shear=shear,
        perspective=perspective,
        mosaic=mosaic, mixup=mixup, close_mosaic=close_mosaic
    )

    # 3) Métricas do treino (robusto à variação de chaves)
    md = getattr(results, "results_dict", {}) or {}
    map50 = md.get('metrics/mAP50(B)', md.get('metrics/mAP50', md.get('metrics/mAP_0.5')))
    prec  = md.get('metrics/precision(B)', md.get('metrics/precision'))
    rec   = md.get('metrics/recall(B)', md.get('metrics/recall'))
    print(f"[train] mAP50={map50} | P={prec} | R={rec}")

    # 4) Predição/Avaliação com tqdm
    metrics = None
    if do_predict:
        src = predict_source_dir or test_img_dir
        _, metrics = predict_and_evaluate_yolov12(
            weights=os.path.join(project, run_name, 'weights', 'best.pt'),
            source_images_dir=src,
            data_yaml=data_yaml_path,
            img_size=img_size,
            conf_thres=pred_conf,
            iou_thres=pred_iou,
            device=device,
            workers=workers,
            save_txt=save_txt, save_img=save_img, save_crop=save_crop,
            project=project,
            name=f'{run_name}_pred'
        )

    return model, results, metrics