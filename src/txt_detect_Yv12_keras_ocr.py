# txt_detect_Yv12_keras_ocr.py
"""
Script para:
1. Rodar YOLOv12 sobre imagens (detecção de regiões de texto).
2. Gerar crops das regiões detectadas.
3. Reconhecer o texto nos crops usando keras-ocr (Recognizer isolado).
4. Retornar DataFrame com resultados (osmId, image_id, texto reconhecido, path_crop).
"""

import os
import cv2
import torch
import gc
import keras_ocr
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from ultralytics import YOLO

# Configura o uso dinâmico da GPU para o TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ======================
# --- Inicialização OCR
# ======================
def init_recognizer():
    """Inicializa keras-ocr Recognizer (sem detector)."""
    recognizer = keras_ocr.recognition.Recognizer()
    recognizer.compile()
    return recognizer

# ======================
# --- Predição YOLOv12 + CROP
# ======================

def yolo_predict_and_crop(
    weights, source_dir, output_dir,
    img_size=640, conf_thres=0.25, iou_thres=0.45,
    device=None, workers=4
):
    """
    - Roda inferência YOLOv12 sobre imagens em source_dir.
    - Salva crops no output_dir (preserva hierarquia de pastas).
    - Retorna DataFrame com metadados + salva log de imagens sem detecção.
    """

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Carregar modelo YOLO
    model = YOLO(weights)
    model.to(device)

    results_data = []
    no_detections = []

    # Percorrer imagens
    for root, _, files in os.walk(source_dir):
        rel_dir = os.path.relpath(root, source_dir)
        parts = rel_dir.split(os.sep)

        # Exibir só "classe/tag_value"
        if len(parts) >= 2:
            short_desc = f"{parts[0]}\\{parts[1]}"
        else:
            short_desc = rel_dir  # fallback para raiz

        # Lista apenas imagens
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            continue  # <-- pula diretórios sem imagens

        for fname in tqdm(image_files, desc=f"Processando {short_desc}", unit="img"):
            img_path = os.path.join(root, fname)
            image_id = os.path.splitext(fname)[0]

            # Inferência YOLO
            preds = model.predict(
                source=img_path,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                device=device,
                workers=workers,
                save=False,
                verbose=False
            )

            boxes = preds[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                # Nenhuma detecção
                rel_dir = os.path.relpath(root, source_dir)
                parts = rel_dir.split(os.sep)
                osmId_formatado = parts[-1] if parts else None
                no_detections.append({
                    "osmId_formatado": osmId_formatado,
                    "image_id": image_id,
                    "path": img_path
                })
                continue

            # Salvar crops
            for det_idx, det in enumerate(boxes):
                x1, y1, x2, y2 = map(int, det[:4])
                crop = cv2.imread(img_path)[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue

                # Gerar caminho preservando hierarquia
                rel_dir = os.path.relpath(root, source_dir)
                crop_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(crop_dir, exist_ok=True)

                crop_name = f"{image_id}_crop{det_idx}.jpg"
                crop_path = os.path.join(crop_dir, crop_name)
                cv2.imwrite(crop_path, crop)

                # Extrair atributos
                parts = rel_dir.split(os.sep)
                osmId_formatado = parts[-1] if parts else None
                classe = parts[0] if len(parts) > 0 else None
                tag_value = parts[1] if len(parts) > 1 else None

                results_data.append({
                    "osmId_formatado": osmId_formatado,
                    "classe": classe,
                    "tag_value": tag_value,
                    "image_id": image_id,
                    "path_crop": crop_path
                })

    # Salvar log de não detecções
    log_no_det = os.path.join(output_dir, "no_detections.csv")
    pd.DataFrame(no_detections).to_csv(log_no_det, index=False)
    print(f"[INFO] Log de imagens sem detecção salvo em {log_no_det}")

    return pd.DataFrame(results_data)

# ======================
# --- Reconhecimento Name  KERAS-OCR
# ======================

# --- Pré-processamento de imagem para OCR ---
# Nao utilizado: resultados piores quando utilizado
def preprocess_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Se muito pequeno, aumenta resolução
    if gray.shape[0] < 32 or gray.shape[1] < 128:
        gray = cv2.resize(gray, (max(128, gray.shape[1]*2), max(32, gray.shape[0]*2)))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # keras-ocr espera 3 canais

def run_ocr_on_crops(df_crops, recognizer):
    """
    Roda OCR nos crops listados no DataFrame retornado pelo YOLO.
    Retorna DataFrame com colunas extras:
      - words_sli: lista de palavras detectadas
      - texto_sli: string com palavras separadas por espaço
    """
    results = []

    for path_crop in tqdm(df_crops["path_crop"], desc="OCR nos crops", unit="img"):
        try:
            img = keras_ocr.tools.read(path_crop)
            preds = recognizer.recognize(img)   # <- apenas uma imagem
            words = preds if preds else []
            text  = " ".join(words) if words else ""
        except Exception as e:
            words, text = [], f"[ERRO_OCR:{e}]"

        row = df_crops[df_crops["path_crop"] == path_crop].iloc[0].to_dict()
        row.update({
            "words_sli": words,
            "texto_sli": text
        })
        results.append(row)

        # Liberação de memória
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(results)

    # reorganizar colunas
    preferred = ["osmId_formatado", "classe", "tag_value", "image_id", "words_sli", "texto_sli"]
    others = [c for c in df_out.columns if c not in preferred + ["path_crop"]]
    col_order = preferred + others + ["path_crop"]
    df_out = df_out[col_order]

    return df_out

# ---
def run_ocr_on_crops1(df_crops, recognizer):
    """
    Roda OCR nos crops listados no DataFrame retornado pelo YOLO.
    Retorna DataFrame com coluna extra 'texto_sli'.
    """
    textos = []
    for path_crop in tqdm(df_crops["path_crop"], desc="OCR nos crops"):
        try:
            img = keras_ocr.tools.read(path_crop)
            preds = recognizer.recognize(img)
            texto = " ".join(preds) if preds else ""
        except Exception as e:
            texto = f"[ERRO_OCR:{e}]"
        textos.append(texto)

    df_crops = df_crops.copy()
    df_crops["texto_sli"] = textos
    return df_crops

def run_ocr_on_crops2(df_crops, recognizer):
    """
    Roda OCR nos crops listados no DataFrame retornado pelo YOLO.
    Retorna DataFrame com colunas extras:
      - words_sli: lista de palavras detectadas
      - texto_sli: lista (igual a words_sli, para facilitar comparações posteriores)
    """
    all_words = []
    all_texts = []

    for path_crop in tqdm(df_crops["path_crop"], desc="OCR nos crops", unit="img"):
        try:
            img = keras_ocr.tools.read(path_crop)
            preds = recognizer.recognize(img)  # lista de strings
            words = preds if preds else []
            text  = " ".join(words) if words else ""  # cópia da lista, sem concatenação
        except Exception as e:
            words, text = [], [f"[ERRO_OCR:{e}]"]

        all_words.append(words)
        all_texts.append(text)

    df_out = df_crops.copy()
    df_out["words_sli"] = all_words
    df_out["texto_sli"] = all_texts

    # reorganizar colunas
    preferred = ["osmId_formatado", "classe", "tag_value", "image_id", "words_sli", "texto_sli"]
    others = [c for c in df_out.columns if c not in preferred + ["path_crop"]]
    col_order = preferred + others + ["path_crop"]
    df_out = df_out[col_order]

    return df_out

# ======================
# --- Rebuild df_crops
# ======================

def rebuild_df_crops(output_dir):
    """
    Reconstrói df_crops lendo os crops já salvos no disco.
    Retorna DataFrame no mesmo formato de yolo_predict_and_crop.
    """
    data = []
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path_crop = os.path.join(root, fname)
                image_id = fname.split("_crop")[0]  # recupera id original
                rel_dir = os.path.relpath(root, output_dir)
                parts = rel_dir.split(os.sep)
                osmId_formatado = parts[-1] if len(parts) > 0 else None
                classe = parts[0] if len(parts) > 0 else None
                tag_value = parts[1] if len(parts) > 1 else None

                data.append({
                    "osmId_formatado": osmId_formatado,
                    "classe": classe,
                    "tag_value": tag_value,
                    "image_id": image_id,
                    "path_crop": path_crop
                })

    df_crops = pd.DataFrame(data)
    return df_crops