# módulo utils.py
"""funções reutilizáveis para:
1 - Inicializar e gravar logs (init_log, log_mensagem),
2 - Manter o Colab ativo (start_keep_alive),
3 - Retry com backoff (retry_api_call),
4 - Copiar features com segurança (copiar_feature),
5 - Consolidar arquivos GeoJSON em um só (consolidar_geojson).
"""

import time
import json
import csv
import requests
from datetime import datetime
from pathlib import Path
from copy import deepcopy

# Logging

def init_log(log_path):
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["lote", "mensagem", "timestamp"])

def log_mensagem(log_path, lote, mensagem):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([lote, mensagem, timestamp])

# Keep-alive para Colab

def start_keep_alive(log_path):
    import threading
    def keep_alive():
        while True:
            time.sleep(300)
            print("\u23F3 Ainda trabalhando...")
            log_mensagem(log_path, "keep_alive", "Ainda trabalhando...")
    threading.Thread(target=keep_alive, daemon=True).start()

# Retry wrapper para chamadas de rede

def retry_api_call(func, max_retries=3, backoff_base=2):
    for attempt in range(max_retries):
        result = func()
        if result[0] is not None:
            return result
        time.sleep(backoff_base ** attempt)
    return None, result[1]  # Retorna o erro da última tentativa

# Cópia segura de features

def copiar_feature(feature):
    return deepcopy(feature)

# Consolida arquivos geojson em um só

def consolidar_geojson(pasta_saida, padrao_arquivo, nome_saida):
    import glob
    arquivos = sorted(glob.glob(str(pasta_saida / padrao_arquivo)))
    todas_features = []
    for arquivo in arquivos:
        with open(arquivo, 'r', encoding='utf-8') as f:
            fc_parcial = json.load(f)
            todas_features.extend(fc_parcial['features'])

    final_fc = {"type": "FeatureCollection", "features": todas_features}
    with open(pasta_saida / nome_saida, 'w', encoding='utf-8') as f:
        json.dump(final_fc, f)

    return len(todas_features)