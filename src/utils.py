# módulo utils.py
"""funções reutilizáveis para:
1 - Inicializar e gravar logs (init_log, log_mensagem),
2 - Manter o Colab ativo (start_keep_alive),
3 - Retry com backoff (retry_api_call),
4 - Copiar features com segurança (copiar_feature),
5 - Consolidar arquivos GeoJSON em um só (consolidar_geojson).
"""

# utils.py (atualizado)
import time, json, csv, threading
from datetime import datetime
from copy import deepcopy
import glob

def init_log(log_path):
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["lote", "mensagem", "timestamp"])

def log_mensagem(log_path, lote, mensagem):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([lote, mensagem, timestamp])

def start_keep_alive(log_path, keep_alive_flag):
    import threading
    def keep_alive():
        while keep_alive_flag["running"]:
            time.sleep(300)
            print("Ainda trabalhando...")
            log_mensagem(log_path, "keep_alive", "Ainda trabalhando...")
    thread = threading.Thread(target=keep_alive, daemon=True)
    thread.start()
    return thread

def consolidar_geojson(pasta_saida, padrao_arquivo, nome_saida):
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
