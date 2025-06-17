# src/processar_com_ohsome.py
"""Função para processar células com a API Ohsome, filtrando por classes e tags
e retornando as features pontuais com a contribuição mais recente com name=* em uma janela temporal
considerando a data de inflexão e uma data final fixa (2025-04-06T13:00Z) para cada classe/tag.
"""


# -*- coding: utf-8 -*-
import requests
import pandas as pd
from shapely.geometry import shape, mapping
import json

def processar_com_ohsome(cell_row, classe_et_edgv_to_tags, log_mensagem, log_path):
    id_celula = cell_row["id"]
    bbox = cell_row.geometry.bounds  # (minx, miny, maxx, maxy)
    features_resultantes = []
    url_ohsome_latest = "https://api.ohsome.org/v1/contributions/latest/geometry"

    for classe, tags in classe_et_edgv_to_tags.items():
        ratio_col = f"step1_consolidado_{classe}_name_ratio"
        inflexao_col = f"step6_consolidado_{classe}_inflexao_data"

        if pd.isna(cell_row.get(ratio_col)) or cell_row[ratio_col] <= 0:
            continue

        data_inicio_str = cell_row.get(inflexao_col)
        if not isinstance(data_inicio_str, str) or data_inicio_str.strip() == "" or data_inicio_str.lower() == "none":
            continue

        try:
            data_inicio = pd.to_datetime(data_inicio_str, errors="coerce")
            if pd.isna(data_inicio):
                continue
        except Exception as e:
            log_mensagem(log_path, id_celula, f"[ERRO] Conversão de data inflexão inválida: {data_inicio_str} - {e}")
            continue

        data_fim = pd.Timestamp("2025-04-06T13:00Z").strftime("%Y-%m-%d")
        data_inicio_str = data_inicio.strftime("%Y-%m-%d")

        for tag, value in tags:
            payload = {
                "bboxes": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "time": f"{data_inicio_str},{data_fim}",
                "filter": f"{tag}={value} and name=*",
                "properties": "metadata,tags",
                "clipGeometry": "false"
            }

            try:
                print(f"[DEBUG] Célula: {id_celula}, Classe: {classe}, Tag={tag}, Value={value}")
                print(f"[DEBUG] Payload: {json.dumps(payload)}")
                response = requests.post(url_ohsome_latest, data=payload)
                print(f"[DEBUG] Status Code: {response.status_code}")
                response.raise_for_status()

                dados = response.json()

                for feat in dados.get("features", []):
                    geom_data = feat.get("geometry")
                    props = feat.get("properties", {})

                    if geom_data is None:
                        continue

                    geom = shape(geom_data)
                    if geom.geom_type in ["Polygon", "MultiPolygon"]:
                        geom = geom.centroid

                    props_clean = {
                        "id_celula": id_celula,
                        "classe": classe,
                        "tag": tag,
                        "value": value,
                        **props
                    }

                    features_resultantes.append({
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": props_clean
                    })

            except Exception as e:
                log_mensagem(log_path, id_celula, f"[ERRO OHSOME {classe}] {tag}={value}: {str(e)}")

    return features_resultantes
# Fim do módulo processar_com_ohsome.py