# -*- coding: utf-8 -*-
# módulo processar_com_overpass.py
"""Função para processar células com a API Overpass, filtrando por classes e tags
e retornando as features encontradas. Utiliza a biblioteca requests para fazer
requisições HTTP e pandas para manipulação de dados.
"""

import requests
import pandas as pd
from shapely.geometry import shape, box, Point, mapping

def processar_com_overpass(cell_row, classe_et_edgv_to_tags, log_mensagem, log_path):
    try:
        id_celula = cell_row["id"]
        print(f"[ENTROU] processar_com_overpass para célula {id_celula}")
        bbox = cell_row.geometry.bounds
        features_resultantes = []

        for classe, tags in classe_et_edgv_to_tags.items():
            ratio_col = f"step1_consolidado_{classe}_name_ratio"
            if pd.isna(cell_row.get(ratio_col)) or cell_row[ratio_col] <= 0:
                continue

            for tag, value in tags:
                query = f"""
                [out:json][timeout:25];
                (
                  node["{tag}"="{value}"]["name"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                  way["{tag}"="{value}"]["name"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                  relation["{tag}"="{value}"]["name"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                );
                out body geom;
                """

                print(f"[DEBUG] Célula ID: {id_celula}, Classe: {classe}, Tag={tag}, Value={value}")
                print(f"[DEBUG] Query Overpass:\n{query}")

                response = requests.post("http://overpass-api.de/api/interpreter", data={"data": query})
                print(f"[DEBUG] Status code: {response.status_code}")
                response.raise_for_status()
                dados = response.json()
                elementos = dados.get("elements", [])
                print(f"[DEBUG] Total elementos retornados: {len(elementos)}")

                for el in elementos:
                    if "geometry" not in el and el.get("type") != "node":
                        continue

                    if el.get("type") == "node":
                        geom = Point(el["lon"], el["lat"])
                    else:
                        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                        geom = shape({"type": "LineString", "coordinates": coords}).centroid

                    props_clean = {
                        "id_celula": id_celula,
                        "classe": classe,
                        "tag": tag,
                        "value": value,
                        **{k: el[k] for k in el if k not in {"geometry", "lat", "lon"} and k != "tags"},
                        **el.get("tags", {})
                    }

                    features_resultantes.append({
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": props_clean
                    })

        print(f"[SAIU] {len(features_resultantes)} features encontradas para célula {id_celula}")
        return features_resultantes

    except Exception as e:
        print(f"[ERRO THREAD] {e}")
        log_mensagem(log_path, "thread", f"[ERRO] {e}")
        return []