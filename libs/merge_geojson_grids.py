# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import timedelta
import plotly.express as px
import nbformat

def merge_geojson_files(file_paths, output_path):
    """
    Merge múltiplos arquivos GeoJSON, mantendo apenas a primeira ocorrência de cada célula (baseado no 'id').
    
    Args:
        file_paths (list): Lista de caminhos para os arquivos GeoJSON de entrada.
        output_path (str): Caminho para o arquivo GeoJSON de saída.
    """
    # Dicionário para armazenar features únicas (chave: id da célula)
    unique_features = {}
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        for feature in geojson_data['features']:
            feature_id = feature['properties']['id']
            
            # Só adiciona se o ID não existir no dicionário
            if feature_id not in unique_features:
                unique_features[feature_id] = feature
    
    # Cria a FeatureCollection com as features únicas
    merged_geojson = {
        "type": "FeatureCollection",
        "features": list(unique_features.values())
    }
    
    # Salva o resultado
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_geojson, f, ensure_ascii=False, indent=2)
    
    print(f"Merge concluído! Arquivo salvo em: {output_path}")


def get_geojson_files(directory):
    """
    Obtém todos os arquivos GeoJSON em um diretório específico.
    
    Args:
        directory (str): Caminho do diretório a ser pesquisado.
    
    Returns:
        list: Lista de caminhos para os arquivos GeoJSON encontrados.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.geojson')]
