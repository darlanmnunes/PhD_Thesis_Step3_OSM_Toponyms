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

    Funcionamento do Código:

        1-Lê cada arquivo GeoJSON sequencialmente
        2-Para cada feature, verifica se o ID já foi processado
        3-Mantém apenas a primeira ocorrência de cada ID (garantindo a prioridade do primeiro arquivo)
        4-Gera um novo GeoJSON com todas as features únicas
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

def merge_geojson_to_gpkg(file_paths, output_path):
    """
    Merge múltiplos arquivos GeoJSON, mantendo a primeira ocorrência de cada célula (baseado no 'id'),
    e salva como GeoPackage (GPKG) com o CRS original.

    Args:
        file_paths (list): Lista de caminhos para os arquivos GeoJSON de entrada.
        output_path (str): Caminho para o arquivo GeoPackage de saída (.gpkg).

    Funcionamento:
        1. Lê cada arquivo GeoJSON sequencialmente usando geopandas
        2. Mantém apenas a primeira ocorrência de cada ID
        3. Preserva o CRS do primeiro arquivo
        4. Salva como GeoPackage com índice espacial automático
    """
    # Lista para armazenar os GeoDataFrames
    gdfs = []
    input_crs = None  # CRS padrão inicialmente como None
    
    for file_path in file_paths:
        # Lê o arquivo com geopandas
        gdf = gpd.read_file(file_path)
        
        # Pega o CRS do primeiro arquivo para usar como referência
        if input_crs is None:
            input_crs = gdf.crs
        
        # Verifica se o CRS é compatível
        elif gdf.crs != input_crs:
            print(f"Aviso: CRS do arquivo {file_path} ({gdf.crs}) difere do CRS de referência ({input_crs}).")
        
        gdfs.append(gdf)
    
    # Concatena todos os GeoDataFrames
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=False))
    
    # Remove duplicatas baseado no ID (mantém a primeira ocorrência)
    merged_gdf = merged_gdf[~merged_gdf['id'].duplicated(keep='first')]
    
    # Garante o CRS original
    merged_gdf = merged_gdf.set_crs(input_crs)
    
    # Salva como GeoPackage
    if not output_path.endswith('.gpkg'):
        output_path = output_path + '.gpkg'
    
    merged_gdf.to_file(output_path, driver='GPKG', layer='merged_grid')
    print(f"Merge concluído! Arquivo salvo como GeoPackage em: {output_path}")
    print(f"CRS mantido: {input_crs}")
    print(f"Total de células únicas: {len(merged_gdf)}")
