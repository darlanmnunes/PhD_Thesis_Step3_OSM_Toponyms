# Mapillary Tile Downloader
# src/mapillary_tile_downloader.py
# -*- coding: utf-8 -*-
import requests
import mercantile
import mapbox_vector_tile
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape
from shapely.validation import make_valid
from pathlib import Path
import time
import json
import os

def ler_token_mapillary(caminho="configs/mapillary_token.txt"):
    with open(caminho, 'r') as f:
        return f.read().strip()

def obter_tiles_para_bbox(bbox, zoom=14):
    """Calcula tiles necessários para cobrir uma bbox com validação"""
    # Se for uma lista de pontos, converter para bbox simples
    if isinstance(bbox[0], (list, tuple)):
        all_lons = [coord[0] for coord in bbox]
        all_lats = [coord[1] for coord in bbox]
        bbox = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
    
    try:
        west, south, east, north = bbox
        return list(mercantile.tiles(west, south, east, north, zoom))
    except Exception as e:
        print(f"Erro ao processar bbox {bbox}: {e}")
        raise ValueError("Formato de bbox inválido. Esperado: [west, south, east, north]")
    

def baixar_tile(z, x, y, token):
    url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def converter_coordenadas_tile(x, y, z, coord, extent=4096):
    """Conversão precisa de coordenadas vetoriais para EPSG:4326"""
    # Normalizar coordenadas
    u, v = coord
    u_norm = u / extent
    v_norm = v / extent
    
    # Obter os limites do tile
    bounds = mercantile.bounds(mercantile.Tile(x, y, z))
    
    # Calcular longitude/latitude
    lon = bounds.west + u_norm * (bounds.east - bounds.west)
    lat = bounds.south + v_norm * (bounds.north - bounds.south)
    
    return lon, lat

def converter_captured_at(valor):
    """Conversão segura para evitar overflow e valores negativos"""
    try:
        str_val = str(valor).strip()
        if str_val.replace('-', '').isdigit():
            num = int(str_val)
            return num if num > 0 else None
        return None
    except:
        return None

def processar_feature_ponto(feature, x, y, z):
    """Processa apenas features do tipo ponto da camada 'image'"""
    if feature['geometry']['type'] != 'Point':
        return None
    
    props = feature['properties']
    
    # Verificar se é um ponto de imagem (não sequência)
    if 'id' not in props:
        return None
    
    # Converter coordenadas
    coords = feature['geometry']['coordinates']
    lon, lat = converter_coordenadas_tile(x, y, z, coords)
    
    # Criar ponto com todas propriedades
    #ponto = Point(lon, lat)
    #feature_data = {
    #    'geometry': ponto,
    #    'properties': props
    #}

    return {
        'geometry': Point(lon, lat),
        'properties': {
            'image_id': props.get('id'),
            'captured_at': converter_captured_at(props.get('captured_at')),
            'compass_angle': props.get('compass_angle'),
            'creator_id': props.get('creator_id'),
            'sequence_id': props.get('sequence_id'),
            'is_pano': props.get('is_pano'),
            'organization_id': props.get('organization_id'),
            'tile_z': z,
            'tile_x': x,
            'tile_y': y
        }
    }

def extrair_pontos_do_tile(tile_data, x, y, z):
    features = []
    try:
        decoded = mapbox_vector_tile.decode(tile_data)
        
        # Processar apenas a camada 'image' (pontos)
        layer_data = decoded.get('image', None)
        if not layer_data:
            return features
            
        for feature in layer_data['features']:
            feature_data = processar_feature_ponto(feature, x, y, z)
            if feature_data:
                features.append(feature_data)
                
    except Exception as e:
        print(f"Erro na decodificação do tile: {e}")
    return features

def corrigir_geometrias(gdf):
    """Corrige geometrias inválidas e remove vazias"""
    if gdf.empty:
        return gdf
    
    # Remover geometrias vazias
    gdf = gdf[~gdf.geometry.is_empty]
    
    # Corrigir geometrias inválidas
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        print(f"  → Corrigindo {invalid_mask.sum()} geometrias inválidas")
        gdf.loc[invalid_mask, 'geometry'] = gdf[invalid_mask].geometry.apply(make_valid)
    
    return gdf

def padronizar_schema(gdf):
    """Garante colunas consistentes para todos os lotes"""
    #colunas_esperadas = [
    #    'image_id', 'captured_at', 'creator_id', 'sequence_id', 
    #    'is_pano', 'compass_angle', 'organization_id', 'tile_layer'
    #]
    # Verificar se o GeoDataFrame já tem as colunas esperadas
    #for col in colunas_esperadas:
    #    if col not in gdf.columns:
    #        gdf[col] = None
    #return gdf[['geometry'] + colunas_esperadas]

    colunas_prioritarias = [
        'image_id', 'captured_at', 'compass_angle', 'creator_id',
        'sequence_id', 'is_pano', 'organization_id', 'geometry'
    ]
    
    # Adicionar colunas faltantes
    for col in colunas_prioritarias:
        if col not in gdf.columns:
            gdf[col] = None
    
    # Ordenar colunas
    outras_colunas = [c for c in gdf.columns if c not in colunas_prioritarias]
    return gdf[colunas_prioritarias + outras_colunas]

def processar_area_abrangente(bbox, token, zoom=14):
    """Processa todos os tiles de uma área completa"""
    tiles = obter_tiles_para_bbox(bbox, zoom)
    todas_features = []
    
    print(f"Processando {len(tiles)} tiles para a área...")
    
    for i, tile in enumerate(tiles):
        try:
            tile_data = baixar_tile(tile.z, tile.x, tile.y, token)
            features = extrair_pontos_do_tile(tile_data, tile.x, tile.y, tile.z)
            todas_features.extend(features)
            
            if (i + 1) % 10 == 0:
                print(f"  → Processados {i+1}/{len(tiles)} tiles")
                
        except Exception as e:
            print(f"Erro no tile {tile.z}/{tile.x}/{tile.y}: {e}")
    
    if not todas_features:
        return gpd.GeoDataFrame()
    
    # Criar GeoDataFrame
    rows = []
    for feat in todas_features:
        row = feat['properties'].copy()
        row['geometry'] = feat['geometry']
        rows.append(row)
    
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")
    return gdf

def salvar_resultados(gdf, output_path):
    """Salva todos os pontos em um único arquivo GPKG"""
    if gdf.empty:
        print("Nenhum ponto para salvar")
        return False
    
    # Etapas críticas de processamento
    gdf = padronizar_schema(gdf)
    gdf = corrigir_geometrias(gdf)
    
    # Garantir tipos de dados consistentes
    for col in gdf.columns:
        if gdf[col].dtype == object:
            gdf[col] = gdf[col].astype(str)
    
    # Salvar em GPKG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        gdf.to_file(output_path, driver='GPKG', layer='mapillary_points')
        print(f"Pontos salvos em {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar GPKG: {e}")
        # Backup em GeoJSON
        backup_path = output_path.with_suffix('.geojson')
        gdf.to_file(backup_path, driver='GeoJSON')
        print(f"  → Backup salvo em {backup_path}")
        return False