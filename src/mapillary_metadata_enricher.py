# Mapillary Metadata Enricher
# src/mapillary_metadata_enricher.py
# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
import geopandas as gpd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ler_token_mapillary(caminho="configs/mapillary_token.txt"):
    """Lê o token de acesso do Mapillary"""
    try:
        with open(caminho, 'r') as f:
            token = f.read().strip()
        if not token:
            raise ValueError("Token vazio")
        return token
    except Exception as e:
        logger.error(f"Erro ao ler token: {e}")
        raise

def obter_metadados_imagem(image_id, token, campos=None):
    """Obtém metadados completos para uma imagem específica"""
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"access_token": token}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error para imagem {image_id}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Erro geral para imagem {image_id}: {e}")
        return None

def processar_campos_especiais(metadata):
    """Processa campos aninhados e especiais com datetime corrigido"""
    processed = {}
    
    # Campos aninhados
    nested_fields = {
        'creator': ['id', 'username'],
        'mesh': ['id', 'url'],
        'sfm_cluster': ['id', 'url'],
        'computed_geometry': ['type', 'coordinates']
    }
    
    for field, subfields in nested_fields.items():
        if field in metadata:
            for subfield in subfields:
                if subfield in metadata[field]:
                    key = f"{field}_{subfield}"
                    processed[key] = metadata[field][subfield]
    
    # Campo de data/hora (corrigido para usar timezone UTC)
    if 'captured_at' in metadata:
        try:
            timestamp = int(metadata['captured_at'])
            # Correção: usando timezone UTC
            dt = datetime.fromtimestamp(timestamp/1000, tz=timezone.utc)
            processed['captured_date'] = dt.isoformat()
        except Exception as e:
            logger.error(f"Erro ao converter captured_at: {e}")
            processed['captured_date'] = None
    
    return processed

def enriquecer_geodataframe(gdf, token, max_workers=10, batch_size=100):
    """Enriquece um GeoDataFrame com metadados completos do Mapillary"""
    if gdf.empty:
        logger.warning("GeoDataFrame vazio - nada para enriquecer")
        return gdf
    
    # Lista de IDs únicos de imagens (usando 'image_id' conforme solicitado)
    image_ids = gdf['image_id'].unique().tolist()
    logger.info(f"Iniciando enriquecimento para {len(image_ids)} imagens únicas")
    
    # Dicionário para armazenar metadados
    todos_metadados = {}
    
    # Processar em lotes para evitar sobrecarga de memória
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i:i+batch_size]
        logger.info(f"Processando lote {i//batch_size + 1}: IDs {i} a {i+len(batch_ids)-1}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(obter_metadados_imagem, img_id, token): img_id
                for img_id in batch_ids
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    metadata = future.result()
                    if metadata:
                        # Processar campos especiais e aninhados
                        processed = processar_campos_especiais(metadata)
                        todos_metadados[img_id] = {**metadata, **processed}
                except Exception as e:
                    logger.error(f"Erro ao processar imagem {img_id}: {e}")
        
        # Respeitar rate limits (1 lote por segundo)
        time.sleep(1)
    
    logger.info("Metadados coletados, adicionando ao GeoDataFrame...")
    
    # Adicionar campos ao GeoDataFrame
    for campo in set().union(*(d.keys() for d in todos_metadados.values())):
        if campo not in gdf.columns:
            gdf[campo] = None
    
    # Preencher valores
    for idx, row in gdf.iterrows():
        img_id = row['image_id']
        if img_id in todos_metadados:
            for campo, valor in todos_metadados[img_id].items():
                # Não sobrescrever captured_at original
                if campo != 'captured_at' or campo not in gdf.columns:
                    gdf.at[idx, campo] = valor
    
    logger.info("Enriquecimento concluído!")
    return gdf

def salvar_geodataframe_enriquecido(gdf, output_path):
    """Salva o GeoDataFrame enriquecido em formato GPKG"""
    if gdf.empty:
        logger.warning("Nada para salvar - GeoDataFrame vazio")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Simplificar campos complexos
        for col in gdf.columns:
            if gdf[col].dtype == object:
                try:
                    gdf[col] = gdf[col].astype(str)
                except:
                    gdf[col] = gdf[col].apply(str)
        
        # Salvar em GPKG
        gdf.to_file(output_path, driver='GPKG', layer='mapillary_points_enriched')
        logger.info(f"Dados enriquecidos salvos em {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar GPKG: {e}")
        return False