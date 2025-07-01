# Mapillary Metadata Enricher
# src/mapillary_metadata_enricher.py
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import geopandas as gpd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
from pathlib import Path
from tqdm import tqdm

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

def obter_metadados_imagem(image_id, token):
    """Obtém metadados completos para uma imagem específica"""
    url = f"https://graph.mapillary.com/{image_id}"
    # fields a serem solicitados conforme https://www.mapillary.com/developer/api-documentation?locale=pt_PT#image
    fields = [
        "altitude", "atomic_scale", "camera_parameters", "camera_type", "captured_at",
        "compass_angle", "computed_altitude", "computed_compass_angle", "computed_geometry",
        "computed_rotation", "creator", "exif_orientation", "geometry", "height", "is_pano",
        "make", "model", "thumb_256_url", "thumb_1024_url", "thumb_2048_url",
        "thumb_original_url", "merge_cc", "mesh", "sequence", "sfm_cluster", "width",
        "detections"
    ]
    params = {
        "access_token": token,
        "fields": ",".join(fields)
    }
    
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
    
    # Campo de data/hora (timezone UTC)
    if 'captured_at' in metadata:
        try:
            timestamp = int(metadata['captured_at'])
            dt = datetime.fromtimestamp(timestamp/1000, tz=timezone.utc)
            processed['captured_date'] = dt.isoformat()
        except Exception as e:
            logger.error(f"Erro ao converter captured_at: {e}")
            processed['captured_date'] = None
    
    return processed

def enriquecer_geodataframe(gdf, token, max_workers=10, batch_size=100):
    """Enriquece um GeoDataFrame com metadados completos do Mapillary image entity"""
    if gdf.empty:
        logger.warning("GeoDataFrame vazio - nada para enriquecer")
        return gdf
    
    image_ids = gdf['image_id'].unique().tolist()
    logger.info(f"Iniciando enriquecimento para {len(image_ids)} imagens únicas")
    
    todos_metadados = {}
    pbar = tqdm(total=len(image_ids), desc="Obtendo metadados")
    
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i:i+batch_size]
        batch_metadados = {}
        
        sleep_time = max(0.5, batch_size / 100)
        time.sleep(sleep_time)
        
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
                        processed = processar_campos_especiais(metadata)
                        # Junta os campos do dicionário original + processados
                        batch_metadados[img_id] = {**metadata, **processed}
                    else:
                        logger.warning(f"Metadados vazios para {img_id}")
                except Exception as e:
                    logger.error(f"Erro ao processar imagem {img_id}: {e}")
                finally:
                    pbar.update(1)
        
        todos_metadados.update(batch_metadados)
        del batch_metadados
    
    pbar.close()
    
    if not todos_metadados:
        logger.warning("Nenhum metadado obtido. Retornando GeoDataFrame original.")
        return gdf
    
    logger.info(f"Metadados obtidos para {len(todos_metadados)} imagens")
    logger.info("Criando DataFrame de metadados...")
    metadados_df = pd.DataFrame.from_dict(todos_metadados, orient='index')
    metadados_df.index.name = 'image_id'
    metadados_df.reset_index(inplace=True)
    
    logger.info(f"Campos de metadados obtidos ({len(metadados_df.columns)}): {list(metadados_df.columns)}")
    logger.info("Realizando merge com dados originais...")
    
    # Merge mantendo a geometria original
    gdf_enriched = gdf.merge(
        metadados_df,
        how='left',
        on='image_id',
        suffixes=('_original', '_meta')
    )
    
    # Forçar coluna de geometria correta (original)
    if 'geometry_original' in gdf_enriched.columns:
        gdf_enriched['geometry'] = gdf_enriched['geometry_original']
        gdf_enriched.drop(columns=['geometry_original'], inplace=True)
    elif 'geometry' not in gdf_enriched.columns:
        # Se perdeu por algum motivo, restaura do gdf original
        gdf_enriched['geometry'] = gdf['geometry']
    
    # Criar novo GeoDataFrame e setar CRS
    gdf_enriched = gpd.GeoDataFrame(gdf_enriched, geometry='geometry', crs=gdf.crs)
    logger.info(f"Enriquecimento concluído! Total de colunas: {len(gdf_enriched.columns)}")
    return gdf_enriched

def salvar_geodataframe_enriquecido(gdf, output_path):
    """Salva o GeoDataFrame enriquecido em formato GPKG"""
    if gdf.empty:
        logger.warning("Nada para salvar - GeoDataFrame vazio")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Converter campos complexos para string
        for col in gdf.columns:
            if pd.api.types.is_object_dtype(gdf[col]) and col != 'geometry':
                gdf[col] = gdf[col].astype(str)
        
        # Salvar em GPKG
        gdf.to_file(output_path, driver='GPKG', layer='mapillary_points_enriched')
        logger.info(f"Dados enriquecidos salvos em {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar GPKG: {e}")
        return False