# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import geopandas as gpd

def merge_geojson2gpkg(file_paths, output_path):

    """
    Mescla múltiplos arquivos GeoJSON preservando atributos únicos por step, com prefixo e removendo colunas duplicadas.
    Mantém apenas a primeira ocorrência de atributos com nomes idênticos entre steps.

    Args:
        file_paths (list): Lista de caminhos para os arquivos GeoJSON de entrada.
        output_path (str): Caminho para o arquivo GeoPackage de saída.

    Funcionamento do Código:

        1-Lê cada arquivo GeoJSON sequencialmente
        2-Renomeia os atributos para evitar colisões de nomes
        3-Mescla os atributos em um único GeoDataFrame
        4-Garante que o CRS seja armazenado no arquivo final (default EPSG:4674, caso ausente)
        5-Salva como GeoPackage
    """

    base_gdf = None
    seen_attributes = set()
    id_column = 'id'
    fixed_columns = ['id', 'POP10']
    geometry_column = 'geometry'
    steps_added = []

    for i, file_path in enumerate(file_paths):
        gdf = gpd.read_file(file_path)

        # CRS do primeiro arquivo
        if base_gdf is None:
            base_gdf = gdf[[id_column, geometry_column]].copy()
            if 'POP10' in gdf.columns:
                base_gdf['POP10'] = gdf['POP10']
            crs_ref = gdf.crs

        # Nome do step a partir do nome do arquivo
        step_name = os.path.splitext(os.path.basename(file_path))[0]

        # Selecionar atributos variáveis
        variable_cols = [col for col in gdf.columns if col not in fixed_columns + [geometry_column]]

        step_df = pd.DataFrame()
        step_df[id_column] = gdf[id_column]

        for col in variable_cols:
            attr_name = col.lower().strip()
            if attr_name not in seen_attributes:
                step_df[f"{step_name}_{col}"] = gdf[col]
                seen_attributes.add(attr_name)

        # Mesclar com base_gdf
        base_gdf = base_gdf.merge(step_df, on=id_column, how='left')
        steps_added.append(step_name)

    # Garantir o CRS
    base_gdf.set_crs(crs_ref, inplace=True)

    # Salvar como GeoPackage
    if not output_path.endswith('.gpkg'):
        output_path += '.gpkg'

    base_gdf.to_file(output_path, driver='GPKG', layer='merged_grid')

    print("Merge concluído!")
    print(f"Arquivo salvo como GeoPackage em: {output_path}")
    print(f"Steps processados: {steps_added}")
    print(f"CRS mantido: {crs_ref}")
    print(f"Total de células únicas: {len(base_gdf)}")

    return base_gdf

def merge_geojson2geojson(file_paths, output_path):
    """
    Mescla múltiplos arquivos GeoJSON preservando atributos únicos por step, com prefixo e removendo colunas duplicadas.
    Mantém apenas a primeira ocorrência de atributos com nomes idênticos entre steps.
    Salva o resultado como GeoJSON.
    """
    base_gdf = None
    seen_attributes = set()
    id_column = 'id'
    fixed_columns = ['id', 'POP10']
    geometry_column = 'geometry'
    steps_added = []

    for i, file_path in enumerate(file_paths):
        gdf = gpd.read_file(file_path)

        # CRS do primeiro arquivo
        if base_gdf is None:
            base_gdf = gdf[[id_column, geometry_column]].copy()
            if 'POP10' in gdf.columns:
                base_gdf['POP10'] = gdf['POP10']
            crs_ref = gdf.crs

        # Nome do step a partir do nome do arquivo
        step_name = os.path.splitext(os.path.basename(file_path))[0]

        # Selecionar atributos variáveis
        variable_cols = [col for col in gdf.columns if col not in fixed_columns + [geometry_column]]

        step_df = pd.DataFrame()
        step_df[id_column] = gdf[id_column]

        for col in variable_cols:
            attr_name = col.lower().strip()
            if attr_name not in seen_attributes:
                step_df[f"{step_name}_{col}"] = gdf[col]
                seen_attributes.add(attr_name)

        # Mesclar com base_gdf
        base_gdf = base_gdf.merge(step_df, on=id_column, how='left')
        steps_added.append(step_name)

    # Garantir o CRS
    base_gdf.set_crs(crs_ref, inplace=True)

    # Salvar como GeoJSON
    if not output_path.endswith('.geojson'):
        output_path += '.geojson'

    base_gdf.to_file(output_path, driver='GeoJSON')

    print("Merge concluído!")
    print(f"Arquivo salvo como GeoJSON em: {output_path}")
    print(f"Steps processados: {steps_added}")
    print(f"CRS mantido: {crs_ref}")
    print(f"Total de células únicas: {len(base_gdf)}")

    return base_gdf