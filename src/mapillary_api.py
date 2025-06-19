"""
This module contains functions to interact with the Mapillary API.
"""

import numpy as np

import os, json, requests
import requests
import wget
import geopandas as gpd
import pandas as pd
import mercantile
import folium
from folium.plugins import MarkerCluster
from shapely import Point,box
from math import cos, pi
from time import sleep
from datetime import datetime, timezone
from tqdm import tqdm


# from tqdm import tqdm
def get_mapillary_token():
    """
    Opens and reads the Mapillary token from the file 'configs/mapillary_token'.

    Returns:
        str: The Mapillary API token read from the file.
    """

    with open('configs/mapillary_token.txt', 'r') as f:
        return f.readline()
    
# right after the function definition
MAPPILARY_TOKEN = get_mapillary_token()

# --- Specific codes to deal with the input data ---
def meters_to_degrees(meters, latitude):
    """
    Convert meters to degrees of latitude and longitude.

    Parameters:
        meters (float): The distance in meters.
        latitude (float): The latitude in degrees.

    Returns:
        tuple: A tuple containing the degrees of latitude and longitude.
    """
    # 1 degree of latitude = approximately 111 km
    degrees_lat = meters / 111000
    # Adjust the calculation for longitude using cosine of latitude
    degrees_lon = meters / (111000 * abs(np.cos(np.radians(latitude))))
    return degrees_lat, degrees_lon


def create_buffer_poi(gdf, dist):
    """
    Create a GeoDataFrame of buffers around points of interest (POIs) in the given GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the POIs.
        dist (float): The distance in meters to create the buffers.

    Returns:
        region_roi (GeoDataFrame): The GeoDataFrame containing the buffers around the POIs.
    """
    buffers = []
    for idx, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        degrees_lat, degrees_lon = meters_to_degrees(dist, lat)
        poi_buffer = box(lon - degrees_lon, lat - degrees_lat, lon + degrees_lon, lat + degrees_lat)
        buffers.append({
            'study_case': row['study_case'],
            'name': row['name'],
            'geometry': poi_buffer
        })
    region_roi = gpd.GeoDataFrame(buffers, crs=gdf.crs)

    return region_roi

def visualizar_pontos_no_buffer(gdf, gdf_roi, output_path="map.html"):
    """
    Visualizes points within a buffer on a map using Folium library.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the points to be visualized.
        gdf_roi (GeoDataFrame): The GeoDataFrame containing the buffers to be added to the map.
        output_path (str, optional): The path to save the map as an HTML file. Defaults to "map.html".

    Returns:
        folium.Map: The Folium map object with the visualized points and buffers.

    Description:
        This function creates a map centered on the median point of the data. It adds the points of interest to the map as markers and the buffers as GeoJSON objects. 
        The map is then saved as an HTML file if the `output_path` parameter is provided.
    """
    # Criar um mapa centrado no ponto médio dos dados
    centroid = gdf.unary_union.centroid
    mapa = folium.Map(location=[centroid.y, centroid.x], zoom_start=15)
    
    # Adicionar os pontos de interesse ao mapa
    marker_cluster = MarkerCluster().add_to(mapa)
    for idx, row in gdf.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=row['study_case'],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)
    
    # Adicionar os buffers ao mapa (somente contornos externos)
    for idx, row in gdf_roi.iterrows():
        folium.GeoJson(row.geometry.__geo_interface__, style_function=lambda x: {'fillColor': 'none', 'color': 'red'}).add_to(mapa)
    
    # Salvar o mapa em um arquivo HTML
    #mapa.save(output_path)
    #print(f"Mapa salvo em: {output_path}")

    return mapa


# --- Specific codes to deal with Mapillary API ---

ZOOM_LEVEL = 18


def tile_bbox_to_box(tile_bbox,swap_latlon=False):
    """
    Convert a tile bounding box to a shapely box.

    Args:
        tile_bbox (Box): The tile bounding box.
        swap_latlon (bool, optional): Whether to swap the latitude and longitude coordinates. Defaults to False.

    Returns:
        Box: The shapely box representing the tile bounding box.
    """
    if swap_latlon:
        return box(tile_bbox.south,tile_bbox.west,tile_bbox.north,tile_bbox.east)
    else:
        return box(tile_bbox.west,tile_bbox.south,tile_bbox.east,tile_bbox.north)

def tilebboxes_from_bbox(minlat,minlon,maxlat,maxlon,zoom=ZOOM_LEVEL,as_list=False):
    """
    Generate a list of tile bounding boxes from a given bounding box.

    Args:
        minlat (float): The minimum latitude of the bounding box.
        minlon (float): The minimum longitude of the bounding box.
        maxlat (float): The maximum latitude of the bounding box.
        maxlon (float): The maximum longitude of the bounding box.
        zoom (int, optional): The zoom level of the tiles. Defaults to ZOOM_LEVEL.
        as_list (bool, optional): Whether to return the tile bounding boxes as lists. Defaults to False.

    Returns:
        list: A list of tile bounding boxes. Each element is a tuple representing the south, west, north, and east coordinates of the tile. If as_list is True, each element is a list containing the same coordinates.
    """
    if as_list:
        return [list(mercantile.bounds(tile)) for tile in mercantile.tiles(minlon,minlat,maxlon,maxlat,zoom)]
    else:
        return [mercantile.bounds(tile) for tile in mercantile.tiles(minlon,minlat,maxlon,maxlat,zoom)]

def check_type_by_first_valid(input_iterable):
    """
    Check the type of the first non-null item in the input iterable.

    Args:
        input_iterable (Iterable): The iterable to check.

    Returns:
        type: The type of the first non-null item in the input iterable.

    Raises:
        ValueError: If the input iterable is empty.

    """
    for item in input_iterable:
        if item:
            return type(item)

def selected_columns_to_str(df,desired_type=list):
    """
    Converts the values of selected columns in a DataFrame to strings.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        desired_type (type, optional): The type to check for in the DataFrame columns. Defaults to list.

    Returns:
        None

    Raises:
        None
    """
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])
        
        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))

def dump_json(data, path):
    """
    Writes the given data to a JSON file specified by the path.

    Args:
        data: The data to be written to the JSON file.
        path: The path to the JSON file.

    Returns:
        None
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(path):
    """
    Reads a JSON file from the specified path and returns its contents as a Python object.

    Parameters:
        path (str): The path to the JSON file.

    Returns:
        dict or list: The contents of the JSON file as a Python object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file cannot be decoded as JSON.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_coordinates_as_point(inputdict):
    """
    Converts input coordinates from a dictionary to a Point object.

    Parameters:
        inputdict (dict): A dictionary containing the coordinates.

    Returns:
        Point: A Point object with the specified coordinates.
    """
    return Point(inputdict['coordinates'])


def convert_timestamp_to_date(metadata):
    """
    Converts the 'captured_at' timestamp in the 'data' field of the provided metadata dictionary to a formatted date string.

    Parameters:
        metadata (dict): A dictionary containing metadata information, including a 'data' field with feature dictionaries.

    Returns:
        dict: The modified metadata dictionary with the 'captured_at' timestamp converted to a formatted date string.
    """
    for feature in metadata.get('data', []):
        if 'captured_at' in feature:
            timestamp = feature['captured_at']
            feature['captured_at'] = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return metadata


def get_mapillary_images_metadata(minLon, minLat, maxLon, maxLat, token=MAPPILARY_TOKEN,outpath=None,limit=5000):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.

    Returns:
        dict: A dictionary containing the response from the API.
    """
    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        'limit': limit,
        "access_token": token,
        "fields": ",".join([
            "altitude", 
            "atomic_scale", 
            "camera_parameters", 
            "camera_type", 
            "captured_at",
            "compass_angle", 
            "computed_altitude", 
            "computed_compass_angle", 
            "computed_geometry",
            "computed_rotation", 
            "creator", 
            "exif_orientation", 
            "geometry", 
            "height", 
            # "is_pano",
            "make", 
            "model", 
            # "thumb_256_url", 
            # "thumb_1024_url", 
            # "thumb_2048_url",
            "thumb_original_url", 
            "merge_cc", 
            # "mesh", 
            "sequence", 
            # "sfm_cluster", 
            "width",
            # "detections",
        ])
    }
    response = requests.get(url, params=params)

    as_dict = response.json()

    # Convert timestamps to readable dates
    as_dict = convert_timestamp_to_date(as_dict)

    if outpath:
        dump_json(as_dict, outpath)

    return as_dict

def radius_to_degrees(radius,lat):
    """
    Convert a radius in meters to degrees.  
    """
    return radius / (111320 * cos(lat * pi / 180))

def degrees_to_radius(degrees, lat):
    """
    Convert a radius in degrees to meters.  
    """
    return degrees * 111320 * cos(lat * pi / 180)

def get_bounding_box(lon, lat, radius):
    """
    Return a bounding box tuple as (minLon, minLat, maxLon, maxLat) from a pair of coordinates and a radius, using shapely.

    Parameters:
        lon (float): The longitude of the center of the bounding box.
        lat (float): The latitude of the center of the bounding box.
        radius (float): The radius of the bounding box in meters.

    Returns: 
        tuple: A tuple containing the minimum and maximum longitude and latitude of the bounding box.
    """


    # Convert radius from meters to degrees
    radius_deg = radius_to_degrees(radius, lat)

    point = Point(lon, lat)
    return box(point.x - radius_deg, point.y - radius_deg, point.x + radius_deg, point.y + radius_deg).bounds

# function to download an image from a url:
def download_mapillary_image(url, outfilepath,cooldown=1):
    """
    Downloads an image from a given URL and saves it to the specified output file path.

    Parameters:
        url (str): The URL of the image to download.
        outfilepath (str): The path where the downloaded image will be saved.
        cooldown (int, optional): The number of seconds to wait before downloading the next image. Defaults to 1.

    Raises:
        Exception: If there is an error during the download process.

    Returns:
        None
    """
    try:
        wget.download(url, out=outfilepath)

        if cooldown:
            sleep(cooldown)
    except Exception as e:
        print('error:',e)

def download_all_pictures_from_gdf1(gdf, outfolderpath,id_field='id',url_field='thumb_original_url'):
    """
    Downloads all the pictures from a GeoDataFrame (gdf) and saves them to the specified output folder.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        outfolderpath (str): The path to the output folder where the pictures will be saved.
        id_field (str, optional): The name of the field in the GeoDataFrame that contains the unique identifier for each picture. Default is 'id'.
        url_field (str, optional): The name of the field in the GeoDataFrame that contains the URL of the picture. Default is 'thumb_original_url'.

    Returns:
        None
    """
    for row in tqdm(gdf.itertuples(),total=len(gdf)):
        try:
            download_mapillary_image(getattr(row,url_field),os.path.join(outfolderpath,getattr(row,id_field)+'.jpg'))
        except Exception as e:
            print('error:',e)

def download_all_pictures_from_gdf(gdf, outfolderpath,id_field='id',url_field='thumb_original_url'):
    """
    Downloads all the pictures from a GeoDataFrame (gdf) and saves them to the specified output folder.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        outfolderpath (str): The path to the output folder where the pictures will be saved.
        id_field (str, optional): The name of the field in the GeoDataFrame that contains the unique identifier for each picture. Default is 'id'.
        url_field (str, optional): The name of the field in the GeoDataFrame that contains the URL of the picture. Default is 'thumb_original_url'.

    Returns:
        None
    """
    for row in tqdm(gdf.itertuples(), total=len(gdf)):
        try:
            study_case = getattr(row, 'study_case')
            picture_id = getattr(row, id_field)
            url = getattr(row, url_field)

            # Formatar o caminho de saída para a imagem
            image_path = os.path.join(outfolderpath, study_case, f"{picture_id}.jpg")

            # Verificar se a imagem já existe
            if not os.path.exists(image_path):
                # Fazer o download da imagem
                download_mapillary_image(url, image_path)
            else:
                print(f"Imagem já existe: {image_path}")

        except Exception as e:
            print('error:', e)

def mapillary_data_to_gdf(data,outpath=None,filtering_polygon=None):
    """
    Convert Mapillary data to a GeoDataFrame.

    Args:
        data (dict): The Mapillary data to convert. It should have a 'data' key containing a list of records.
        outpath (str, optional): The path to save the resulting GeoDataFrame as a file. Defaults to None.
        filtering_polygon (GeoDataFrame or GeoSeries, optional): A polygon to filter the resulting GeoDataFrame by. Defaults to None.

    Returns:
        GeoDataFrame: The converted GeoDataFrame. If the input data does not have a 'data' key, an empty GeoDataFrame is returned.
    """
    if data.get('data'):

        as_df = pd.DataFrame.from_records(data['data'])

        as_df.geometry = as_df.geometry.apply(get_coordinates_as_point)

        as_gdf = gpd.GeoDataFrame(as_df,crs='EPSG:4326',geometry='geometry')

        selected_columns_to_str(as_gdf)

        if filtering_polygon:
            as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)]

        if outpath:
            as_gdf.to_file(outpath)

        return as_gdf
    else:
        return gpd.GeoDataFrame()

def tiled_mapillary_data_to_gdf(input_polygon, token,zoom=ZOOM_LEVEL,outpath=None):
    """
    Convert tiled mapillary data to a GeoDataFrame.

    Args:
        input_polygon: The input polygon to extract data from.
        token: The token for authentication.
        zoom: The zoom level for tile retrieval (default is ZOOM_LEVEL).
        outpath: The output path for saving the GeoDataFrame (default is None).

    Returns:
        GeoDataFrame: The converted GeoDataFrame. If the input data does not have a 'data' key, an empty GeoDataFrame is returned.
    """
    # get the bbox of the input polygon:
    minLon, minLat, maxLon, maxLat = input_polygon.bounds

    # get the bboxes of the tiles:
    bboxes = tilebboxes_from_bbox(minLat, minLon, maxLat, maxLon, zoom)

    # get the metadata for each tile:
    gdfs_list = []

    for bbox in tqdm(bboxes):
    # for i, bbox in enumerate(tqdm(bboxes)):

        # get the tile as geometry:
        bbox_geom = tile_bbox_to_box(bbox)


        # check if the tile intersects the input polygon:
        if not bbox_geom.disjoint(input_polygon):
            # get the metadata for the tile:
            data = get_mapillary_images_metadata(*resort_bbox(bbox),token) #,outpath=f'tests\small_city_tiles\{i}.json')

            if data.get('data'):
                # convert the metadata to a GeoDataFrame:
                    gdfs_list.append(mapillary_data_to_gdf(data,outpath,input_polygon))


    # concatenate the GeoDataFrames:
    as_gdf = pd.concat(gdfs_list)

    if outpath:
        as_gdf.to_file(outpath)

    return as_gdf


def resort_bbox(bbox):
    """
    A function that resorts the bounding box coordinates.

    Args:
        bbox (list): The input bounding box coordinates [minLon, minLat, maxLon, maxLat].

    Returns:
        list: The bounding box coordinates in the order [minLat, minLon, maxLat, maxLon].
    """
    return [bbox[1],bbox[0],bbox[3],bbox[2]]

def get_territory_polygon(place_name,outpath=None):
    """
    Retrieves the polygon of a territory based on the provided place name from the Nominatim API.

    Args:
        place_name (str): The name of the place to retrieve the territory polygon for.
        outpath (str, optional): The path to save the polygon GeoJSON file. Defaults to None.

    Returns:
        dict or None: The polygon GeoJSON object representing the territory, or None if no polygon is found.
    """
    # Make a request to Nominatim API with the place name
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "polygon_geojson": 1}
    response = requests.get(url, params=params)

    # Parse the response as a JSON object
    data = response.json()
    
    # sort data by "importance", that is a key in each dictionary of the list:
    data.sort(key=lambda x: x["importance"], reverse=True)
    
    # removing all non-polygon objects:
    data = [d for d in data if d['geojson']['type'] == 'Polygon']

    # Get the polygon of the territory as a GeoJSON object
    if data:
        polygon = data[0]['geojson']

        if outpath:
            dump_json(polygon, outpath)

        # Return the polygon
        return polygon

def process_mapillary_data(gdf_roi, base_path):
    """
    Process Mapillary data for each row in the given GeoDataFrame `gdf_roi`.
    
    Args:
        gdf_roi (GeoDataFrame): A GeoDataFrame containing the region of interest (ROI) for each study case.
        base_path (str): The base path where the output files will be saved.
        
    Returns:
        GeoDataFrame or None: A GeoDataFrame containing the Mapillary metadata for each study case, or None if no data is found.
    
    Raises:
        Exception: If there is an error processing a study case.
    
    Description:
        This function iterates over each row in the given GeoDataFrame `gdf_roi`. For each row, it retrieves the study case name,
        calculates the buffer bounding box, and formats the output paths for the metadata and GeoJSON files. It then calls the
        `get_mapillary_images_metadata` function to obtain the Mapillary images metadata within the buffer bounding box, and saves
        the metadata to the specified metadata output path. It then calls the `mapillary_data_to_gdf` function to convert the
        Mapillary metadata to a GeoDataFrame and saves it to the specified GeoJSON output path. If the resulting GeoDataFrame is
        not empty, it prints a message indicating that the process was successful. Otherwise, it prints a message indicating that
        no data was found for the study case. If any error occurs during the process, it prints an error message with the study
        case name and the specific error.
    """
    for idx, row in gdf_roi.iterrows():
        case = row['study_case']
        buffer_bbox = row.geometry.bounds
        
        # Formatar o caminho de saída para os arquivos
        metadata_outpath = os.path.join(base_path, case, f'{case}_metadata.json')
        geojson_outpath = os.path.join(base_path, case, f'{case}_metadata.geojson')

        try:
            # Chamar a função para obter os metadados do Mapillary
            mappillary_md = get_mapillary_images_metadata(*buffer_bbox, outpath=metadata_outpath)

            # Chamar a função para converter os dados do Mapillary para GeoDataFrame
            mappillary_md_gdf = mapillary_data_to_gdf(mappillary_md, outpath=metadata_outpath)

            if not mappillary_md_gdf.empty:
                # Adicionar o campo 'study_case' ao GeoDataFrame de metadados
                mappillary_md_gdf['study_case'] = case

                # Salvar o GeoDataFrame em GeoJSON
                mappillary_md_gdf.to_file(geojson_outpath, driver='GeoJSON')

                print(f"Processado {case}")
            else:
                print(f"Nenhum dado encontrado para {case}")
        
        except Exception as e:
            print(f"Erro ao processar {case}: {e}")

    return mappillary_md_gdf

def scan_images_and_save_to_csv(outfolderpath, output_csv_path):
    """
    Scans all .jpg images in the given outfolderpath and saves the result to a CSV file.
    
    Parameters:
        outfolderpath (str): The path to the folder where the images are saved.
        output_csv_path (str): The path to the output CSV file.
        
    Returns:
        None
    """
    image_data = []

    # Percorre todos os diretórios e arquivos em outfolderpath
    for root, dirs, files in os.walk(outfolderpath):
        for file in files:
            if file.endswith('.jpg'):
                image_id = os.path.splitext(file)[0]  # Extrai o id do nome do arquivo (sem extensão)
                full_path = os.path.join(root, file)
                image_data.append({'id': image_id, 'path': full_path})
    
    # Cria um DataFrame a partir dos dados da imagem
    df = pd.DataFrame(image_data)
    
    # Salva o DataFrame em um arquivo CSV
    df.to_csv(output_csv_path, index=False)
    print(f"CSV salvo em: {output_csv_path}")