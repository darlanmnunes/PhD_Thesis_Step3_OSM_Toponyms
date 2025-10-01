"""
This module contains functions applied to Toponyms correspondence analysis
from SLI and OSM.
"""

# Libraries
import warnings
warnings.filterwarnings('ignore')

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import folium
import Levenshtein
import matplotlib.pyplot as plt
from shapely import wkt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from unidecode import unidecode
from shapely.geometry import Point
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap


# Functions to visualize the spatial distribution os POIs 
def calculate_centroid(geometry):
    """
    Calculate the centroid of a given geometry.

    Parameters:
        geometry (shapely.geometry.base.BaseGeometry): The input geometry.

    Returns:
        List[float]: A list containing the y and x coordinates of the centroid.
    """

    centroid = geometry.centroid
    return [centroid.y, centroid.x]

def plot_filtered_data(gdf,centroid_coords):
    """
    Plots the filtered GeoDataFrame on a map using the Folium library.

    Parameters:
        gdf (GeoDataFrame): The filtered GeoDataFrame to be plotted.

    Returns:
        None

    Description:
        This function initializes a map centered on the centroid of the filtered GeoDataFrame.
        It adds the filtered GeoDataFrame as a GeoJSON layer to the map, with custom tooltips
        and styling.

        The map is then displayed using the `display` function.

    Note:
        The `centroid_coords` variable is expected to be defined in the global scope.
    """

    #initiate the map
    m = folium.Map(location=centroid_coords, tiles='OpenStreetMap', zoom_start=14)

    #add the filtered GeoDataFrame as a GeoJSON layer
    folium.GeoJson(
        gdf,
        name='Casos Filtrados',
        tooltip=folium.GeoJsonTooltip(fields=['study_case', 'name']),
        style_function=lambda x: {'fillColor': '#8C8989', 'color': '#e31a1c', 'weight': 2}
    ).add_to(m)

    #display the map
    display(m)


# Function to load CSV files
def load_csv_files(*file_paths):
    """
    Load multiple CSV files into a list of pandas DataFrames.

    Parameters:
        *file_paths (str): Variable length argument list of file paths to CSV files.

    Returns:
        list: A list of pandas DataFrames, each corresponding to a CSV file.
    """

    dataframes = [pd.read_csv(file) for file in file_paths]
    return dataframes

# Function to preprocess the text in the column of the DataFrame
def preprocess_dataframe(df, column_name):
    """
    Preprocess the text in the specified column of the DataFrame.

    This function applies the following preprocessing steps to the text in the specified column:
    1. Converts the text to lowercase.
    2. Removes accents.
    3. Removes special characters, numbers, and symbols.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be preprocessed.

    Returns:
        pandas.DataFrame: The input DataFrame with the preprocessed text in the specified column.
    """

    def preprocess_text(text):
        """
        Preprocess a string by converting it to lowercase, removing accents, and removing special characters, numbers, and symbols.

        Parameters:
            text (str): The string to be preprocessed.

        Returns:
            str: The preprocessed string.
        """

        # Convert to lowercase
        text = text.lower()
        
        # Remove accents
        text = unidecode(text)
        
        # Remove special characters, numbers, and symbols
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text
    
    # Apply the preprocessing function
    df[f'cleaned_{column_name}'] = df[column_name].apply(preprocess_text)
    
    return df

def explode_cleaned_names(df):

    """
    Explode a DataFrame with cleaned names into multiple rows, each containing a term from the cleaned name.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with columns 'study_case', 'name', 'geometry', and 'cleaned_name'.

    Returns:
        pandas.DataFrame: The exploded DataFrame with additional columns 'subcase_id' and 'cleaned_name_subterm'.
    """

    expanded_rows = []

    for _, row in df.iterrows():
        osmId_formatado = row['osmId_formatado']
        name_osm = row['name_osm']
        geometry = row['geometry']
        cleaned_name_osm = row['cleaned_name_osm']

        # Split cleaned_name into terms/words
        terms = cleaned_name_osm.split()

        # Create a row for each term
        for idx, term in enumerate(terms, start=1):
            # Identificador: Exemplo 'case5.1', 'case5.2'...
            subcase_id = f"{osmId_formatado}.{idx}"

            expanded_rows.append({
                "osmId_formatado": osmId_formatado,
                "subcase_id": subcase_id,
                "name_osm": name_osm,  
                "cleaned_name_osm": cleaned_name_osm,  
                "cleaned_name_subterm": term,
                "geometry": geometry
            })

    # Create a DataFrame from the expanded rows
    exploded_df = pd.DataFrame(expanded_rows)

    return exploded_df

def explode_cleaned_names2(gdf, id_col="osmId_formatado", cleaned_col="cleaned_name_osm"):
    """
    Explode o DF/GDF em múltiplas linhas, uma por termo do 'cleaned_name_osm',
    preservando TODAS as colunas originais e adicionando:
      - 'cleaned_name_subterm' (cada termo)
      - 'subcase_id' = f"{osmId_formatado}.{idx}"

    Parâmetros
    ----------
    df : (Geo)DataFrame
        Deve conter as colunas `id_col` (padrão: 'osmId_formatado') e `cleaned_col` (padrão: 'cleaned_name_osm').
    id_col : str
        Nome da coluna com o identificador base para o subcase_id.
    cleaned_col : str
        Nome da coluna com o texto já limpo a ser explodido em termos.

    Retorno
    -------
    gdf_out : (Geo)DataFrame
        Mesmo schema do df de entrada + colunas 'cleaned_name_subterm' e 'subcase_id'.
        Mantém geometry e CRS se o input for GeoDataFrame.
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    if id_col not in gdf.columns:
        raise KeyError(f"Coluna obrigatória ausente: '{id_col}'")
    if cleaned_col not in gdf.columns:
        raise KeyError(f"Coluna obrigatória ausente: '{cleaned_col}'")

    is_geo = isinstance(gdf, gpd.GeoDataFrame)
    geom_col = gdf.geometry.name if is_geo else None
    crs_in = gdf.crs if is_geo else None

    tmp = gdf.copy()

    # índice original para enumerar termos por linha original
    tmp["_orig_idx"] = np.arange(len(tmp))

    # normaliza e separa termos; linhas sem termos serão descartadas
    cleaned = (
        tmp[cleaned_col]
        .astype(str)
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tmp["__terms__"] = cleaned.str.split(" ")

    # explode (uma linha por termo)
    exploded = tmp.explode("__terms__", ignore_index=False)

    # remove termos vazios (quando cleaned_name_osm era vazio)
    exploded = exploded[exploded["__terms__"].notna() & (exploded["__terms__"].str.len() > 0)]

    # enumeração local por linha original
    exploded["_term_idx"] = exploded.groupby("_orig_idx").cumcount() + 1

    # novas colunas
    exploded["cleaned_name_subterm"] = exploded["__terms__"]
    exploded["subcase_id"] = exploded[id_col].astype(str) + "." + exploded["_term_idx"].astype(str)

    # limpa auxiliares
    exploded = exploded.drop(columns=["__terms__", "_term_idx", "_orig_idx"])

    # retorna no mesmo tipo do input
    if is_geo:
        gdf_out = gpd.GeoDataFrame(exploded, geometry=geom_col, crs=crs_in)
    else:
        gdf_out = exploded

    return gdf_out


#### -------------------------------------------------------------------------------
# Toponyms Correspondence Analysis

# Levenshtein functions
# Function to calculate the Levenshtein distance
def compute_levenshtein_distance(str1, str2):
    """
    A function to compute the Levenshtein distance between two input strings.

    :param str1: The first input string.
    :param str2: The second input string.
    :return: The Levenshtein distance between the two input strings.
    """
    return Levenshtein.distance(str1, str2)

# Function to calculate the similarity percentage based on the Levenshtein distance
def compute_similarity_percentage(str1, str2):
    """
    Calculates the similarity percentage between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        float: The similarity percentage between the two strings.

    Raises:
        None.

    Examples:
        >>> compute_similarity_percentage("apple", "banana")
        0.0
        >>> compute_similarity_percentage("kitten", "sitting")
        57.142857142857146
        >>> compute_similarity_percentage("", "")
        100.0
    """

    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 100.0
    
    distance = Levenshtein.distance(str1, str2)
    similarity = (1 - distance / max_len) * 100
    return similarity

# Version 1 with improvements of the levenshtein_analysis function
def levenshtein_analysis1(df_text, gdf, text_column, study_case_column, case_column, column_name):
    """
    Calculate the Levenshtein distance and similarity percentage for each pair of unique names in the given GeoDataFrame and text DataFrame.

    Args:
        df_text (pandas.DataFrame): The text DataFrame containing the text column.
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the study_case, name, and geometry columns.
        text_column (str): The name of the text column in the text DataFrame.
        study_case_column (str): The name of the study_case column in the GeoDataFrame.
        case_column (str): The name of the case column in the text DataFrame.
        column_name (str): The name of the cleaned text column in the GeoDataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the study_case, name, text, Levenshtein distance, and similarity percentage for each pair of unique names.
    """

    results = []

    for case in gdf[study_case_column].unique():
        gdf_case = gdf[gdf[study_case_column] == case]
        df_case = df_text[df_text[case_column] == case]

        if not df_case.empty and not gdf_case.empty:
            for name in gdf_case[column_name].unique():
                for text in df_case[text_column].dropna().unique():
                    distance = compute_levenshtein_distance(str(name), str(text))
                    similarity = compute_similarity_percentage(str(name), str(text))
                    results.append((case, name, text, distance, similarity))

    return pd.DataFrame(results, columns=[study_case_column, column_name, text_column, 'levenshtein_distance', 'similarity_percentage'])


# Version 2 with improvements of the levenshtein_analysis function

def levenshtein_analysis2(df_text, gdf, text_column, study_case_column, case_column, column_name):
    """
    Calculate the Levenshtein distance and similarity percentage for each pair of unique names in the given GeoDataFrame and text DataFrame.

    Args:
        df_text (pandas.DataFrame): The text DataFrame containing the text column from SLI
        gdf (geopandas.GeoDataFrame): OSM terms GeoDataFrame containing study_case, cleaned_name_subterm, subcase_id, etc.
        text_column (str): The name of the text column in the text DataFrame
        study_case_column (str): The name of the study_case column in the GeoDataFrame
        case_column (str): Case identifier in df_text (should match study_case_column values)
        column_name (str): Name of the subterm in gdf (e.g., 'cleaned_name_subterm')

    Returns:
        pandas.DataFrame: A DataFrame containing all columns from the original DataFrames, with Levenshtein distance and similarity percentage for each pair of unique names.
    """

    results = []

    # Loop through each study case
    for case in gdf[study_case_column].unique():
        gdf_case = gdf[gdf[study_case_column] == case]
        df_case = df_text[df_text[case_column] == case]
        # Check if both subsets are non-empty
        if not gdf_case.empty and not df_case.empty:
            # for each subcase_id (case) in the exploded dataframe

            for name in gdf_case[column_name].unique():
                for text in df_case[text_column].unique():
                    # Check for non-empty text and name before calculation
                    if pd.notna(name) and pd.notna(text):
                        distance = compute_levenshtein_distance(str(name), str(text))
                        similarity = compute_similarity_percentage(str(name), str(text))
                    else:
                        distance = None
                        similarity = None
                    results.append((case, name, text, text, distance, similarity))
        else:
            # Add empty metrics for rows with no matching entries
            for name in gdf_case[column_name].unique():
                for text in df_case[text_column].unique():
                    results.append((case, name, text, text, None, None))

    # Create a results DataFrame with calculated Levenshtein metrics and the additional 'Text_detect_sli' column
    result_df = pd.DataFrame(results, columns=[study_case_column, column_name, text_column, 'Text_detect_sli', 'levenshtein_distance', 'similarity_percentage'])

    # Renomear coluna 'Case' para 'study_case' em df_text antes do merge
    df_text = df_text.rename(columns={case_column: study_case_column})

    # Remover a coluna 'geometry' de gdf antes do merge para evitar duplicação
    gdf = gdf.drop(columns=['geometry'])

    # Merge result_df with original DataFrames to retain all metadata columns
    merged_gdf = gdf.merge(result_df, on=[study_case_column, column_name], how='left')
    final_df = df_text.merge(merged_gdf, on=[study_case_column, text_column], how='left')

    return final_df


# Graphs from the levenshtein metrics
# Função para plotar e salvar histogramas de similaridade
def plot_similarity_histogram(df, title):
    """
    Plots a histogram of the 'similarity_percentage' column in the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'similarity_percentage' column.
        title (str): The title of the histogram.

    Returns:
        matplotlib.figure.Figure: The figure object of the plotted histogram.
    """

    fig = plt.figure(figsize=(10, 6))
    plt.hist(df['similarity_percentage'], bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel('Similaridade (%)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()
    return fig

# Violin Graph 
def plot_similarity_violin(df, title):
    """
    Plot a violin graph showing the similarity percentage for different study cases.

    Parameters:
        df (DataFrame): The DataFrame containing the data to plot.
        title (str): The title of the plot.

    Returns:
        figure: The figure object of the plotted graph.
    """
    
    fig = plt.figure(figsize=(10, 6))
    
    #order study cases
    df['study_case'] = pd.Categorical(df['study_case'], categories=sorted(df['study_case'].unique(), key=lambda x: int(x[4:])))
    
    #create violin plot
    sns.violinplot(x='study_case', y='similarity_percentage', data=df, inner=None, palette='Blues')
    
    #add statistics values highlighted with boxplots
    sns.boxplot(x='study_case', y='similarity_percentage', data=df, whis=np.inf, palette='Blues', fliersize=0, width=0.2, boxprops=dict(alpha=0.3))
    
    plt.title(title, fontsize=16)
    plt.xlabel('Study Case', fontsize=14)
    plt.ylabel('Similarity (%)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Definir o limite inferior do eixo y para 0 e o máximo com base nos dados
    ymin = 0
    ymax = 100
    plt.ylim(ymin, ymax + 10)  # Adiciona um pequeno padding no limite superior
    
    # Definir os steps do eixo y em 25%
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    
    plt.grid(True)
    plt.show()
    return fig

# Box Plot
def plot_box(df, y_column, title):
    """
    Plot a box plot for the given y_column with specified palette and outline only.
    BoxPlot: shows the data distribution with the median, quartiles, and outliers.

    Parameters:
        df (DataFrame): The DataFrame containing the data to plot.
        y_column (str): The column name for the y-axis.
        title (str): The title of the plot.

    Returns:
        figure: The figure object of the plotted graph.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordenar os casos
    df['study_case'] = pd.Categorical(df['study_case'], categories=sorted(df['study_case'].unique(), key=lambda x: int(x[4:])))
    
    # Criar o box plot com contornos apenas
    sns.boxplot(x='study_case', y=y_column, data=df, palette='colorblind', fliersize=0, ax=ax, 
                boxprops=dict(facecolor='none'),
                whiskerprops=dict(linestyle='--', color='gray'))
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Study Case', fontsize=14)
    ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=14)
    plt.xticks(rotation=45)
    
    #setup grid
    #ax.grid(False, which='both', axis='y', color='gray', linestyle='-', linewidth=0.5)
    ax.grid(False, axis='y')
    ax.grid(False, axis='x')
    
    plt.show()
    return fig

# Ridgeline plot with inside plot and annotations
    # Ref: https://r-graph-gallery.com/web-ridgeline-plot-with-inside-plot-and-annotations.html
def plot_density_with_annotations(df, column_value, title):
    """
    Plot a density plot for the given column_value to show density distributions with annotations.

    Parameters:
        df (DataFrame): The DataFrame containing the data to plot.
        column_value (str): The column name for the x-axis.
        title (str): The title of the plot.

    Returns:
        figure: The figure object of the plotted graph.
    """
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")

    # Create a custom colormap for the gradient green
    colors = sns.color_palette("Greens", 256)
    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

    # Main plot with density distributions
    ax = sns.violinplot(x=column_value, y='study_case', data=df, inner=None, color='lightgray', cut=0)

    # Add the gradient green bars manually
    for study_case in df['study_case'].unique():
        subset = df[df['study_case'] == study_case]
        iqr_25 = subset[column_value].quantile(0.25)
        iqr_75 = subset[column_value].quantile(0.75)
        x_positions = np.linspace(iqr_25, iqr_75, 256)
        y_position = np.where(df['study_case'].unique() == study_case)[0][0]
        ax.fill_betweenx([y_position - 0.4, y_position + 0.4], iqr_25, iqr_75, color=cmap(np.linspace(0, 1, 256)))

    sns.pointplot(x=column_value, y='study_case', data=df, join=False, color='black', markers="d", ci='sd')

    # Add annotations for each study case
    for i, study_case in enumerate(df['study_case'].unique()):
        subset = df[df['study_case'] == study_case]
        median = subset[column_value].median()

        ax.annotate(f'{median:.1f}', xy=(median, i), 
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center', 
                    fontsize=10, color='black', backgroundcolor='white', fontweight='normal')
        ax.scatter(median, i, color='black', zorder=5)
    
    # Overall median
    overall_median = df[column_value].median()
    ax.axvline(overall_median, color='gray', linestyle='--', linewidth=1)
    ax.annotate('Overall Median', xy=(overall_median, -0.5), xytext=(10, -5), textcoords='offset points', 
                ha='center', va='center', fontsize=10, color='black', backgroundcolor='white')

    # Adding text and annotations
    plt.title(title, fontsize=16)
    plt.xlabel(column_value.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Study Case', fontsize=14)

    # Custom Legend (Inset)
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='lightgray', alpha=0.5, label='Median and IQR'),
        Patch(facecolor='green', edgecolor='green', alpha=0.6, label='Density Distribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return ax.get_figure()

# Function to salve figures with 300 dpi
def save_figure(fig, file_path, dpi=300):

    """
    Save a matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        file_path (str): The path to save the figure.
        dpi (int, optional): The resolution in dots per inch. Default is 300.
    """
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')


#### -------------------------------------------------------------------------------
# Evidence from subterms 

# Aggregate Similarity Score (Average Weighted by Word Length)
def calculate_wighted_score(df):
    """
    Calculate a weighted similarity score for each study case.

    Parameters:
        df (pandas.DataFrame): DataFrame with columns 'study_case', 'cleaned_name_subterm', and 'similarity_percentage'.

    Returns:
        pandas.DataFrame: DataFrame with columns 'study_case' and 'wighted_score', where 'wighted_score' is the weighted sum of
            the similarity scores divided by the total weight (sum of the word lengths) for each study case.
    """

    df['weight'] = df['cleaned_name_subterm'].apply(len)

    score_ponderado_df = (
        df.groupby('osmId_formatado')
        .apply(lambda x: pd.Series({
            'wighted_sum': (x['similarity_percentage'] * x['weight']).sum(),
            'total_weight': x['weight'].sum()
        }))
        .reset_index()
    )

    score_ponderado_df['wighted_score'] = (
        score_ponderado_df['wighted_sum'] / score_ponderado_df['total_weight']
    )

    return score_ponderado_df[['osmId_formatado', 'wighted_score']]


# ICTVAE
def calculate_ictvae(df_levenshtein, min_similarity=0.0):
    """
    Calcula a confiança por evidência acumulada com base no dataframe de resultados das métricas de Levenshtein.
    Considera o número de subtermos detectados, a qualidade das similaridades e o peso pelo tamanho da palavra.
    Agora inclui fator de cobertura baseado na detecção relevante de subtermos.

    Args:
        df_levenshtein (DataFrame): DataFrame contendo as colunas:
            - study_case
            - subcase_id
            - name (nome completo)
            - cleaned_name (topônimo OSM limpo)
            - cleaned_name_subterm (subtermo do nome OSM)
            - similarity_percentage

    Returns:
        DataFrame com as métricas de confiança por evidência acumulada para cada study_case.
    """

    # Step 1: Get the best similarity for each subterm
    best_sim_subterm = (
        df_levenshtein.groupby(['osmId_formatado', 'subcase_id', 'cleaned_name_subterm'])
        .agg({'similarity_percentage': 'max'})
        .reset_index()
    )

    # Step 2: Add weight based on subterm length
    best_sim_subterm['weight'] = best_sim_subterm['cleaned_name_subterm'].apply(len)

    # Step 3: Define criteria for relevant subterm detection
     # # Considers detected if there was any similarity value greater than zero or any value defined by min_similarity
    best_sim_subterm['detected'] = best_sim_subterm['similarity_percentage'] > min_similarity

    # Passo 4 - Agrupar por study_case para consolidar métricas
    confianca_df = (
        best_sim_subterm.groupby('osmId_formatado')
        .apply(lambda x: pd.Series({
            'total_subterms_n': len(x),
            'detected_subterms_D': x['detected'].sum(),  # How many subterms were detected?
            'coverage_ratio': x['detected'].sum() / len(x) if len(x) > 0 else 0,  # % of coverage
            'total_weight': x['weight'].sum(),  # Sum of all subterm sizes
            'sum_similarity_weight': (x['similarity_percentage'] * x['weight']).sum()  # Similarity weighted by the size of the term
        }))
        .reset_index()
    )

    # Step 5: Score final (ICTVAE)
    confianca_df['ICTVAE'] = (
        (confianca_df['sum_similarity_weight'] / confianca_df['total_weight']) *
        confianca_df['coverage_ratio']
    )

    # Passo 6 - Organize the final dataframe
    resultado_df = confianca_df[[
        'osmId_formatado',
        'total_subterms_n',
        'detected_subterms_D',
        'coverage_ratio',
        'ICTVAE'
    ]].copy()

    return resultado_df
