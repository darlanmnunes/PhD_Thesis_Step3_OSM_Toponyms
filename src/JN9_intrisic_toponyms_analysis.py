# %% [markdown]
# # Jupyter notebook 09: Analysis of Collaborative Toponyms in OpenStreetMap using Intrinsic Parameters

# %% [markdown]
# **Applying the methods presented in the Paper:**
# 
#  *- Collaborative Toponyms in OpenStreetMap: an open-source framework to investigate the relationship with intrinsic quality parameters*

# %% [markdown]
# ## Install and import the necessary libraries

# %%
# Import library and some pre-installed modules
import os
import sys
import numpy as np
import json
import folium
import pandas as pd
import geopandas as gpd
import mapclassify
import matplotlib.colors
import seaborn as sns
import statsmodels.api as sm
import warnings
import pyproj
import branca.colormap as cm
from matplotlib import pyplot as plt
from ipywidgets import widgets, Layout, Button, interact, Dropdown, SelectMultiple, HBox, VBox, Output
from IPython.display import display, clear_output, FileLink
from folium import plugins, Map, Element, Figure, LayerControl, TileLayer
from folium.features import GeoJson, GeoJsonTooltip,Choropleth, CircleMarker
from pysal.explore import esda
from pysal.lib import weights
from shapely.geometry import MultiPolygon, box
from shapely.ops import transform
from jinja2 import Template
from tqdm.notebook import tqdm
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from esda import G_Local
from esda.moran import Moran
from splot.esda import plot_moran
from statsmodels.stats.outliers_influence import variance_inflation_factor
%matplotlib inline

# %%
# Sets the root directory of the project as the working directory
os.chdir('..')

# %%
# Check the current working directory
os.getcwd()

# %%
# Check the current directories in the working directory
%ls

# %% [markdown]
# ## Analysis of Collaborative Toponyms in OpenStreetMap

# %% [markdown]
# #### PostGIS - Open the database connection

# %%
# Conexão ao Banco PostGIS
import psycopg2

# Function to load database credentials from a text file
def load_credentials_from_txt(file_path):
    credentials = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    credentials[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Arquivo de credenciais não encontrado: {file_path}")
    except Exception as e:
        print(f"Erro ao ler credenciais: {e}")
    return credentials

def connect_to_postgis(txt_path='configs/db_credentials.txt'):
    creds = load_credentials_from_txt(txt_path)

    required_keys = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
    if not all(k in creds for k in required_keys):
        print("Credenciais incompletas no arquivo de configuração.")
        return None

    try:
        conn = psycopg2.connect(
            dbname=creds['DB_NAME'],
            user=creds['DB_USER'],
            password=creds['DB_PASSWORD'],
            host=creds['DB_HOST'],
            port=creds['DB_PORT']
        )
        print("Conexão ao PostGIS estabelecida com sucesso!")
        return conn

    except Exception as e:
        print(f" Erro ao conectar ao PostGIS: {e}")
        return None

# %%
# Open the database connection
conn = connect_to_postgis()

# %%
# Criar uma engine do SQLAlchemy
from sqlalchemy import create_engine

def create_sqlalchemy_engine_from_txt(file_path='configs/db_credentials.txt'):
    creds = load_credentials_from_txt(file_path)
    conn_str = f"postgresql://{creds['DB_USER']}:{creds['DB_PASSWORD']}@{creds['DB_HOST']}:{creds['DB_PORT']}/{creds['DB_NAME']}"
    engine = create_engine(conn_str)
    return engine

# %%
# Iniciar a engine do SQLAlchemy
engine = create_sqlalchemy_engine_from_txt()
engine

# %% [markdown]
# ### Import the regular grid cells with the aggregated data extracted from OSM edit history

# %%
# Lista de tabelas disponíveis (pode ser fixa ou dinâmica)
camadas_postgis = [
    "steps_merged_1to6_reclass",
]

# Dropdown para seleção da tabela
dropdown_pg = widgets.Dropdown(
    options=["Selecione a camada do banco PostGIS:"] + camadas_postgis,
    description='Camada:',
    style={'description_width': 'initial'}
)

# Saída de mensagens
output_pg = widgets.Output()

# Variável global para armazenar o GeoDataFrame
gdf_osm = None

# Função ao selecionar uma camada
def select_pg_layer(change):
    global gdf_osm
    selected_layer = change['new']

    if selected_layer != "Selecione a camada do banco PostGIS:":
        query = f'SELECT * FROM public."{selected_layer}"'
        try:
            gdf_osm = gpd.read_postgis(query, con=engine, geom_col="geom")
            with output_pg:
                clear_output(wait=True)
                print(f"Camada '{selected_layer}' carregada com sucesso.")
                print(f"{len(gdf_osm)} registros | {len(gdf_osm.columns)} colunas.")
        except Exception as e:
            with output_pg:
                clear_output(wait=True)
                print(f"Erro ao carregar camada '{selected_layer}': {e}")

# Conectar dropdown ao evento
dropdown_pg.observe(select_pg_layer, names='value')

# Exibir UI
display(dropdown_pg, output_pg)


# %%
# Check the five first and last records of GeoDataFrame
display(gdf_osm)

# %%
# === Definição de classes e nomes de colunas ===

feature_classes = {
    "cbge_area_verde":       "cbge_area_verde",
    "cbge_praca":            "cbge_praca",
    "laz_campo_quadra":      "laz_campo_quadra",
    "edif_ensino":           "edif_ensino",
    "edif_saude":            "edif_saude",          # <- corrigido (antes estava "edfi_saude")
    "edif_comerc_serv":      "edif_comerc_serv",
    "edif_desenv_social":    "edif_desenv_social",
    "edif_constr_lazer":     "edif_constr_lazer",
    "edif_turistica":        "edif_turistica",
    "edif_pub_civil":        "edif_pub_civil",
    "edif_metro_ferroviaria":"edif_metro_ferroviaria"
}

# Colunas reais (step1 - name_ratio) na ordem desejada
feature_order = [f"step1_consolidado_{cls}_name_ratio" for cls in feature_classes.keys()]

# Rótulos legíveis para legenda
feature_labels = {
    f"step1_consolidado_{cls}_name_ratio": label
    for cls, label in feature_classes.items()
}
print(f"[OK] feature_classes carregadas ({len(feature_classes)} classes).")

# %% [markdown]
# ---
# ### Quantitative Analysis

# %% [markdown]
# #### 1. Preliminary Exploratory analysis

# %% [markdown]
# ##### 1.1 Histogram Analysis

# %%
# === Histograma (interativo) ===
# Preliminary analysis with histograms - one plot version
import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import ticker as mticker
from matplotlib.font_manager import FontProperties
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Diretório de saída ---
histogram_output_dir = 'results/9_instrisic_analysis/1_descriptive_statistics/1_histograms/'
os.makedirs(histogram_output_dir, exist_ok=True)

# --- Área de estudo  ---
study_area = "Belo Horizonte"
study_area_name_only = study_area

# === Widgets ===
palette_mode = widgets.ToggleButtons(
    options=[('Color (Colorblind-safe)', 'color'), ('Grayscale (B&W)', 'gray')],
    value='color',
    description='Palette:',
    button_style=''
)

show_kde = widgets.Checkbox(
    value=True, description='Show KDE curves', indent=False
)

# NOVO: normalizar por classe (percentual) vs contagem (log)
normalize_toggle = widgets.ToggleButtons(
    options=[('Counts (log scale)', 'count'), ('Normalize per class (%)', 'pct')],
    value='count',
    description='Y-axis:',
    button_style=''
)

plot_button  = widgets.Button(description='Generate Histogram', button_style='success', icon='bar-chart')
save_button  = widgets.Button(description='Save PNG', button_style='info', icon='save', disabled=True)
clear_button = widgets.Button(description='Clear', button_style='warning', icon='trash')

output = widgets.Output()

# --- Estado global do último plot, para salvar sem redesenhar ---
_last_fig = None
_last_ax  = None
_last_save_path = None

# === Paletas ===
def get_draw_style(mode: str, n: int):
    """
    Retorna (colors, hatches) para n séries.
    - 'color': começa com 2 verdes (colorblind-safe) p/ as 2 primeiras classes, depois expande.
    - 'gray' : tons de cinza + hachuras (P&B).
    """
    if mode == 'color':
        # dois verdes (Okabe-Ito / Set2-like), depois continua com tab20
        start = ['#009E73', '#66A61E']   # verdes (área verde, praça)
        rest  = list(plt.get_cmap('tab10').colors)
        base  = start + rest
        while len(base) < n:
            base = base + rest
        colors = base[:n]
        hatches = [None] * n
        return colors, hatches
    else:
        # tons de cinza + hachuras
        grays = np.linspace(0.15, 0.85, n)
        colors = [(g, g, g) for g in grays]
        hatch_cycle = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
        hatches = [hatch_cycle[i % len(hatch_cycle)] for i in range(n)]
        return colors, hatches

# === util: ticks "bonitos" para symlog e formatação em inteiros ===
def make_symlog_ticks(ymax):
    if ymax <= 1:
        return [0, 1]
    ticks = [0, 1]
    v = 1
    while v < ymax:
        for m in (1, 2, 5):
            t = m * v
            if t <= ymax:
                ticks.append(t)
        v *= 10
    return sorted(set(int(t) for t in ticks))

int_fmt = mticker.FuncFormatter(lambda x, pos: f"{int(x):d}")

# === Núcleo de desenho  ===
def draw_histogram(ax, gdf, cols, labels_map, mode='color', show_kde_lines=True, normalize=False):
    """
    Desenha o histograma agrupado (side-by-side por classe para cada bin).
    - Se normalize=True, y mostra % de células por bin (0–100), mesma escala para todas as classes.
    - Se normalize=False, y mostra contagens e usa escala log (symlog) automática.
    Retorna: (handles, y_lim, ns) onde ns é a lista de n por classe (células >0).
    """
    # Bins de 0 a 100, largura 5
    bin_edges   = np.arange(0, 101, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_groups = len(cols)
    if total_groups == 0:
        ax.text(0.5, 0.5, "No available columns to plot.", ha='center', va='center', transform=ax.transAxes)
        return [], 0, []

    # largura de barra calculada para caber no bin
    bin_width = np.diff(bin_edges)[0]
    bar_width = bin_width / (total_groups + 1)

    colors, hatches = get_draw_style(mode, total_groups)
    handles = []
    max_count = 0
    ns = []  # n por classe (células com name_ratio > 0)

    for i, col in enumerate(cols):
        values = pd.to_numeric(gdf[col], errors="coerce").dropna()
        values = values[values > 0]           # só células com name_ratio > 0
        n = len(values)
        ns.append(n)

        counts, _ = np.histogram(values, bins=bin_edges)
        if normalize:
            counts = (counts / n * 100.0) if n > 0 else np.zeros_like(counts, dtype=float)

        max_count = max(max_count, counts.max() if len(counts) else 0)

        offset = (i - (total_groups - 1) / 2) * bar_width
        bars = ax.bar(
            bin_centers + offset,
            counts,
            width=bar_width * 0.95,
            color=colors[i],
            alpha=0.85 if mode == 'color' else 1.0,
            label=labels_map.get(col, col),
            edgecolor='black' if mode == 'gray' else 'none',
            linewidth=0.3 if mode == 'gray' else 0.0,
            hatch=hatches[i]
        )
        handles.append(bars[0])

        # KDE (apenas se houver variância)
        if show_kde_lines:
            if len(values) >= 5 and np.nanstd(values) > 0:
                sns.kdeplot(
                    values, ax=ax, 
                    linewidth=1.5, 
                    bw_adjust=0.8,
                    clip=(0, 100),
                    cut=0, 
                    warn_singular=False,
                    zorder=10
                    )
                # pega a última linha desenhada e aplica um "halo" para destacar
                ln = ax.lines[-1]
                ln.set_path_effects([pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal()])
                # sombreia sob a KDE
                xs, ys = ln.get_xdata(), ln.get_ydata()
                ax.fill_between(xs, 0, ys, color=colors[i], alpha=0.15, zorder=9)

            elif len(values) > 0:
                ax.plot(values, np.full_like(values, -0.5), '|',
                        alpha=0.7, markersize=8, clip_on=False, zorder=8
                        )
                
    # Eixos e grade
    ax.set_title(f"Histogram of name ratio (%) for each classes in {study_area_name_only}", pad=14, fontsize=13)
    ax.set_xlabel("Grid cell values of name_ratio (%) > 0", labelpad=8, fontsize=12)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 101, 20)))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(False, axis='x')
    ax.grid(False, axis='y')

    if normalize:
        y_lim = 100
        ax.set_ylim(0, y_lim)
        ax.set_ylabel("Share of grid cells (%)", labelpad=8, fontsize=12)
        ax.set_yscale('linear')
        ax.yaxis.set_major_locator(mticker.FixedLocator([0,20,40,60,80,100]))
    else:
        # Y dinâmico + escala symlog com *ticks* inteiros e explicativos
        y_max = max_count if max_count > 0 else 1
        y_lim = int(math.ceil(y_max * 1.10))
        ax.set_ylim(0, y_lim)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylabel("Frequência de células da grade\n(log10; rótulos são contagens: 1, 2, 5, 10 …)", labelpad=10, fontsize=12)
        ax.set_yticks(make_symlog_ticks(y_lim))
        ax.yaxis.set_major_formatter(int_fmt)

    return handles, y_lim, ns

# === Handlers dos botões ===
def plot_handler(_):
    global _last_fig, _last_ax, _last_save_path
    _last_fig = None
    _last_ax  = None
    _last_save_path = None
    save_button.disabled = True

    # Reexibir a UI
    clear_output(wait=True)
    display(widgets.VBox([
        widgets.HBox([palette_mode, show_kde, normalize_toggle]),
        widgets.HBox([plot_button, save_button, clear_button]),
        output
    ]))

    with output:
        output.clear_output(wait=True)

        # Filtra colunas disponíveis (evita KeyError)
        available_cols = [c for c in feature_order if c in gdf_osm.columns]
        missing = [c for c in feature_order if c not in gdf_osm.columns]
        if missing:
            print(f"[WARN] {len(missing)} colunas ausentes no gdf_osm (serão ignoradas). Ex.: {missing[:4]}")

        fig, ax = plt.subplots(figsize=(14, 7))

        normalize = (normalize_toggle.value == 'pct')
        handles, y_lim, ns = draw_histogram(
            ax=ax,
            gdf=gdf_osm,
            cols=available_cols,
            labels_map=feature_labels,
            mode=palette_mode.value,
            show_kde_lines=show_kde.value,
            normalize=normalize
        )

        # Legenda à direita (fora da área do plot) com (n=…)
        if handles:
            legend_labels = []
            for col, n in zip(available_cols, ns):
                base = feature_labels.get(col, col)
                legend_labels.append(f"{base} (n={n})")
            title_fp = FontProperties(weight='bold', size=11)
            ax.legend(
                handles=handles,
                labels=legend_labels,
                title="Classes ET-EDGV/OSM tag (nº)\n",
                title_fontproperties=title_fp,
                fontsize=10,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                labelspacing=0.8,      # espaço entre os itens
                handletextpad=0.5,     # espaço entre símbolo e texto
                borderpad=0.8          # acolchoamento interno geral

            )
        fig.tight_layout(rect=[0, 0, 0.83, 1])  # reserva espaço para a legenda à direita

        plt.show()

        # guarda estado para salvar
        _last_fig = fig
        _last_ax  = ax
        save_button.disabled = False
        if normalize:
            print("[OK] Histograma gerado (normalizado por classe, Y = 0–100%).")
        else:
            print(f"[OK] Histograma gerado (contagens em escala log; Y até ≈ {y_lim}).")

def save_handler(_):
    global _last_fig, _last_ax, _last_save_path
    if _last_fig is None:
        with output:
            print("[INFO] Nada para salvar. Gere o histograma primeiro.")
        return

    # Caminho de saída com sufijos do modo
    fname = f'grouped_histogram_kde_classes_BH_{study_area_name_only.replace(" ", "_").lower()}_{palette_mode.value}_{normalize_toggle.value}.png'
    save_path = os.path.join(histogram_output_dir, fname)

    _last_fig.tight_layout(rect=[0, 0, 0.83, 1])
    _last_fig.savefig(save_path, dpi=300)
    _last_save_path = save_path

    with output:
        print(f"[OK] Plot salvo em: {save_path}")

def clear_handler(_):
    global _last_fig, _last_ax, _last_save_path
    plt.close('all')
    _last_fig = None
    _last_ax  = None
    _last_save_path = None
    save_button.disabled = True

    with output:
        output.clear_output()
        print("[OK] Limpo. Gere um novo histograma quando quiser.")

# === Conecta botões ===
plot_button.on_click(plot_handler)
save_button.on_click(save_handler)
clear_button.on_click(clear_handler)

# === Mostra UI ===
display(widgets.VBox([
    widgets.HBox([palette_mode, show_kde, normalize_toggle]),
    widgets.HBox([plot_button, save_button, clear_button]),
    output
]))


# %% [markdown]
# ##### 1.2 BoxPlot Analysis

# %%
# Preliminary analysis with boxplots

# Setup directory for saving boxplots
boxplot_output_dir = 'results/1_descriptive_statistics/2_boxplots/'
os.makedirs(boxplot_output_dir, exist_ok=True)

# Extract study area name function (based on selected_file)
def extract_study_area_name(selected_file):
    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]
    return study_area_token.replace("-", " ").replace("_", " ").title()

# Ensure that selected_file is already defined elsewhere
study_area_name_only = extract_study_area_name(selected_file)

# --- Widgets ---
dropdown_boxplot = widgets.Dropdown(
    options=[col for col in gdf_osm.columns if col != 'geometry'],
    value='leisure_name_ratio',
    description='Column:',
    disabled=False
)

plot_boxplot_button = widgets.Button(
    description='Generate Boxplot',
    button_style='success'
)

save_boxplot_button = widgets.Button(
    description='Save Boxplot as PNG',
    button_style='info',
    disabled=True
)

output_boxplot = widgets.Output()

# --- Functions ---
def plot_boxplot(b=None):
    clear_output(wait=True)
    display(widgets.VBox([dropdown_boxplot, plot_boxplot_button, save_boxplot_button, output_boxplot]))

    column = dropdown_boxplot.value
    with output_boxplot:
        output_boxplot.clear_output()
        plt.figure(figsize=(6, 6))
        sns.boxplot(y=gdf_osm[column].dropna())
        plt.title(f'Boxplot of {column} of {study_area_name_only}', fontsize=14)
        plt.ylabel(f'{column} values')
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        save_boxplot_button.disabled = False

def save_boxplot(b):
    column = dropdown_boxplot.value
    save_path = os.path.join(
        boxplot_output_dir,
        f'boxplot_{column.lower()}_{study_area_name_only.replace(" ", "_").lower()}.png'
    )
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=gdf_osm[column].dropna())
    plt.title(f'Boxplot of {column} of {study_area_name_only}', fontsize=14)
    plt.ylabel(f'{column} values')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    with output_boxplot:
        print(f"Boxplot saved at: {save_path}")

# --- Bind Buttons ---
plot_boxplot_button.on_click(plot_boxplot)
save_boxplot_button.on_click(save_boxplot)

# --- Display ---
display(widgets.VBox([dropdown_boxplot, plot_boxplot_button, save_boxplot_button, output_boxplot]))

# Generate the initial plot
plot_boxplot()

# %% [markdown]
# #### 2. Ordinary Least Squares (OLS) regression

# %%
# Convert GeoDataFrame to DataFrame by removing the geometry column
gdf_osm_copy = gdf_osm.copy()
df_osm = gdf_osm_copy.drop(columns=['geometry'])

# %%
# Check the five first and last records of DataFrame
display(df_osm)

# %%
# Check the data types
df_osm.dtypes

# %%
# Perform Multiple Linear Regression (OLS)
# Dynamic selection of the dependent variable and the independent variables
# Dependent variables
dependent_var_widget = Dropdown(options=[col for col in df_osm.columns if col != 'id'],
                                description='Dependent Var:',value = 'leisure_name_ratio')

# Independent variables
independent_vars_widget = SelectMultiple(
    options=[col for col in df_osm.columns if col != 'id'],
    description='Independent Vars (use ctrl to select multiple variables):',
    layout={'width': '90%', 'height': '200px'})

# Global Dictionary to store all regression summaries
regression_summaries = {}

# Global dictionary for storing R² values
r2_values = {}

# --- Utility Function ---
def extract_study_area_name(selected_file):
    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]
    return study_area_token.replace("-", " ").replace("_", " ").title()

# Function to run the regression
def run_regression(button):
    global regression_summaries, r2_values
    dependent_var = dependent_var_widget.value
    independent_vars = list(independent_vars_widget.value)
    X = df_osm[independent_vars]
    y = df_osm[dependent_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Print the summary for the current run
    display(model.summary())

    # To save the summary as text file
    regression_summary = model.summary().as_text()

    # Store the summary using the model name as the key
    model_name = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    regression_summaries[model_name] = regression_summary

    # Store R² Adjusted values
    r2_values[model_name] = model.rsquared_adj

# Function to clear the outputs
def clear_outputs(button):
    clear_output()
    display_widgets()

# Function to save all regression results
def save_regression_results(button):
    study_area_name_only = extract_study_area_name(selected_file)
    detected_tag = dependent_var_widget.value.split('_')[0] if '_' in dependent_var_widget.value else 'unknown'
    save_filename = f'regression_summaries_{detected_tag}_{study_area_name_only.replace(" ", "_")}.txt'
    save_path = os.path.join('results/2_OLS_regression/', save_filename)

    if regression_summaries:
        with open(save_path, "w") as file:
            for model, summary in regression_summaries.items():
                file.write(f"Model: {model}\n")
                file.write(f"{summary}\n\n")
        print(f"All results saved successfully at: {save_path}")
    else:
        print("No results to save.")

# Function to Generate the Bar Chart for Adjusted R² Values
def plot_r2_bar_chart():
    # Convert the dictionary of R² values into a DataFrame for visualization
    df_r2 = pd.DataFrame(list(r2_values.items()), columns=['Model', 'Adjusted R2'])

    # Sort the models by 'Adjusted R2' values for better visualization
    df_r2_sorted = df_r2.sort_values(by='Adjusted R2', ascending=True)

    # Create the bar chart with bars closer together
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = sns.barplot(x='Adjusted R2', y='Model', data=df_r2_sorted, palette='viridis', dodge=False)

    # Add labels and title for the chart
    ax.set_xlabel('Adjusted R²')
    ax.set_ylabel('Model')
    ax.set_title('Adjusted R² Values for Different Regression Models')

    # Add the values on the bars
    for bar in bars.patches:
        ax.text(
            bar.get_width(),  # get the horizontal end point of the bar
            bar.get_y() + bar.get_height() / 2,  # get the y position of the bar
            f'{bar.get_width():.2f}',  # the value we want to add as text
            va='center',  # vertical alignment
            ha='left',  # horizontal alignment
            color='black',  # text color
            fontweight='light'  # font weight
        )

    # Display the chart
    plt.tight_layout()
    plt.show()

# Function to Buttons actions
def button_actions(b):
    if b.description == 'Run Regression':
        run_regression(b)
    elif b.description == 'Clear Outputs':
        clear_outputs(b)
    elif b.description == 'Save Results':
        save_regression_results(b)
    elif b.description == 'Generate R² bar chart':
        plot_r2_bar_chart()

# Create Buttons to run the analysis
run_button = Button(description='Run Regression',
                    button_style='success')
heatmap_button = Button(description='Generate R² bar chart',
                        button_style='success')
save_button = Button(description="Save Results",
                     button_style='info')
clear_button = Button(description="Clear Outputs",
                      button_style='danger')

# Assign the button_actions function as the on_click event handler for all buttons
for button in [run_button, heatmap_button, save_button, clear_button]:
    button.on_click(button_actions)

# Organize the widgets
variable_selection_box = VBox([dependent_var_widget, independent_vars_widget])
buttons_box = VBox([run_button, heatmap_button, save_button, clear_button])

# Function to display the widgets
def display_widgets():
    hbox = HBox([variable_selection_box, buttons_box])
    display(hbox)

# Display the Widgets and the Buttons
display_widgets()

# %% [markdown]
# ---
# ### Spatial Patterns Analysis

# %% [markdown]
# #### 3. Choropleth maps

# %%
# ===========================
# Choropleth maps (generate ≠ save) + GRID (11 classes)
# ===========================
import os, warnings, webbrowser, re
import numpy as np
import pandas as pd
import folium, mapclassify, matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import MultiPolygon
from ipywidgets import VBox, HBox, Dropdown, Checkbox, Button, Layout, Output
from IPython.display import display, clear_output, HTML
from folium import Element
from branca.element import Figure


# -------------------------
# 1) Dados/CRS/IDs
# -------------------------
if gdf_osm.crs is None:
    gdf_osm = gdf_osm.set_crs("EPSG:4674")
if 'id' not in gdf_osm.columns:
    gdf_osm['id'] = np.arange(len(gdf_osm))

# -------------------------
# 2) Saída e warnings
# -------------------------
output_path = 'results/9_Instrisic_analysis/3_choropleth_maps/'
os.makedirs(output_path, exist_ok=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------------
# 3) Utilidades
# -------------------------
study_area_name_only = "Belo Horizonte"

def calculate_centroid_of_union(gdf):
    union_poly = gdf.geometry.unary_union
    if isinstance(union_poly, MultiPolygon):
        union_poly = MultiPolygon(union_poly).convex_hull
    return [union_poly.centroid.y, union_poly.centroid.x]

def add_choropleth(map_obj, column, method, palette, manual_style=False,
                   bins_override=None, labels_override=None, colors_override=None):
    series = pd.to_numeric(gdf_osm[column], errors='coerce').fillna(0)

    # bins/labels
    if bins_override is not None:
        bins = bins_override
        class_labels = pd.cut(series, bins=bins, labels=False, include_lowest=True, right=True)
        labels = labels_override if labels_override is not None else ['' for _ in range(len(bins)-1)]
    else:
        unique_values = len(series.unique())
        k = min(4, unique_values) if unique_values > 0 else 4

        if method == 'Quantiles':
            clf = mapclassify.Quantiles(series, k=k)
            class_labels = clf.yb
            bins = [float(series.min())] + list(map(float, clf.bins))
            labels = [f"Q{i+1}" for i in range(len(bins)-1)]
        elif method == 'EqualInterval':
            clf = mapclassify.EqualInterval(series, k=k)
            class_labels = clf.yb
            bins = [float(series.min())] + list(map(float, clf.bins))
            labels = ['' for _ in range(len(bins)-1)]
        elif method == 'NaturalBreaks':
            clf = mapclassify.NaturalBreaks(series, k=k)
            class_labels = clf.yb
            bins = [float(series.min())] + list(map(float, clf.bins))
            labels = ['' for _ in range(len(bins)-1)]
        elif method == 'EqualIntervalCustom':
            bins = [0.0, 25.0, 50.0, 75.0, 100.0]
            class_labels = pd.cut(series, bins=bins, labels=False, include_lowest=True, right=True)
            labels = ['' for _ in range(len(bins)-1)]
        else:
            raise ValueError("Classification method not supported")

    gdf_osm['__class_bin__'] = class_labels

    # cores
    if colors_override is not None:
        colors = colors_override
    else:
        cmap = matplotlib.colormaps.get_cmap(palette)
        colors = [matplotlib.colors.to_hex(cmap(i / max(1, len(bins)-2))) for i in range(len(bins)-1)]

    if manual_style:
        def style_function(feature):
            class_value = feature['properties'].get('__class_bin__', None)
            if class_value is None or (isinstance(class_value, float) and np.isnan(class_value)):
                return {'fillOpacity': 0, 'weight': 0}
            else:
                return {
                    'fillColor': colors[int(class_value)],
                    'color': 'black',
                    'weight': 0.4,
                    'fillOpacity': 0.6
                }
        folium.GeoJson(
            data=gdf_osm.to_json(),
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['id', column])
        ).add_to(map_obj)
    else:
        choropleth = folium.Choropleth(
            geo_data=gdf_osm.to_json(),
            data=gdf_osm,
            columns=['id', '__class_bin__'],
            key_on='feature.properties.id',
            fill_color=palette,
            fill_opacity=0.5,
            line_opacity=0.4,
            legend_name=''
        ).add_to(map_obj)
        folium.GeoJsonTooltip(fields=['id', column]).add_to(choropleth.geojson)

    return bins, colors, labels

def create_legend_html(bins, colors, labels, method, column):
    # título em 2 linhas, negrito; legenda à ESQUERDA
    legend_title_html = f"""
    <div style="font-weight:700; line-height:1.15; margin-bottom:6px;">
        {method} classification<br/>{column}
    </div>
    """
    legend_html = f'''
    <div style="
        position: fixed; top: 80px; left: 20px;
        width: auto; max-width: 260px; height: auto;
        border: 1px solid #888; z-index: 9999;
        font-size: 13px; padding: 8px 10px;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    ">
        {legend_title_html}
    '''
    for i in range(len(colors)):
        color = colors[i]
        if method == 'Quantiles' and i < len(labels):
            interval_text = f"{labels[i]} ({bins[i]:.0f}-{bins[i+1]:.0f})"
        else:
            interval_text = f"{bins[i]:.0f} – {bins[i+1]:.0f}"
        legend_html += f'''
        <div style="display:flex; align-items:center; margin:2px 0;">
            <i style="background:{color}; width:14px; height:14px; border:1px solid #555; margin-right:6px;"></i>
            <span>{interval_text}</span>
        </div>
        '''
    legend_html += '</div>'
    return legend_html

# -------------------------
# 4) SINGLE MAP (gera, não salva) + fit_bounds
# -------------------------
def update_map(column, method, palette, manual_style=False):
    fig = Figure(width=1080, height=650)
    # cria com centro, depois ajusta a EXTENSÃO TOTAL (fit_bounds)
    center = calculate_centroid_of_union(gdf_osm)
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    m.add_to(fig)

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        attr='CartoDB', name='CartoDB light'
    ).add_to(m)

    bins, colors, labels = add_choropleth(m, column, method, palette, manual_style)
    legend_html = create_legend_html(bins, colors, labels, method, column)
    m.get_root().html.add_child(Element(legend_html))

    title_html = f'''
    <div style="position: fixed;
        top: 10px; left: 50%; transform: translate(-50%, 0);
        width: auto; max-width: 70%;
        border: 1px solid #888; z-index:9999;
        font-size: 16px; font-weight: 700;
        background: rgba(255,255,255,0.85); text-align: center; padding: 6px 10px;">
        Choropleth map of "{column}" in {study_area_name_only}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # >>> centraliza pela EXTENSÃO inteira
    xmin, ymin, xmax, ymax = gdf_osm.total_bounds
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    display(m)

    filename = f"{column}-{study_area_name_only}-choropleth-{method}.html".replace(" ", "_")
    save_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return fig, m, save_path

# -------------------------
# 5) GRID ESTÁTICA (11 mapas) com fundo transparente (PNG)
# -------------------------
def compute_global_bins(cols, method):
    if method == 'EqualIntervalCustom':
        return [0, 25, 50, 75, 100]
    vals = []
    for c in cols:
        x = pd.to_numeric(gdf_osm[c], errors='coerce').dropna()
        if len(x): vals.append(x)
    if not vals:
        return [0, 25, 50, 75, 100]
    all_values = pd.concat(vals)
    k = 4
    if method == 'Quantiles':
        clf = mapclassify.Quantiles(all_values, k=k); return [float(all_values.min())] + list(map(float, clf.bins))
    if method == 'EqualInterval':
        clf = mapclassify.EqualInterval(all_values, k=k); return [float(all_values.min())] + list(map(float, clf.bins))
    if method == 'NaturalBreaks':
        clf = mapclassify.NaturalBreaks(all_values, k=k); return [float(all_values.min())] + list(map(float, clf.bins))
    return [0, 25, 50, 75, 100]

def build_static_grid_11(method:str, palette:str):
    """Retorna (fig, suggested_filename) com uma grade 4x3 (11 mapas) e legenda comum — fundo transparente."""
    cols = [c for c in feature_order if c in gdf_osm.columns][:11]
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_alpha(0); ax.set_facecolor('none')
        ax.axis('off'); ax.text(0.5, 0.5, "Nenhuma coluna disponível.", ha='center', va='center')
        return fig, "grid11_empty.png"

    bins = compute_global_bins(cols, method)
    N = len(bins) - 1
    cmap_cont = matplotlib.colormaps.get_cmap(palette)
    colors = [cmap_cont(i/(N-1) if N > 1 else 0.5) for i in range(N)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, N)

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 18), constrained_layout=True)
    fig.patch.set_alpha(0)  # fundo transparente da figura

    axes = np.atleast_2d(axes)
    xmin, ymin, xmax, ymax = gdf_osm.total_bounds

    for i in range(nrows*ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.set_facecolor('none')  # fundo transparente do eixo
        if i >= len(cols):
            ax.set_axis_off()
            continue

        col = cols[i]
        gdf_osm.plot(
            column=col,
            ax=ax,
            cmap=cmap, norm=norm,
            linewidth=0.2, edgecolor='black',
            missing_kwds={'color': (0,0,0,0), 'edgecolor':'none'}
        )
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_title(feature_labels.get(col, col), fontsize=10, pad=4)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.035, pad=0.04)
    cbar.set_label("Name ratio (%)", fontsize=11)
    cbar.set_ticks(bins); cbar.set_ticklabels([f"{b:.0f}" for b in bins])

    fname = f"grid11-static-{study_area_name_only}-{method}-{palette}.png".replace(" ", "_")
    return fig, fname

# -------------------------
# 6) Widgets & Estado (somente colunas das CLASSES)
# -------------------------
feature_columns = [c for c in feature_order if c in gdf_osm.columns]
column_options  = [(feature_labels.get(c, c), c) for c in feature_columns]
default_column  = column_options[0][1] if column_options else (gdf_osm.columns[0])

column_selector = Dropdown(
    options=column_options,
    value=default_column,
    description='Column:',
    layout=Layout(width='40%')
)
method_selector = Dropdown(
    options=['Quantiles', 'EqualInterval', 'EqualIntervalCustom', 'NaturalBreaks'],
    value='EqualIntervalCustom',
    description='Method:',
    layout=Layout(width='28%')
)
palette_selector = Dropdown(
    options=plt.colormaps(),
    value='YlGn',
    description='Palette:',
    layout=Layout(width='28%')
)
manual_style_selector = Checkbox(value=True, description='Use Manual Style?')

# botões
generate_button       = Button(description="Generate Single Map", button_style='success')
save_single_button    = Button(description="Save Single Map (HTML)", button_style='info', disabled=True)
render_inline_button  = Button(description="Render Inline (HTML)", button_style='primary', disabled=True)
generate_grid_button  = Button(description="Generate 11-Class Grid (PNG)", button_style='primary')
save_grid_button      = Button(description="Save 11-Class Grid", button_style='', disabled=True)
clear_button          = Button(description="Clear Output", button_style='warning')

single_output = Output()
grid_output   = Output()

# estado
_last_map   = {"fig": None, "m": None, "save_path": None}
_last_grid  = {"fig": None, "fname": None}

# -------------------------
# 7) Handlers
# -------------------------
def on_generate_button_clicked(b):
    with single_output:
        clear_output(wait=True)
        fig, m, save_path = update_map(
            column_selector.value, method_selector.value, palette_selector.value, manual_style_selector.value
        )
        _last_map.update({"fig": fig, "m": m, "save_path": save_path})
        save_single_button.disabled   = False
        render_inline_button.disabled = False
        print("[OK] Mapa gerado. Use 'Save' para salvar HTML, ou 'Render Inline' para forçar render no output.")

def on_save_single_clicked(b):
    with single_output:
        if not _last_map["m"]:
            print("[INFO] Gere o mapa antes de salvar.")
            return
        _last_map["m"].save(_last_map["save_path"])
        print(f"[OK] Choropleth map saved in: {_last_map['save_path']}")
        try:
            webbrowser.open(f"file://{os.path.abspath(_last_map['save_path'])}")
        except Exception:
            pass

def on_render_inline_clicked(b):
    with single_output:
        if not _last_map["m"]:
            print("[INFO] Gere o mapa primeiro.")
            return
        # Renderiza o HTML do folium inline (útil no VSCode/mac)
        display(HTML(_last_map["m"].get_root().render()))

def on_generate_grid_button_clicked(b):
    with grid_output:
        clear_output(wait=True)
        fig, fname = build_static_grid_11(method_selector.value, palette_selector.value)
        display(fig)
        _last_grid["fig"] = fig
        _last_grid["fname"] = fname
        save_grid_button.disabled = False
        print("[OK] Grade 11 classes gerada. Use 'Save' para exportar PNG.")

def on_save_grid_clicked(b):
    with grid_output:
        if _last_grid["fig"] is None:
            print("[INFO] Gere a grade primeiro.")
            return
        save_path = os.path.join(output_path, _last_grid["fname"])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _last_grid["fig"].savefig(save_path, dpi=300, transparent=True)  # <<< fundo transparente
        print(f"[OK] 11-Class grid saved in: {save_path}")

def on_clear_button_clicked(b):
    with single_output:
        clear_output(wait=True)
    with grid_output:
        clear_output(wait=True)
    _last_map.update({"fig": None, "m": None, "save_path": None})
    _last_grid.update({"fig": None, "fname": None})
    save_single_button.disabled   = True
    render_inline_button.disabled = True
    save_grid_button.disabled     = True
    print("[OK] Limpo.")

def display_widgets():
    display(VBox([
        HBox([column_selector, method_selector, palette_selector]),
        manual_style_selector,
        HBox([generate_button, save_single_button, render_inline_button, clear_button]),
        single_output,
        HBox([generate_grid_button, save_grid_button]),
        grid_output
    ]))

# conecta
generate_button.on_click(on_generate_button_clicked)
save_single_button.on_click(on_save_single_clicked)
render_inline_button.on_click(on_render_inline_clicked)
generate_grid_button.on_click(on_generate_grid_button_clicked)
save_grid_button.on_click(on_save_grid_clicked)
clear_button.on_click(on_clear_button_clicked)

# UI
display_widgets()


# %% [markdown]
# #### 4. Global Moran's Index

# %% [markdown]
# - The bellow ceel code allows the user to select multiple variables from a GeoDataFrame and then analyze each one for spatial autocorrelation using Moran's I. The results, including the value of Moran's I and its p-value, are printed and visualized in a scatterplot for each selected variable. The selection widget's width is adjusted for better display.
# 
# 
# - Statistical test
# 
#   - Null hypothesis represents the spatial independence of the data (Spatial distribution of the variable in question is random )
#   - Confidence margin of 95% (p value < .05)

# %%
# FutureWarning ignore
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to perform Moran's I analysis
def analyze_morans_i(selected_vars):
    w = weights.Queen.from_dataframe(gdf_osm, use_index=True)
    w.transform = 'r'

    for var in selected_vars:
        moran = Moran(gdf_osm[var], w)
        print(f"Moran's Index for {var}: {moran.I:.4f}, p-value: {moran.p_sim}")

        # Plot the Moran scatterplot
        plot_moran(moran, zstandard=True, figsize=(10, 4))
        plt.show()

# Button to perform the analysis and clear output
analyze_button = widgets.Button(description="Analyze Moran's I")
clear_output_button = widgets.Button(description="Clear Output")

# Create the selection widget for selecting multiple variables
select_variables = widgets.SelectMultiple(
    options=gdf_osm.columns,
    value=['leisure_name_ratio'],  # default value
    description='Variables:',
    disabled=False,
    layout={'width': '50%', 'height': '200px'}
)

# Function to handle button click event for analyzing Moran's I
def on_analyze_button_clicked(b):
    clear_output(wait=True)
    display_widgets()  # Redisplay widgets to maintain UI state
    analyze_morans_i(select_variables.value)

# Function to handle button click event for clearing the output
def on_clear_output_button_clicked(b):
    clear_output(wait=True)
    display_widgets()

# Function to display widgets with the appropriate layout
def display_widgets():
    display(VBox([
        select_variables,
        HBox([analyze_button, clear_output_button])
    ]))

# Connect the buttons to their respective event handlers
analyze_button.on_click(on_analyze_button_clicked)
clear_output_button.on_click(on_clear_output_button_clicked)

# Display the widgets initially
display_widgets()

# %% [markdown]
# #### 5. Hot Spot analysis using the Getis-Ord Gi statistic

# %%
# Hot Spot Analysis using Getis-Ord Gi* Statistic

# Ensure that gdf_osm and selected_file are loaded

# Set CRS if missing
if gdf_osm.crs is None:
    gdf_osm = gdf_osm.set_crs("EPSG:4674")

# Output directory
output_path = 'results/5_getis_ord_maps/'

warnings.simplefilter(action='ignore', category=FutureWarning)

# Calculate centroid of union of geometries
def calculate_centroid_of_union(gdf):
    union_poly = gdf.geometry.unary_union
    if isinstance(union_poly, MultiPolygon):
        union_poly = MultiPolygon(union_poly).convex_hull
    return [union_poly.centroid.y, union_poly.centroid.x]

# Color assignment based on Z-score
def color_producer(val):
    if val > 2.0:
        return 'red' # Vermelho
    elif 1.0 < val <= 2.0:
        return 'orange' # laranja
    elif -1.0 <= val <= 1.0:
        return 'darkgray' # cinza escuro
    elif -2.0 <= val < -1.0:
        return 'darkturquoise' # Ciano
    else:
        return 'blue' # Azul

# Calculate Getis-Ord Gi*
def calculate_getis_ord_gi(gdf, var_name):
    w = weights.distance.KNN.from_dataframe(gdf, k=8)
    w.transform = 'B'
    gi = G_Local(gdf[var_name], w, star=True)
    gdf[f"Gi_Z_{var_name}"] = gi.Zs
    return gdf

# Title HTML
def create_title_html(selected_var, study_area_name_only):
    title_html = f'''
    <div style="position: fixed;
    top: 10px; left: 50%; transform: translate(-50%, 0); width: auto;
    border:2px solid grey; z-index:9999; font-size:16px; font-weight: bold;
    background: rgba(255, 255, 255, 0.8); text-align: center; padding: 5px;">
    Getis-Ord Gi* Analysis for {selected_var} of {study_area_name_only}</div>
    '''
    return title_html

# Legend HTML
def create_legend_html():
    legend_html = '''
    <div style="position: fixed;
    bottom: 50px; left: 50px; width: 270px; height: auto;
    border:2px solid grey; z-index:9999; font-size:14px;
    background: rgba(255, 255, 255, 0.8);">
    &nbsp; <b>Legend</b> <br>
    &nbsp; Significant hot-spot (Z > 2.0) &nbsp; <i style="background:red;width:10px;height:10px;display:inline-block;"></i><br>
    &nbsp; Moderate hot-spot (1.0 < Z ≤ 2.0) &nbsp; <i style="background:orange;width:10px;height:10px;display:inline-block;"></i><br>
    &nbsp; Non-significant (-1.0 ≤ Z ≤ 1.0) &nbsp; <i style="background:darkgray;width:10px;height:10px;display:inline-block;"></i><br>
    &nbsp; Moderate cold-spot (-2.0 ≤ Z < -1.0) &nbsp; <i style="background:darkturquoise;width:10px;height:10px;display:inline-block;"></i><br>
    &nbsp; Significant cold-spot (Z < -2.0) &nbsp; <i style="background:blue;width:10px;height:10px;display:inline-block;"></i>
    </div>
    '''
    return legend_html

# Main function to analyze and generate map
def analyze_getis_ord_gi(selected_var):
    display("Wait: processing Getis-Ord Gi* analysis...")
    gdf_with_gi = calculate_getis_ord_gi(gdf_osm.copy(), selected_var)
    centroid_coords = calculate_centroid_of_union(gdf_with_gi)
    fig = Figure(width=1080, height=650)

    m = folium.Map(location=centroid_coords, zoom_start=14, control_scale=True)
    fig.add_child(m)

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        attr='CartoDB',
        name='CartoDB light'
    ).add_to(m)

    for _, row in gdf_with_gi.iterrows():
        centroid = row.geometry.centroid if isinstance(row.geometry, MultiPolygon) else row.geometry
        z_value = row[f"Gi_Z_{selected_var}"]
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=3,
            color=color_producer(z_value),
            fill=True,
            fill_opacity=1
        ).add_to(m)

    def style_function(feature):
        value = feature['properties'][f"Gi_Z_{selected_var}"]
        return {
            'fillColor': color_producer(value),
            'color': 'gray',
            'weight': 1,
            'fillOpacity': 0.2,
            'lineOpacity': 0.7
        }

    folium.GeoJson(
        gdf_with_gi,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[f"Gi_Z_{selected_var}"],
            aliases=["Z-score:"],
            localize=True
        )
    ).add_to(m)

    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]

    study_area_name_only = study_area_token.replace("-", " ").replace("_", " ").title()

    m.get_root().html.add_child(Element(create_legend_html()))
    m.get_root().html.add_child(Element(create_title_html(selected_var, study_area_name_only)))

    display(fig)

    output_filename = f"{selected_var}-{study_area_name_only}-getisord.html"
    save_path = os.path.join(output_path, output_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    m.save(save_path)
    display(f"Map saved at: {save_path}")

# Widgets
select_variable_widget = widgets.Select(
    options=gdf_osm.columns,
    value='leisure_name_ratio', # default var
    description='Variable:',
    layout={'width': '30%', 'height': '150px'}
)

analyze_button = widgets.Button(description="Analyze and Show Map",
                                button_style='success')
clear_output_button = widgets.Button(description="Clear Output",
                                     button_style='danger')

box_layout = VBox([select_variable_widget, HBox([analyze_button, clear_output_button])])

def on_analyze_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)
    analyze_getis_ord_gi(select_variable_widget.value)

def on_clear_output_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)

analyze_button.on_click(on_analyze_button_clicked)
clear_output_button.on_click(on_clear_output_button_clicked)

display(box_layout)

# %% [markdown]
# #### 6. Geographically Weighted Regression (GWR)

# %% [markdown]
# ##### Reproject the GeoDataFrame to apply GWR

# %%
# Function to calculate the centroid of a GeoDataFrame

def get_utm_zone_crs(gdf):
    bounds = gdf.total_bounds
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    centroid = bbox.centroid
    
    # Determine the UTM zone from the centroid's longitude
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y > 0 else 'south'

    # Create a UTM CRS based on the centroid location
    utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    
    return utm_crs

# Set the initial CRS if necessary
gdf_osm = gdf_osm.set_crs("EPSG:4674")

# Obtain the appropriate UTM CRS
utm_crs = get_utm_zone_crs(gdf_osm)

# Project the GeoDataFrame to the detected UTM CRS
gdf_osm_projected = gdf_osm.to_crs(utm_crs.to_string())

gdf_osm_projected.crs

# %% [markdown]
# ##### Applying VIF (Variance Inflation Factor) to identify multicollinearity

# %%
# Applying VIF (Variance Inflation Factor) to identify multicollinearity
# among independent variables in regression models

# --- Setup ---

# Output directories
vif_output_dir = 'results/6_gwr/vif/'
corr_output_dir = 'results/6_gwr/vif/correlation_matrix/'

# Ensure directories exist
os.makedirs(vif_output_dir, exist_ok=True)
os.makedirs(corr_output_dir, exist_ok=True)

warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to extract study area name from selected_file
def extract_study_area_name(selected_file):
    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]
    return study_area_token.replace("-", " ").replace("_", " ").title()

# --- Widgets ---
select_x_variables_widget = SelectMultiple(
    options=[col for col in gdf_osm_projected.columns if col != 'geometry'],
    description='Independent Variables:',
    disabled=False,
    layout=Layout(width='40%', height='200px')
)

vif_button = Button(description="Calculate VIF")
correlation_button = Button(description="Show Correlation Matrix")
clear_button = Button(description="Clear Results")

box_layout = VBox([
    select_x_variables_widget,
    HBox([vif_button, correlation_button, clear_button])
])

display(box_layout)

# Global variable to store detected OSM tag
detected_tag = 'unknown'

# --- Function to calculate VIF, save CSV, and show download link ---
def calculate_vif(b):
    global detected_tag # Declare usage of the detected_tag as global variable
    clear_output(wait=True)
    display(box_layout)
    selected_vars = list(select_x_variables_widget.value)

    if len(selected_vars) < 2:
        print("Please select at least two variables to calculate VIF.")
        return

    variables = gdf_osm_projected[selected_vars]

    vif_data = pd.DataFrame({
        "feature": variables.columns,
        "VIF": [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    })

    display(vif_data)

    # Detect OSM Tag dynamically
    tags_in_selection = [col.split('_')[0] for col in selected_vars if '_' in col]
    if tags_in_selection:
        detected_tag = tags_in_selection[0]
    else:
        detected_tag = 'unknown'

    # Add study area name
    study_area_name_only = extract_study_area_name(selected_file)

    # Save VIF to CSV
    save_path = os.path.join(vif_output_dir, f'vif_{detected_tag}_{study_area_name_only}.csv')
    vif_data.to_csv(save_path, index=False)
    print(f"\u2705 VIF saved at: {save_path}")

    # Provide download link
    display(FileLink(save_path, result_html_prefix="Click to download VIF file: "))

# --- Function to show correlation matrix, save PNG, and show download link ---
def show_correlation_matrix(b):
    clear_output(wait=True)
    display(box_layout)
    selected_vars = list(select_x_variables_widget.value)

    if len(selected_vars) < 2:
        print("Please select at least two variables to view the correlation matrix.")
        return

    variables = gdf_osm_projected[selected_vars]
    corr_matrix = variables.corr()

    # Plot and save
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Independet Variables')

    # Add study area name
    study_area_name_only = extract_study_area_name(selected_file)

    corr_save_path = os.path.join(corr_output_dir, f'correlation_matrix_{detected_tag}_{study_area_name_only}.png')
    plt.savefig(corr_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\u2705 Correlation matrix saved at: {corr_save_path}")
    display(FileLink(corr_save_path, result_html_prefix="Click to download correlation matrix: "))

# --- Function to clear the output ---
def on_clear_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)

# --- Event bindings ---
vif_button.on_click(calculate_vif)
correlation_button.on_click(show_correlation_matrix)
clear_button.on_click(on_clear_button_clicked)

# %% [markdown]
# ##### Check explained variance ratio

# %%
# Checking the explained variance

features = [col for col in gdf_osm_projected.columns if col != 'geometry' and col not in ['leisure_name_ratio', 'building_name_ratio', 'amenity_name_ratio']]
select_features_widget = SelectMultiple(
    options=features,
    value=[features[0]],
    description='Variables:',
    disabled=False,
    layout=Layout(width='50%', height='120px')
)

plot_button = Button(description="Plot Explained Variance")
clear_button = Button(description="Clear Results")

# Layout setup
box_layout = VBox([select_features_widget,
                   HBox([plot_button, clear_button])
                   ])
display(box_layout)

# Function to plot explained variance
def plot_variance(b):
    clear_output(wait=True)
    display(box_layout)
    selected_features = list(select_features_widget.value)
        
    # Standardizing the selected features
    x = gdf_osm_projected[selected_features]
    x_scaled = StandardScaler().fit_transform(x)
        
     # Applying PCA and capturing the explained variance
    pca = PCA()
    pca.fit(x_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
        
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Function to clear results
def clear_results(b):
    clear_output(wait=True)
    display(box_layout)

# Binding the buttons to their respective functions
plot_button.on_click(plot_variance)
clear_button.on_click(clear_results)

# %% [markdown]
# ##### Perform Geographically Weighted Regression (GWR)

# %%
# Perform Geographically Weighted Regression (GWR) with PCA

gdf_osm_projected = gdf_osm_projected.copy()

# --- Setup Directories ---

gwr_output_dir = 'results/6_gwr/gwr_pca/'

os.makedirs(gwr_output_dir, exist_ok=True)

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Utility Functions ---

def extract_study_area_name(selected_file):
    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]
    return study_area_token.replace("-", " ").replace("_", " ").title()

# --- Widgets ---
select_y_variable_widget = Dropdown(
    options=[col for col in gdf_osm_projected.columns if col != 'geometry'],
    value='leisure_name_ratio',  # Default suggested
    description='Dependent Variable:',
    disabled=False
)

select_x_variables_widget = SelectMultiple(
    options=[col for col in gdf_osm_projected.columns if col != 'geometry'],
    description='Independent Variables:',
    disabled=False,
    layout=Layout(width='50%', min_width='300px', height='200px')
)

analyze_button = Button(description="Perform GWR with PCA", layout=Layout(width='auto', min_width='200px'))
clear_button = Button(description="Clear Results", layout=Layout(width='auto', min_width='120px'))

box_layout = VBox([
    select_y_variable_widget,
    select_x_variables_widget,
    HBox([analyze_button, clear_button], layout=Layout(margin='10px 0'))
], layout=Layout(margin='0 0 10px 0'))

display(box_layout)

# --- Main Analysis Function ---
def on_analyze_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)
    display(" Processing GWR with PCA, please wait...")

    selected_y_var = select_y_variable_widget.value
    selected_x_vars = list(select_x_variables_widget.value)

    if len(selected_x_vars) < 2:
        display("Please select at least two independent variables.")
        return

    # Detect TAG from independent variables
    tags_in_selection = [col.split('_')[0] for col in selected_x_vars if '_' in col]
    detected_tag = tags_in_selection[0] if tags_in_selection else 'unknown'

    study_area_name_only = extract_study_area_name(selected_file)

    # --- Standardize Features ---
    features = gdf_osm_projected[selected_x_vars]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- Perform PCA ---
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['DependentVar'] = gdf_osm_projected[selected_y_var]

    # --- Save PCA Loadings ---
    pca_loadings = pd.DataFrame(
        pca.components_,
        columns=selected_x_vars,
        index=['PC1', 'PC2']
    )
    display(pca_loadings)

    loadings_save_path = os.path.join(
        gwr_output_dir, f'pca_loadings_{detected_tag}_{study_area_name_only}.csv'
    )
    pca_loadings.to_csv(loadings_save_path)
    print(f"PCA loadings saved at: {loadings_save_path}")

    # --- Prepare coordinates and target ---
    coords = np.array(list(zip(gdf_osm_projected.geometry.centroid.x, gdf_osm_projected.geometry.centroid.y)))
    y = principal_df['DependentVar'].values.reshape((-1, 1))
    X = principal_df[['PC1', 'PC2']].values

    # --- Perform GWR ---
    gwr_selector = Sel_BW(coords, y, X, kernel='bisquare', fixed=False)
    bw = gwr_selector.search()
    gwr_model = GWR(coords, y, X, bw).fit()

    # --- Store GWR results with TAG in column names ---
    gdf_osm_projected[f'Local_R2_{detected_tag}'] = gwr_model.localR2
    gdf_osm_projected[f'PC1_coef_{detected_tag}'] = gwr_model.params[:, 1]
    gdf_osm_projected[f'PC2_coef_{detected_tag}'] = gwr_model.params[:, 2]

    # --- Plot Results ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    scatter1 = axs[0].scatter(coords[:, 0], coords[:, 1], c=gdf_osm_projected[f'Local_R2_{detected_tag}'], cmap='viridis_r') #YlGn
    axs[0].set_title(f'Local R²')
    #axs[0].set_title(f'Local R² \n{selected_y_var} of {study_area_name_only}')
    plt.colorbar(scatter1, ax=axs[0])

    scatter2 = axs[1].scatter(coords[:, 0], coords[:, 1], c=gdf_osm_projected[f'PC1_coef_{detected_tag}'], cmap='coolwarm')
    axs[1].set_title(f'GWR Local Regression Coefficient - PC1')
    #axs[1].set_title(f'GWR Local Regression Coefficient - PC1\n{selected_y_var} of {study_area_name_only}')
    plt.colorbar(scatter2, ax=axs[1])

    scatter3 = axs[2].scatter(coords[:, 0], coords[:, 1], c=gdf_osm_projected[f'PC2_coef_{detected_tag}'], cmap='coolwarm')
    axs[2].set_title(f'GWR Local Regression Coefficient - PC2')
    #axs[2].set_title(f'GWR Local Regression Coefficient - PC2\n{selected_y_var} of {study_area_name_only}')
    plt.colorbar(scatter3, ax=axs[2])

    plt.tight_layout()

    map_save_path = os.path.join(
        gwr_output_dir, f'gwr_pca_maps_{detected_tag}_{study_area_name_only}.png'
    )
    plt.savefig(map_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"GWR maps saved at: {map_save_path}")

# --- Clear Function ---
def on_clear_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)

# --- Bind Buttons ---
analyze_button.on_click(on_analyze_button_clicked)
clear_button.on_click(on_clear_button_clicked)

# %%
display(gdf_osm_projected)

# %%
# Save the final GeoDataFrame as GeoJSON
gdf_osm_projected.to_file(os.path.join(gwr_output_dir, 'gdf_osm_projected_gwr_pca_results_passare.geojson'), driver='GeoJSON')
print(f"GeoJSON saved at: {gwr_output_dir}")

# %%
# Perform GWR + PCA Visualization (without recalculating anything!)

# --- Setup ---
gwr_output_dir = 'results/6_gwr/gwr_pca/folium_maps/'
os.makedirs(gwr_output_dir, exist_ok=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Utility Function ---
def extract_study_area_name(selected_file):
    study_area_name, _ = os.path.splitext(selected_file)
    tokens = study_area_name.replace("-", "_").split("_")
    try:
        idx_results = tokens.index('results')
        study_area_token = tokens[idx_results - 1]
    except ValueError:
        study_area_token = tokens[-1]
    return study_area_token.replace("-", " ").replace("_", " ").title()

# --- Widgets ---
map_selector = Dropdown(
    options=[],
    description='Select Map:',
    disabled=True
)

prepare_button = Button(description="Press here first to prepare maps", layout=Layout(width='auto', min_width='200px'))
run_button = Button(description="Display Selected GWR Map on Folium", layout=Layout(width='auto', min_width='200px'), disabled=True)
clear_button = Button(description="Clear Output", layout=Layout(width='auto', min_width='120px'))

output = Output()

box_layout = VBox([
    HBox([prepare_button, clear_button], layout=Layout(margin='10px 0')),
    map_selector,
    run_button,
    output
], layout=Layout(margin='0 0 10px 0'))

display(box_layout)

# --- Global ---
saved_maps = {}

# --- Main Function ---
def on_prepare_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)
    output.clear_output()
    with output:
        print("Preparing maps from existing GWR + PCA results...")

    study_area_name_only = extract_study_area_name(selected_file)

    gdf_to_map = gdf_osm_projected.to_crs(epsg=4326)

    # Detect GWR result columns automatically
    variables = {}
    for col in gdf_to_map.columns:
        if col.startswith(('Local_R2_', 'PC1_coef_', 'PC2_coef_')):
            # Extract Tag
            tag = col.split('_')[-1]  # leisure, building, amenity...
            if 'Local_R2' in col:
                title = f'GWR Local R² for {tag}'
            elif 'PC1_coef' in col:
                title = f'GWR Local Regression Coefficient - PC1 for {tag}'
            elif 'PC2_coef' in col:
                title = f'GWR Local Regression Coefficient - PC2 for {tag}'
            else:
                title = col
            variables[col] = title

    if not variables:
        with output:
            print("No GWR or PCA results found in the data!")
        return

    saved_maps.clear()

    for var_col, var_title in variables.items():
        values = gdf_to_map[var_col]
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
        cmap = plt.get_cmap('coolwarm')

        m = folium.Map(
            location=[gdf_to_map.geometry.centroid.y.mean(), gdf_to_map.geometry.centroid.x.mean()],
            zoom_start=14,
            control_scale=True
        )

        # Basemap
        TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            attr='CartoDB',
            name='CartoDB light'
        ).add_to(m)

        # Transparent grid
        GeoJson(
            gdf_to_map,
            style_function=lambda x: {
                'fillColor': 'white',
                'fill': True,
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.01,
                'lineOpacity': 0.6
            },
            tooltip=GeoJsonTooltip(
                fields=[var_col],
                aliases=[f'{var_title}: '],
                localize=True
            )
        ).add_to(m)

        # Circle markers colored by value
        for _, row in gdf_to_map.iterrows():
            centroid = row.geometry.centroid if isinstance(row.geometry, MultiPolygon) else row.geometry
            val = row[var_col]
            color = matplotlib.colors.to_hex(cmap(norm(val)))
            CircleMarker(
                location=[centroid.y, centroid.x],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1
            ).add_to(m)
        
        # Color scale for the legend
        colormap = cm.LinearColormap(
            colors=['blue', 'white', 'red'],
            vmin=values.min(),
            vmax=values.max()
        )
        colormap.caption = var_title
        colormap.add_to(m)

        # Title
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translate(-50%, 0);
        width: auto; border:2px solid gray; z-index:9999; font-size:16px; font-weight: bold;
        background: rgba(255, 255, 255, 0.8); text-align: center; padding: 5px;">
        {var_title}<br>{study_area_name_only}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Save map
        map_save_path = os.path.join(
            gwr_output_dir, f'gwr_pca_{var_col.lower()}_{study_area_name_only.replace(" ", "_").lower()}.html'
        )
        m.save(map_save_path)
        saved_maps[var_title] = m

    map_selector.options = list(saved_maps.keys())
    map_selector.disabled = False
    run_button.disabled = False

    with output:
        print(f"Maps ready! \nFiles saved at: {gwr_output_dir}")

# --- Display Map ---
def on_run_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)
    output.clear_output()
    selected_map_title = map_selector.value
    if selected_map_title in saved_maps:
        display(saved_maps[selected_map_title])

# --- Clear Output ---
def on_clear_button_clicked(b):
    clear_output(wait=True)
    display(box_layout)

# --- Bind ---
prepare_button.on_click(on_prepare_button_clicked)
run_button.on_click(on_run_button_clicked)
clear_button.on_click(on_clear_button_clicked)

# %% [markdown]
# #### PostGIS - Close the database connection

# %%
# Close the database connection
if conn and conn.closed == 0:
    # conexão ainda aberta
    with conn.cursor() as cur:
        ...
        cur.close()
    conn.close()
    print("Conexão com o banco de dados fechada.")
else:
    print("Conexão com o banco de dados já estava fechada ou não foi estabelecida.")


