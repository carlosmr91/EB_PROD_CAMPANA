# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:44:54 2024

@author: juan.melendez
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
def configure_page():
    st.set_page_config(page_title="MONITOREO PRODUCCIÓN")
    st.markdown("<h1 style='text-align: center; color: black;'>HISTÓRICO DE PRODUCCIÓN</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: green;'>TABLERO DE PRODUCCIÓN ALOCADA AGOSTO 2024</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: gray;'>⚙️ CONTROLADORES</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>________________________________</p>", unsafe_allow_html=True)

# CARGA DE ARCHIVO
def load_data():
    uploaded_file = st.sidebar.file_uploader("📂 Cargar archivo (formato CSV UTF-8)", type=["csv", "CSV", "TXT", "txt"])
    if uploaded_file:
        return pd.read_csv(uploaded_file, sep=",")
    st.toast("ARCHIVO NO CARGADO ❗❗")
    st.stop()

# PROCESAMIENTO DE DATOS
def process_data(df):
    df["Fecha"] = pd.to_datetime(df["Fecha"], format='%d/%m/%Y %H:%M')
    columns_of_interest = ["Pozo_Oficial", "CAMPO", "MAESTRA BATERIA", "MAESTRA CAMPANA", "Fecha", "NumeroMeses",
                            "AceiteAcumulado Mbbl", "AguaAcumulada Mbbl", "GasAcumulado MMpc","BrutoAcumulado Mbbl",
                            "AceiteDiario bpd", "AguaDiaria bpd", "GasDiario pcd", "BrutoDiario bpd", "RPM"]
    ofm_df = df[columns_of_interest]
    
    # Agrupación y fusión
    max_values = ofm_df.groupby("Pozo_Oficial")[["AceiteAcumulado Mbbl", "GasAcumulado MMpc", "AguaAcumulada Mbbl", "NumeroMeses"]].max().reset_index()
    max_values = pd.merge(max_values, ofm_df[["Pozo_Oficial", "MAESTRA CAMPANA"]].drop_duplicates(), on="Pozo_Oficial", how="left")
    
    return ofm_df, max_values

# CREACIÓN DE LA TABLA
def create_table(merged_df, campana_seleccion):
    campana_df = merged_df[merged_df["MAESTRA CAMPANA"].isin(campana_seleccion)]
    figTabla = go.Figure(data=[go.Table(
        header=dict(
            values=["CAMPAÑA","POZO", "MESES ACTIVO (meses)", "Np TOTAL (Mbbl)", "Gp TOTAL (MMpc)", "Wp TOTAL (Mbbl)"],
            fill_color='#E6E6E6',
            align='center',
            font=dict(size=12, color='black', family='Arial black'),
            line_color='darkslategray'
        ),
        cells=dict(
            values=[campana_df['MAESTRA CAMPANA'],campana_df['Pozo_Oficial'], campana_df['NumeroMeses'], campana_df['AceiteAcumulado Mbbl'], campana_df['GasAcumulado MMpc'], campana_df['AguaAcumulada Mbbl']],
            fill_color=[['#F0F0F0', '#E8E8E8']*len(campana_df)],
            align='right',
            font=dict(size=12, color='black', family='Arial'),
            line_color='darkslategray'
        )
    )])
    figTabla.update_layout(width=980, height=1000)
    return figTabla, campana_df

# CREACIÓN DE LOS GRÁFICOS
def create_donut_charts(campana_df):
    figDona = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])
    metrics = [("AceiteAcumulado Mbbl", "Np"),
               ("AguaAcumulada Mbbl", "Wp"),
               ("GasAcumulado MMpc", "Gp")]
    
    for i, (col, name) in enumerate(metrics):
        campana_df[col] = campana_df[col].astype(float)
        figDona.add_trace(go.Pie(labels=campana_df["Pozo_Oficial"], values=campana_df[col], name=name, marker=dict(colors=px.colors.qualitative.Pastel)), 1, i+1)
    
    figDona.update_traces(hole=.4, hoverinfo="label+percent+name")
    figDona.update_layout(
        title_text="PROPORCION PORCENTUAL DE LA PRODUCCIÓN ACUMULADA POR POZO",
        annotations=[dict(text=name, x=0.12 + 0.38 * i, y=0.5, font_size=25, showarrow=False) for i, (_, name) in enumerate(metrics)],
        font=dict(family="Arial", size=12, color="RebeccaPurple"),
        paper_bgcolor='#F0F0F0',
        plot_bgcolor='#E6E6E6',
        width=980,
        height=625
    )
    return figDona

def create_time_series_chart(ofm_df, campana_seleccion):
    filtered_df = ofm_df[ofm_df["MAESTRA CAMPANA"].isin(campana_seleccion)]
    time_series_df = filtered_df.groupby(["Fecha", "MAESTRA CAMPANA"]).sum().reset_index()

    figAceite = px.area(time_series_df, x="Fecha", y="AceiteDiario bpd", color="MAESTRA CAMPANA", 
                        labels={"AceiteDiario bpd": "Aceite Diario (bpd)", "Fecha": "Fecha", "MAESTRA CAMPANA": "Campaña"},
                        title="PRODUCCIÓN DIARIA DE ACEITE", color_discrete_sequence=px.colors.qualitative.Pastel)
    figAgua = px.area(time_series_df, x="Fecha", y="AguaDiaria bpd", color="MAESTRA CAMPANA", 
                      labels={"AguaDiaria bpd": "Agua Diaria (bpd)", "Fecha": "Fecha", "MAESTRA CAMPANA": "Campaña"},
                      title="PRODUCCIÓN DIARIA DE AGUA", color_discrete_sequence=px.colors.qualitative.Pastel)
    figGas = px.area(time_series_df, x="Fecha", y="GasDiario pcd", color="MAESTRA CAMPANA", 
                     labels={"GasDiario pcd": "Gas Diario (pcd)", "Fecha": "Fecha", "MAESTRA CAMPANA": "Campaña"},
                     title="PRODUCCIÓN DIARIA DE GAS", color_discrete_sequence=px.colors.qualitative.Pastel)
    
    active_wells_df = filtered_df[filtered_df["AceiteDiario bpd"] > 0]
    active_wells_count_df = active_wells_df.groupby(["MAESTRA CAMPANA", "Fecha"]).size().reset_index(name='Active_Wells')
    figActiveWells = px.area(active_wells_count_df, x="Fecha", y="Active_Wells", color="MAESTRA CAMPANA",
                            labels={"Active_Wells": "Pozos Activos", "Fecha": "Fecha","MAESTRA CAMPANA": "Campaña"},
                            title="CUENTA DE POZOS ACTIVOS", color_discrete_sequence=px.colors.qualitative.Pastel)

    for fig in [figAceite, figAgua, figGas, figActiveWells]:
        fig.update_layout(
            font=dict(family="Arial", size=12, color="RebeccaPurple"),
            paper_bgcolor='#EAEAEA',
            plot_bgcolor='#FFFFFF',
            width=980,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(size=12, family="Arial", color="black")),
            xaxis_rangeslider=dict(visible=True)
        )
        fig.update_xaxes(title_font=dict(size=14, family='Arial', color='black'))
        fig.update_yaxes(title_font=dict(size=14, family='Arial', color='black'))
    
    return figAceite, figAgua, figGas, figActiveWells

def create_bar_chart(campana_df):
    campana_df['Año_Campaña'] = pd.to_datetime(campana_df['MAESTRA CAMPANA']).dt.year
    ordered_pozos = campana_df.sort_values(by=['Año_Campaña', 'Pozo_Oficial'])['Pozo_Oficial'].unique()
    unique_campaigns = campana_df['MAESTRA CAMPANA'].unique()
    color_map = {campaign: px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)] for i, campaign in enumerate(unique_campaigns)}

    figBar = go.Figure()
    for year in campana_df['Año_Campaña'].unique():
        df_year = campana_df[campana_df['Año_Campaña'] == year]
        figBar.add_trace(go.Bar(
            x=df_year['Pozo_Oficial'],
            y=df_year['AceiteAcumulado Mbbl'],
            name=f'Año {year}',
            marker_color=[color_map[campaign] for campaign in df_year['MAESTRA CAMPANA']]
            
        ))

    figBar.update_layout(
        title='PRODUCCIÓN ACUMULADA POR POZO Y AÑO DE CAMPAÑA',
        xaxis_title='Pozo Oficial',
        yaxis_title='Np (Mbbls)',
        barmode='group',
        paper_bgcolor='#EAEAEA',
        plot_bgcolor='#FFFFFF',
        width=980,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(font=dict(size=12, family="Arial", color="black")),
        xaxis=dict(categoryorder='array', categoryarray=ordered_pozos),

    )
    figBar.update_xaxes(title_font=dict(size=14, family='Arial', color='black'))
    figBar.update_yaxes(title_font=dict(size=14, family='Arial', color='black'))
    return figBar


#---------------------
def crear_figura(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    """
    Crea un gráfico de líneas con la opción de agregar una traza de suma total.
    """
    # Calcular la suma de la columna seleccionada
    df_sum = df.groupby('Fecha').agg({y_col: 'sum'}).reset_index()
    
    # Calcular el rango máximo del eje y
    max_y = max(df[y_col].max(), df_sum[y_col].max()) * 1.1
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="Fecha", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
    
    # Añadir la suma total de los pozos seleccionados como una nueva traza
    fig.add_trace(go.Scatter(x=df_sum['Fecha'], y=df_sum[y_col],
                             mode=mode,
                             name='Suma Total',
                             marker=dict(color='black', symbol='cross', size=marker_size),
                             yaxis='y1'))
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=650,
        height=250,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
        yaxis2=dict(range=[0, max_y], title=f"Suma Total {yaxis_title}", 
                    side='right', overlaying='y', showgrid=False, zeroline=False),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    return fig


def crear_figura_rpm(df, selected_pozos):
    """
    Crea un gráfico de dispersión para RPM para cada pozo seleccionado.
    """
    fig = go.Figure()
    for pozo in selected_pozos:
        df_pozo = df[df['Pozo_Oficial'] == pozo]
        fig.add_trace(go.Scatter(x=df_pozo['Fecha'], y=df_pozo['RPM'],
                                mode='lines+markers',
                                name=pozo,
                                marker=dict(symbol='cross', size=5)))
    
    max_rpm = df['RPM'].max() * 1.1  # Ajustar el rango del eje Y
    fig.update_layout(
        title="RPM",
        width=650,
        height=250,
        paper_bgcolor="#ECECEC",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_rpm], title="RPM", side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
        legend=dict(font=dict(size=15, family="Calibri", color="black"))
    )
    
    return fig


def crear_figuraACUM(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    """
    Crea un gráfico de líneas con la opción de agregar una traza de suma total.
    """
    # Calcular la suma de la columna seleccionada
    #df_sum = df.groupby('Fecha').agg({y_col: 'sum'}).reset_index()
    
    # Calcular el rango máximo del eje y
    max_y = df[y_col].max()* 1.1
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="Fecha", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
    
    # Añadir la suma total de los pozos seleccionados como una nueva traza
    #fig.add_trace(go.Scatter(x=df_sum['Fecha'], y=df_sum[y_col],
    #                         mode=mode,
    #                         name='Suma Total',
    #                         marker=dict(color='black', symbol='cross', size=marker_size),
    #                         yaxis='y1'))
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=650,
        height=250,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
        #yaxis2=dict(range=[0, max_y], title=f"Suma Total {yaxis_title}", 
        #            side='right', overlaying='y', showgrid=False, zeroline=False),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    return fig


def crear_figuraACUMAPLI(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    """
    Crea un gráfico de áreas apiladas con "carry forward" para proyectar el mismo valor 
    de la columna entre fechas sin datos hasta la fecha más reciente.
    """
    # Verificar si y_col existe en el DataFrame
    if y_col not in df.columns:
        raise KeyError(f"La columna '{y_col}' no existe en el DataFrame")

    # Obtener la fecha máxima del DataFrame
    max_date = df['Fecha'].max()

    # Crear un DataFrame extendido con fechas que van desde la mínima hasta la fecha más reciente
    extended_dates = pd.date_range(start=df['Fecha'].min(), end=max_date, freq='D')
    extended_df = pd.DataFrame({'Fecha': extended_dates})

    # Inicializar una figura
    fig = go.Figure()

    # Iterar sobre cada pozo único
    for pozo in df[color_col].unique():
        # Filtrar datos para el pozo actual
        df_pozo = df[df[color_col] == pozo].copy()

        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Extender el DataFrame del pozo actual hasta la fecha más reciente
            extended_df_pozo = pd.merge(extended_df, df_pozo[['Fecha', y_col]], on='Fecha', how='left')

            # Aplicar carry forward: rellenar valores faltantes con el último valor conocido
            extended_df_pozo[y_col].fillna(method='ffill', inplace=True)

            # Si quedan valores NaN después de fillna, rellenarlos con 0 o el valor deseado
            extended_df_pozo[y_col].fillna(0, inplace=True)

            # Añadir la traza al gráfico
            fig.add_trace(
                go.Scatter(
                    x=extended_df_pozo['Fecha'],
                    y=extended_df_pozo[y_col],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                )
            )

    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=650,
        height=250,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(title=yaxis_title, side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )

    return fig


#-------------------------------


# ESTRUCTURA DE PÁGINA
def main():
    configure_page()
    df = load_data()
    

    with st.spinner("Procesando datos..."):
        ofm_df, merged_df = process_data(df)

    tabs = st.tabs([" CAMPAÑA ", " PRODUCCIÓN HISTÓRICA", "DINÁMICO"])

    with tabs[0]:
        campana_seleccion = st.multiselect("SELECCIONA CAMPAÑA", merged_df["MAESTRA CAMPANA"].unique())
        c1, c2 = st.columns(2)
        with c1:
            
            figTabla, campana_df = create_table(merged_df, campana_seleccion)
            if campana_seleccion:
                figAceite, figAgua, figGas, figActiveWells = create_time_series_chart(ofm_df, campana_seleccion)
                for fig in [figActiveWells, figAceite, figAgua, figGas]:
                    c1.plotly_chart(fig)

        with c2:
            if not campana_df.empty:
                c2.plotly_chart(create_bar_chart(campana_df))
                c2.plotly_chart(create_donut_charts(campana_df))
                csv_data = campana_df.to_csv(index=False)
                c2.download_button(
                    label="📥 Descargar tabla como CSV",
                    data=csv_data,
                    file_name="resultado_campana.csv",
                    mime="text/csv"
                )
                c2.write(figTabla)
                
                
        with tabs[1]:
            selected_pozos = st.multiselect("SELECCIONA POZO", ofm_df["Pozo_Oficial"].unique())
            c1, c2, c3 = st.columns(3)
            
            with c1:
                # Multiselect para seleccionar Pozos
                #selected_pozos = st.multiselect("SELECCIONA POZO", ofm_df["Pozo_Oficial"].unique())
            
                if selected_pozos:
                    # Filtrar datos por pozos seleccionados
                    df_filtrado = ofm_df[ofm_df["Pozo_Oficial"].isin(selected_pozos)]
                    
                    # Verificar si 'Fecha' tiene información y no contiene valores NaT
                    if df_filtrado['Fecha'].notna().any():
                        # Asegurarse de que el DataFrame esté ordenado por fecha y sin valores NaT en 'Fecha'
                        df_filtrado = df_filtrado.dropna(subset=['Fecha']).sort_values(by='Fecha')
                        
                        # Crear gráficos usando la función modularizada
                        figBruto = crear_figura(df_filtrado, 'BrutoDiario bpd', "Producción Bruta Diaria (bpd)", "Pozo_Oficial", 'black', 'markers', "Bruta (bpd)", "#ECECEC")
                        figNeta = crear_figura(df_filtrado, 'AceiteDiario bpd', "Producción Neta Diaria (bpd)", "Pozo_Oficial", 'black', 'markers', "Neta (bpd)", "#E5FDDF")
                        figAgua = crear_figura(df_filtrado, 'AguaDiaria bpd', "Producción de Agua Diaria (bpd)", "Pozo_Oficial", 'black', 'markers', "Agua (bpd)", "#DFF9FD")
                        figGas = crear_figura(df_filtrado, 'GasDiario pcd', "Producción de Gas Diaria (pcd)", "Pozo_Oficial", 'black', 'markers', "Gas (pcd)", "#FEEDE8")
                        figRPM = crear_figura_rpm(df_filtrado, selected_pozos)
                        #figBrutoAC = crear_figuraACUM(df_filtrado, 'BrutoAcumulado Mbbl', "Producción de Bruta Acumulada (Mbpd)", "Pozo_Oficial", 'black', 'markers', "Bruta (bpd)", "#ECECEC")
                        
                        # Mostrar los gráficos en la columna 1
                        for fig in [figBruto, figNeta, figAgua, figGas, figRPM]:
                            c1.plotly_chart(fig)

        with c2:
            figBrutoAC = crear_figuraACUM(df_filtrado, 'BrutoAcumulado Mbbl', "Producción Bruta Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Bruta (Mbbl)", "#ECECEC")
            figNetaAC = crear_figuraACUM(df_filtrado, 'AceiteAcumulado Mbbl', "Producción Neta Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Neta (Mbbl)", "#E5FDDF")
            figAguaAC = crear_figuraACUM(df_filtrado, 'AguaAcumulada Mbbl', "Producción de Agua Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Agua (Mbbl)", "#DFF9FD")
            figGasAC = crear_figuraACUM(df_filtrado, 'GasAcumulado MMpc', "Producción de Gas Acumulada (MMpc)", "Pozo_Oficial", 'black', 'markers', "Gas (MMpc)", "#FEEDE8")
            for fig in [figBrutoAC,figNetaAC,figAguaAC,figGasAC]:
                c2.plotly_chart(fig)
        
        with c3:
            figBrutoACAP = crear_figuraACUMAPLI(df_filtrado, 'BrutoAcumulado Mbbl', "Producción Bruta Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Bruta (Mbbl)", "#ECECEC")
            figNetaACAP = crear_figuraACUMAPLI(df_filtrado, 'AceiteAcumulado Mbbl', "Producción Neta Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Neta (Mbbl)", "#E5FDDF")
            figAguaACAP = crear_figuraACUMAPLI(df_filtrado, 'AguaAcumulada Mbbl', "Producción de Agua Acumulada (Mbbl)", "Pozo_Oficial", 'black', 'markers', "Agua (Mbbl)", "#DFF9FD")
            figGasACAP = crear_figuraACUMAPLI(df_filtrado, 'GasAcumulado MMpc', "Producción de Gas Acumulada (MMpc)", "Pozo_Oficial", 'black', 'markers', "Gas (MMpc)", "#FEEDE8")
            for fig in [figBrutoACAP,figNetaACAP,figAguaACAP,figGasACAP]:
                c3.plotly_chart(fig)


            
if __name__ == "__main__":
    main()