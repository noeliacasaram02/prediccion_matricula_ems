"""Consulta e inferencia del modelo experimental de matrícula EMS."""
from pathlib import Path
from html import escape
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_disminucion_relevante.joblib"
DATA_PATH = BASE_DIR / "data" / "curated" / "base_inferencia_disminucion_relevante.csv"
HISTORY_PATH = BASE_DIR / "data" / "curated" / "historico_matricula_municipal.csv"
CICLOS_HISTORICOS = [f"{anio}-{anio + 1}" for anio in range(2019, 2025)]
CAMPOS_FORMULARIO = ["matricula_actual", "matricula_anterior", "docentes_actual", "docentes_anterior",
    "escuelas_actual", "escuelas_anterior", "poblacion", "escolaridad", "internet", "computadora"]
VARIABLES_ESPERADAS = [
    "matricula_t", "cambio_anterior", "cambio_pct_anterior", "docentes_t",
    "cambio_docentes_anterior", "escuelas_unicas_t", "cambio_escuelas_anterior",
    "poblacion_15_19_t", "graproes_2020", "pct_internet_2020", "pct_computadora_2020",
]
LOGGER = logging.getLogger(__name__)
AZUL_OSCURO = "#006ECF"
AZUL_SECUNDARIO = "#023BE6"
ROSA_PRINCIPAL = "#F9547F"
ROSA_OSCURO = "#D1108D"
ROSA_CLARO = "#FDECF2"
ROSA_MUY_CLARO = "#FFF6F9"
AZUL_MUY_CLARO = "#F2F6FF"
VERDE_ALERTA_OK = "#11AA61"
GRIS_TEXTO = "#5F6B85"
GRIS_BORDE = "#E5E9F2"
FONDO = "#FFFFFF"


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Validar el artefacto, sin entrenarlo ni modificarlo."""
    modelo = joblib.load(MODEL_PATH)
    necesarias = {"pipeline", "variables", "umbral_probabilidad", "umbral_cambio_pct", "version_sklearn"}
    if not isinstance(modelo, dict) or not necesarias.issubset(modelo):
        raise ValueError("El modelo no contiene la información requerida.")
    if sklearn.__version__ != modelo["version_sklearn"]:
        raise ValueError("La versión del entorno no coincide con la del modelo.")
    if modelo["variables"] != VARIABLES_ESPERADAS:
        raise ValueError("El orden de variables no coincide.")
    pipeline = modelo["pipeline"]
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2:
        raise ValueError("Se esperaba un Pipeline de dos pasos.")
    if not isinstance(pipeline.steps[0][1], SimpleImputer) or pipeline.steps[0][1].strategy != "median":
        raise ValueError("La imputación no corresponde al modelo esperado.")
    if not isinstance(pipeline.steps[1][1], RandomForestClassifier):
        raise ValueError("El clasificador no corresponde al modelo esperado.")
    if list(pipeline.feature_names_in_) != modelo["variables"] or set(pipeline.classes_) != {0, 1}:
        raise ValueError("Las entradas o clases no son las esperadas.")
    if modelo["umbral_probabilidad"] != 0.20 or not np.isfinite(modelo["umbral_cambio_pct"]) or modelo["umbral_cambio_pct"] >= 0:
        raise ValueError("Los umbrales no son válidos.")
    return modelo


@st.cache_data(show_spinner=False)
def cargar_datos():
    """Conservar claves, filas y faltantes de la base de inferencia."""
    datos = pd.read_csv(DATA_PATH, dtype={"CVEGEO": "string"}, float_precision="round_trip")
    identificacion = ["CVEGEO", "nombre_entidad", "nombre_municipio", "ciclo_actual"]
    if datos.empty or not set(identificacion + VARIABLES_ESPERADAS).issubset(datos.columns):
        raise ValueError("La base no contiene las columnas necesarias.")
    if datos[identificacion].isna().any().any() or datos["CVEGEO"].duplicated().any():
        raise ValueError("La identificación municipal requiere revisión.")
    if not datos["CVEGEO"].str.fullmatch(r"[0-9]{5}").all():
        raise ValueError("Las claves municipales requieren revisión.")
    for variable in VARIABLES_ESPERADAS:
        datos[variable] = pd.to_numeric(datos[variable], errors="raise")
    if np.isinf(datos[VARIABLES_ESPERADAS].to_numpy(dtype=float)).any():
        raise ValueError("La base contiene valores no válidos.")
    return datos.sort_values(["nombre_entidad", "nombre_municipio"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cargar_historico():
    historico = pd.read_csv(HISTORY_PATH, dtype={"CVEGEO": "string"})
    necesarias = {"CVEGEO", "nombre_entidad", "nombre_municipio", "ciclo", "matricula", "docentes", "escuelas_unicas"}
    if not necesarias.issubset(historico.columns) or historico.empty:
        raise ValueError("El histórico no contiene las columnas necesarias.")
    if historico.duplicated(["CVEGEO", "ciclo"]).any() or not historico["ciclo"].isin(CICLOS_HISTORICOS).all():
        raise ValueError("Revisar los ciclos o duplicados del histórico.")
    if historico[list(necesarias)].isna().any().any():
        raise ValueError("El histórico contiene registros incompletos.")
    return historico.sort_values(["CVEGEO", "ciclo"]).reset_index(drop=True)


def entradas_del_municipio(fila, historico):
    """Recuperar el antecedente observado, nunca reconstruirlo por diferencias."""
    anio = int(fila["ciclo_actual"].split("-")[0])
    ciclo_previo = f"{anio - 1}-{anio}"
    previo = historico.loc[historico["CVEGEO"].eq(fila["CVEGEO"]) & historico["ciclo"].eq(ciclo_previo)]
    antecedente = previo.iloc[0] if len(previo) == 1 else {}
    entradas = {
        "matricula_actual": fila["matricula_t"], "matricula_anterior": antecedente.get("matricula", None),
        "docentes_actual": fila["docentes_t"], "docentes_anterior": antecedente.get("docentes", None),
        "escuelas_actual": fila["escuelas_unicas_t"], "escuelas_anterior": antecedente.get("escuelas_unicas", None),
        "poblacion": fila["poblacion_15_19_t"], "escolaridad": fila["graproes_2020"],
        "internet": fila["pct_internet_2020"], "computadora": fila["pct_computadora_2020"],
    }
    for campo, valor in entradas.items():
        if pd.isna(valor):
            entradas[campo] = None
        elif campo in ["escolaridad", "internet", "computadora"]:
            entradas[campo] = float(valor)
        else:
            entradas[campo] = int(valor)
    return entradas


def mostrar_historico(fila, historico):
    st.subheader("Evolución reciente de la matrícula")
    serie = historico.loc[historico["CVEGEO"].eq(fila["CVEGEO"])].set_index("ciclo")["matricula"].reindex(CICLOS_HISTORICOS)
    if serie.notna().any():
        valores = [int(valor) if pd.notna(valor) else None for valor in serie]
        # El gráfico nativo recibe solo seis observaciones; los ciclos ausentes no se interpolan.
        st.echarts_chart({
            "animation": False,
            "backgroundColor": FONDO,
            "grid": {"left": 75, "right": 35, "top": 40, "bottom": 60},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": CICLOS_HISTORICOS, "name": "ciclo escolar",
                "nameLocation": "middle", "nameGap": 35, "axisLabel": {"interval": 0}},
            "yAxis": {"type": "value", "name": "matrícula", "min": 0,
                "splitLine": {"lineStyle": {"color": GRIS_BORDE}}},
            "series": [{"type": "line", "data": valores, "connectNulls": False,
                "showSymbol": True, "symbolSize": 8, "lineStyle": {"color": ROSA_PRINCIPAL, "width": 2},
                "itemStyle": {"color": ROSA_PRINCIPAL},
                "areaStyle": {"color": ROSA_CLARO, "opacity": 0.5},
                "label": {"show": True, "position": "top", "color": AZUL_OSCURO, "fontSize": 11}}],
            "media": [{"query": {"maxWidth": 640}, "option": {
                "grid": {"left": 50, "right": 15, "top": 35, "bottom": 75},
                "xAxis": {"nameGap": 50, "axisLabel": {"interval": 0, "fontSize": 10, "rotate": 25}},
                "yAxis": {"nameTextStyle": {"fontSize": 11}, "axisLabel": {"fontSize": 10}},
                "series": [{"label": {"fontSize": 10}, "symbolSize": 7}],
            }}],
        }, width="stretch", height=290, alt="Matrícula municipal observada en los seis ciclos 2019–2020 a 2024–2025.")
        if serie.isna().any():
            st.caption("Algunos ciclos no tienen registro disponible; no se completan con valores estimados.")
    else:
        st.info("No hay registros históricos disponibles para este municipio.")
    cambio = formato_numero(fila["cambio_pct_anterior"], 2)
    st.caption("Cambio del último ciclo: " + cambio + (" %" if pd.notna(fila["cambio_pct_anterior"]) else ""))
    st.caption("La gráfica muestra datos observados. La señal para el siguiente ciclo es una clasificación de riesgo y no una proyección del número de alumnos.")


def calcular_variables(entradas, variables):
    """Repetir las fórmulas y las unidades del cuarto experimento."""
    for nombre, valor in entradas.items():
        if valor is None or not np.isfinite(valor) or valor < 0:
            raise ValueError("Completa todos los campos con números no negativos.")
        if nombre in ["internet", "computadora"] and valor > 100:
            raise ValueError("Los porcentajes deben estar entre 0 y 100.")
    actual = entradas["matricula_actual"]
    anterior = entradas["matricula_anterior"]
    registro = {
        "matricula_t": actual,
        "cambio_anterior": actual - anterior,
        "cambio_pct_anterior": 100 * (actual - anterior) / anterior if anterior != 0 else np.nan,
        "docentes_t": entradas["docentes_actual"],
        "cambio_docentes_anterior": entradas["docentes_actual"] - entradas["docentes_anterior"],
        "escuelas_unicas_t": entradas["escuelas_actual"],
        "cambio_escuelas_anterior": entradas["escuelas_actual"] - entradas["escuelas_anterior"],
        "poblacion_15_19_t": entradas["poblacion"],
        "graproes_2020": entradas["escolaridad"],
        "pct_internet_2020": entradas["internet"],
        "pct_computadora_2020": entradas["computadora"],
    }
    if variables != list(registro):
        raise ValueError("Las entradas no coinciden con el modelo.")
    return pd.DataFrame([registro], columns=variables)


def predecir_riesgo(registro, modelo):
    """Obtener la probabilidad de clase 1 y aplicar el umbral guardado."""
    entradas = registro.loc[:, modelo["variables"]]
    if len(entradas) != 1 or np.isinf(entradas.to_numpy(dtype=float)).any():
        raise ValueError("Se necesita un registro válido para evaluar.")
    pipeline = modelo["pipeline"]
    indice = list(pipeline.classes_).index(1)
    probabilidad = float(pipeline.predict_proba(entradas)[0, indice])
    if not np.isfinite(probabilidad) or not 0 <= probabilidad <= 1:
        raise ValueError("No se obtuvo una probabilidad válida.")
    return probabilidad, probabilidad >= modelo["umbral_probabilidad"]


def mostrar_resultado(probabilidad, alerta, modelo, clave="consulta"):
    with st.container(border=True, key="resultado_" + clave):
        st.subheader("Resultado del modelo")
        st.metric("Probabilidad estimada", f"{probabilidad * 100:.1f} %")
        estado = "Riesgo identificado" if alerta else "Sin alerta"
        color_estado = ROSA_OSCURO if alerta else VERDE_ALERTA_OK
        st.html(f'<p role="status" style="color:{color_estado};font-size:1.2rem;font-weight:650;margin:0">{estado}</p>')
        if alerta:
            st.write("El municipio supera el umbral de priorización utilizado por el modelo. Se recomienda revisar con mayor detalle la evolución de su matrícula y su contexto educativo.")
        else:
            st.write("El municipio no supera el umbral de priorización utilizado por el modelo. Esto no garantiza que la matrícula no pueda disminuir.")
        comparacion = "iguala o supera" if alerta else "es menor que"
        st.caption(f"{probabilidad:.1%} {comparacion} el umbral de priorización de {modelo['umbral_probabilidad']:.0%}; "
            + ("el caso se marca para revisión." if alerta else "el caso aparece sin alerta."))
    st.caption("La probabilidad estima la pertenencia al grupo de riesgo; no indica cuánto disminuiría la matrícula.")
    if clave == "nueva":
        st.caption("La probabilidad estimada no representa cuánto disminuirá la matrícula. Representa la señal de riesgo calculada por el modelo a partir de patrones históricos.")
    # Barra sencilla con una marca independiente para el umbral; no representa una caída porcentual.
    st.html(f'''<div role="img" aria-label="Probabilidad estimada {probabilidad:.1%}; umbral de priorización {modelo['umbral_probabilidad']:.0%}">
        <div style="position:relative;height:12px;background:#FDECF2;border-radius:5px;margin:8px 0 25px">
        <div style="width:{probabilidad * 100:.6f}%;height:100%;background:#F9547F;border-radius:5px"></div>
        <span style="position:absolute;left:{modelo['umbral_probabilidad'] * 100:.6f}%;top:-4px;height:20px;border-left:2px solid #011C6B"></span>
        <span style="position:absolute;left:{modelo['umbral_probabilidad'] * 100:.6f}%;top:15px;transform:translateX(-50%);font-size:12px;color:#011C6B">{modelo['umbral_probabilidad']:.0%}</span>
        </div></div>''')
    st.caption(f"Umbral de priorización del modelo: {modelo['umbral_probabilidad']:.0%}. En validación se priorizó detectar disminuciones relevantes, aun aceptando más falsas alertas.")


def formato_numero(valor, decimales=0):
    if pd.isna(valor):
        return "No disponible"
    return f"{valor:,.{decimales}f}"


def consultar_municipio(datos, modelo, historico):
    with st.container(border=True, key="ejemplos_consulta"):
        st.markdown("**¿Quieres probar la aplicación?**")
        st.caption("Selecciona Aguascalientes, Aguascalientes, o Pantelhó, Chiapas. La probabilidad se recalcula siempre con el modelo.")
    izquierda, derecha = st.columns(2)
    entidad = izquierda.selectbox("Entidad", sorted(datos["nombre_entidad"].unique()))
    municipios = datos.loc[datos["nombre_entidad"].eq(entidad)]
    nombres = municipios.set_index("CVEGEO")["nombre_municipio"].to_dict()
    clave = derecha.selectbox("Municipio", municipios["CVEGEO"].tolist(), format_func=nombres.get)
    st.html(f'<p class="seleccion-movil">{escape(nombres[clave])} · {escape(entidad)}</p>')
    fila = municipios.loc[municipios["CVEGEO"].eq(clave)].iloc[0]
    st.subheader(fila["nombre_municipio"])
    st.caption(f"{fila['nombre_entidad']} · Ciclo actual {fila['ciclo_actual']} · Señal para {modelo.get('ciclo_a_estimar', 'el siguiente ciclo')}")
    resumen, resultado = st.columns([1.2, 1], gap="large")
    with resumen:
        a, b = st.columns(2)
        with a.container(border=True, key="tarjeta_matricula"):
            st.metric("Matrícula actual", formato_numero(fila["matricula_t"]))
        with b.container(border=True, key="tarjeta_cambio"):
            st.metric("Cambio respecto al ciclo anterior", formato_numero(fila["cambio_anterior"]))
        sufijo = " %" if pd.notna(fila["cambio_pct_anterior"]) else ""
        st.caption("Cambio porcentual anterior: " + formato_numero(fila["cambio_pct_anterior"], 2) + sufijo)
        a, b = st.columns(2)
        with a.container(border=True, key="tarjeta_docentes"):
            st.metric("Docentes actuales", formato_numero(fila["docentes_t"]))
        with b.container(border=True, key="tarjeta_escuelas"):
            st.metric("Escuelas actuales", formato_numero(fila["escuelas_unicas_t"]))
        with st.container(border=True, key="tarjeta_poblacion"):
            st.metric("Población de 15–19 años", formato_numero(fila["poblacion_15_19_t"]))
        if fila[modelo["variables"]].isna().any():
            st.caption("Algunos datos no están disponibles. El modelo utiliza su imputación histórica para completar las entradas faltantes.")
        with st.expander("Contexto municipal 2020"):
            st.write("Grado promedio de escolaridad: " + formato_numero(fila["graproes_2020"], 2))
            st.write("Viviendas con internet (%): " + formato_numero(fila["pct_internet_2020"], 2))
            st.write("Viviendas con computadora (%): " + formato_numero(fila["pct_computadora_2020"], 2))
    with resultado:
        try:
            registro = municipios.loc[municipios["CVEGEO"].eq(clave), modelo["variables"]]
            probabilidad, alerta = predecir_riesgo(registro, modelo)
            if "probabilidad_disminucion_relevante" in fila and pd.notna(fila["probabilidad_disminucion_relevante"]):
                if not np.isclose(probabilidad, fila["probabilidad_disminucion_relevante"], atol=1e-12, rtol=0):
                    LOGGER.warning("La comprobación interna de probabilidad requiere revisión.")
            mostrar_resultado(probabilidad, alerta, modelo)
        except Exception:
            LOGGER.exception("No se pudo evaluar el municipio.")
            st.error("No fue posible evaluar este municipio. Revisa la disponibilidad de sus datos e intenta nuevamente.")
    mostrar_historico(fila, historico)


def poner_datos_en_formulario(fila, historico):
    for campo, valor in entradas_del_municipio(fila, historico).items():
        st.session_state["entrada_" + campo] = valor
    st.session_state["municipio_cargado"] = f"{fila['nombre_municipio']}, {fila['nombre_entidad']} · ciclo actual {fila['ciclo_actual']}"


def buscar_ejemplo_cdmx(datos):
    opciones = datos.loc[datos["nombre_entidad"].eq("Ciudad de México")]
    if opciones.empty:
        return None
    preferido = opciones.loc[opciones["nombre_municipio"].str.casefold().eq("álvaro obregón")]
    return preferido.iloc[0] if not preferido.empty else opciones.iloc[0]


def nueva_evaluacion(modelo, datos, historico):
    st.subheader("Evaluar un registro nuevo")
    with st.container(border=True, key="guia_evaluacion"):
        st.markdown("**¿Cómo usar esta pestaña?**")
        st.write("Permite evaluar un registro nuevo con el mismo modelo. Puedes partir de los datos de un municipio existente y modificarlos, o capturar los valores manualmente. La aplicación calcula automáticamente los cambios entre ambos ciclos antes de generar la estimación.")
        st.markdown("1. Parte de un ejemplo o captura los datos de dos ciclos consecutivos del mismo municipio.\n2. Revisa los datos del ciclo actual y del ciclo inmediatamente anterior.\n3. Pulsa ‘Evaluar riesgo’ para obtener la probabilidad estimada y la señal de priorización.")
    # La primera visita presenta un ejemplo real, sin ejecutar la predicción del formulario.
    ejemplo = buscar_ejemplo_cdmx(datos)
    if "ejemplo_inicial_cargado" not in st.session_state:
        st.session_state["modo_evaluacion"] = "Partir de un municipio existente"
        st.session_state["modo_formulario_anterior"] = "Partir de un municipio existente"
        st.session_state["ejemplo_inicial_cargado"] = True
        if ejemplo is not None:
            poner_datos_en_formulario(ejemplo, historico)
            st.session_state["entidad_evaluacion"] = ejemplo["nombre_entidad"]
            st.session_state["municipio_evaluacion"] = ejemplo["CVEGEO"]
    st.markdown("**Probar con un ejemplo**")
    cargar_ejemplo = st.button("Cargar ejemplo de Ciudad de México", key="cargar_cdmx", disabled=ejemplo is None)
    st.caption("Este botón carga un ejemplo con datos existentes para mostrar cómo funciona la evaluación. Puedes modificar cualquiera de los valores antes de ejecutar el modelo.")
    if cargar_ejemplo:
        st.session_state["modo_evaluacion"] = "Partir de un municipio existente"
    modo = st.radio("¿Cómo quieres empezar?", ["Partir de un municipio existente", "Capturar datos manualmente"], key="modo_evaluacion")
    if st.session_state.get("modo_formulario_anterior") != modo:
        for campo in CAMPOS_FORMULARIO:
            st.session_state["entrada_" + campo] = None
        st.session_state["municipio_cargado"] = None
        st.session_state["modo_formulario_anterior"] = modo
    if cargar_ejemplo and ejemplo is not None:
        poner_datos_en_formulario(ejemplo, historico)
        st.session_state["entidad_evaluacion"] = ejemplo["nombre_entidad"]
        st.session_state["municipio_evaluacion"] = ejemplo["CVEGEO"]
    if modo == "Partir de un municipio existente":
        a, b = st.columns(2)
        entidad = a.selectbox("Entidad", sorted(datos["nombre_entidad"].unique()), key="entidad_evaluacion")
        opciones = datos.loc[datos["nombre_entidad"].eq(entidad)]
        nombres = opciones.set_index("CVEGEO")["nombre_municipio"].to_dict()
        clave = b.selectbox("Municipio", opciones["CVEGEO"].tolist(), format_func=nombres.get, key="municipio_evaluacion")
        st.html(f'<p class="seleccion-movil">{escape(nombres[clave])} · {escape(entidad)}</p>')
        if st.button("Cargar datos del municipio", key="cargar_municipio"):
            fila = opciones.loc[opciones["CVEGEO"].eq(clave)].iloc[0]
            poner_datos_en_formulario(fila, historico)
        if st.session_state.get("municipio_cargado"):
            st.caption("Datos cargados de " + st.session_state["municipio_cargado"])
        else:
            st.caption("Selecciona el municipio y pulsa ‘Cargar datos del municipio’ para llenar los campos.")
        st.caption("Puedes modificar cualquier valor para simular nueva información antes de evaluar el riesgo. Los datos no disponibles quedan vacíos; complétalos antes de evaluar.")
    else:
        st.caption("También puedes llenar todos los campos manualmente. Los datos no tienen que corresponder a un municipio ya registrado en la base.")
    st.caption("Ejemplos para probar: Aguascalientes, Aguascalientes, para observar un caso sin alerta; Pantelhó, Chiapas, para observar un caso con alerta. La estimación siempre se recalcula con el modelo.")
    with st.form("nueva_evaluacion"):
        actual, anterior = st.columns(2)
        actual.markdown("**Ciclo actual**")
        anterior.markdown("**Ciclo inmediatamente anterior**")
        entradas = {}
        entradas["matricula_actual"] = actual.number_input("Matrícula ciclo actual", min_value=0, value=None, step=1, key="entrada_matricula_actual")
        entradas["matricula_anterior"] = anterior.number_input("Matrícula ciclo anterior", min_value=0, value=None, step=1, key="entrada_matricula_anterior")
        entradas["docentes_actual"] = actual.number_input("Docentes ciclo actual", min_value=0, value=None, step=1, key="entrada_docentes_actual")
        entradas["docentes_anterior"] = anterior.number_input("Docentes ciclo anterior", min_value=0, value=None, step=1, key="entrada_docentes_anterior")
        entradas["escuelas_actual"] = actual.number_input("Escuelas ciclo actual", min_value=0, value=None, step=1, key="entrada_escuelas_actual")
        entradas["escuelas_anterior"] = anterior.number_input("Escuelas ciclo anterior", min_value=0, value=None, step=1, key="entrada_escuelas_anterior")
        st.markdown("**Población y contexto municipal**")
        a, b = st.columns(2)
        entradas["poblacion"] = a.number_input("Población de 15–19 años", min_value=0, value=None, step=1, help="Del año de inicio del ciclo actual.", key="entrada_poblacion")
        entradas["escolaridad"] = b.number_input("Grado promedio de escolaridad 2020", min_value=0.0, value=None, step=0.1, key="entrada_escolaridad")
        entradas["internet"] = a.number_input("Porcentaje de viviendas con internet 2020", min_value=0.0, max_value=100.0, value=None, step=0.1, key="entrada_internet")
        entradas["computadora"] = b.number_input("Porcentaje de viviendas con computadora 2020", min_value=0.0, max_value=100.0, value=None, step=0.1, key="entrada_computadora")
        st.caption("Si la matrícula anterior es cero, el cambio porcentual se deja sin valor para que el modelo lo impute.")
        evaluar = st.form_submit_button("Evaluar riesgo", type="primary")
    if evaluar:
        try:
            registro = calcular_variables(entradas, modelo["variables"])
        except (ValueError, TypeError, KeyError):
            st.error("Completa todos los campos con valores no negativos y porcentajes entre 0 y 100.")
            return
        try:
            probabilidad, alerta = predecir_riesgo(registro, modelo)
            mostrar_resultado(probabilidad, alerta, modelo, clave="nueva")
        except Exception:
            LOGGER.exception("No se pudo evaluar el registro nuevo.")
            st.error("No fue posible completar la evaluación. Revisa los valores e intenta nuevamente.")
    with st.container(border=True, key="ejemplo_rapido"):
        st.markdown("**Ejemplo rápido**")
        st.caption("Si un municipio tuvo 1,200 alumnos en el ciclo actual y 1,280 en el ciclo anterior, el sistema calcula automáticamente el cambio de matrícula y utiliza esta información junto con las demás variables para estimar el nivel de riesgo. Es solo un ejemplo explicativo, no una predicción real.")


def main():
    st.set_page_config(page_title="Riesgo de matrícula EMS", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""<style>
    .stApp {background: #FFFFFF; color: #011C6B; color-scheme: light;}
    .block-container {max-width: 1180px; padding-top: 2rem;}
    h1, h2, h3 {color: #006ECF;}
    h1 {border-left: 4px solid #F9547F; padding-left: 14px;}
    [data-testid="stCaptionContainer"] {color: #5F6B85;}
    [data-testid="stMetricValue"] {color: #011C6B; font-weight: 650;}
    [class*="st-key-tarjeta_"], [class*="st-key-resultado_"] {
        background: #FFF6F9; border: 1px solid #E5E9F2; border-radius: 12px; padding: 16px;
    }
    .st-key-tarjeta_docentes, .st-key-tarjeta_poblacion,
    .st-key-guia_evaluacion, .st-key-ejemplos_consulta, .st-key-ejemplo_rapido {background: #F2F6FF;}
    [data-baseweb="input"], [data-baseweb="select"] > div {background: #FFFFFF; color: #011C6B;}
    button[kind="primary"] {background-color: #F9547F; color: white; border-color: #F9547F;}
    button[kind="primary"]:hover {background-color: #D1108D; color: white; border-color: #D1108D;}
    [data-baseweb="tab-highlight"] {background-color: #F9547F;}
    [role="tab"][aria-selected="true"] {color: #D1108D;}
    [data-testid="stElementContainer"]:has(> [data-testid="stHtml"] .seleccion-movil) {display: none;}
    /* Solo disposición y tamaños: el escritorio y la paleta se conservan. */
    @media (min-width: 641px) and (max-width: 900px) {
        .block-container {padding-left: 1.5rem; padding-right: 1.5rem;}
        [data-testid="stHorizontalBlock"] {gap: 1rem;}
        [data-testid="stColumn"] {min-width: 0;}
        [class*="st-key-tarjeta_"], [class*="st-key-resultado_"] {padding: 12px;}
        [data-testid="stMetricLabel"] p {white-space: normal; overflow-wrap: anywhere;}
        [data-testid="stMetricValue"] {font-size: 1.9rem;}
    }
    @media (max-width: 640px) {
        .block-container {padding: 1rem 1rem 2rem;}
        h1 {font-size: 1.8rem !important; line-height: 1.15; padding-left: 10px;}
        h2 {font-size: 1.45rem !important; line-height: 1.25;}
        h3 {font-size: 1.2rem !important; line-height: 1.3;}
        h1, h2, h3, p, label, summary {overflow-wrap: anywhere; word-break: normal;}
        [data-testid="stHorizontalBlock"] {flex-direction: column; align-items: stretch; gap: .8rem;}
        [data-testid="stColumn"] {width: 100% !important; min-width: 0 !important; flex: 1 1 100% !important;}
        [class*="st-key-tarjeta_"], [class*="st-key-resultado_"] {
            width: 100%; box-sizing: border-box; min-width: 0; padding: 12px;
        }
        [data-testid="stMetricValue"] {font-size: 1.7rem; white-space: normal; overflow-wrap: anywhere;}
        [data-testid="stMetricLabel"] p {white-space: normal; overflow: visible; text-overflow: clip;}
        [data-testid="stSelectbox"], [data-testid="stNumberInput"],
        [data-testid="stExpander"], [data-testid="stHtml"] {min-width: 0; max-width: 100%;}
        [data-baseweb="select"] > div {min-height: 44px; height: auto;}
        [data-baseweb="select"] > div > div:first-child {white-space: normal; overflow-wrap: anywhere;}
        [data-testid="stSelectbox"] input[role="combobox"] {min-height: 44px; font-size: 16px; min-width: 0;}
        [role="option"] {white-space: normal; overflow-wrap: anywhere; height: auto;}
        [data-testid="stElementContainer"]:has(> [data-testid="stHtml"] .seleccion-movil) {display: block;}
        .seleccion-movil {font-size: .9rem; margin: 0; overflow-wrap: anywhere; word-break: normal;}
        [data-testid="stNumberInput"] input {font-size: 16px; min-height: 44px; min-width: 0;}
        [data-testid="stNumberInput"] button {min-height: 44px; min-width: 36px;}
        [role="tablist"] {gap: .25rem; flex-wrap: wrap; height: auto;}
        [role="tab"] {flex: 1 1 45%; min-width: 0; min-height: 44px; height: auto; padding: .5rem .3rem;}
        [role="tab"] p {white-space: normal; font-size: .95rem;}
        [data-testid="stButton"], [data-testid="stFormSubmitButton"] {width: 100%;}
        [data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
        [data-testid="stElementContainer"]:has(> [data-testid="stFormSubmitButton"]),
        [data-testid="stElementContainer"]:has(> div > [data-testid="stFormSubmitButton"]) {width: 100% !important;}
        [data-testid="stElementContainer"]:has(> div > [data-testid="stFormSubmitButton"]) > div {width: 100% !important;}
        [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
            width: 100%; min-height: 44px; white-space: normal; height: auto;
        }
        [data-testid="stButton"] button p, [data-testid="stFormSubmitButton"] button p {white-space: normal;}
        [data-testid="stFormSubmitButton"] button {background: #F9547F; border-color: #F9547F; color: white;}
        [data-testid="stFormSubmitButton"] button:hover {background: #D1108D; border-color: #D1108D;}
        [data-testid="stExpander"] summary {min-height: 44px;}
    }
    </style>""", unsafe_allow_html=True)
    st.caption("EDUCACIÓN MEDIA SUPERIOR · MODELO EXPERIMENTAL")
    st.title("Riesgo de disminución relevante de matrícula EMS")
    st.write("Aplicación experimental para identificar municipios que podrían requerir una revisión más detallada de su evolución de matrícula.")
    try:
        modelo = cargar_modelo()
        datos = cargar_datos()
        historico = cargar_historico()
    except Exception:
        LOGGER.exception("No se pudieron cargar los recursos.")
        st.error("No fue posible cargar los recursos necesarios. Comprueba que la instalación esté completa y que el entorno sea compatible con el modelo.")
        st.stop()
    st.write("El modelo identifica municipios con características similares a casos históricos de disminución relevante. "
        f"En este proyecto, el criterio histórico equivale aproximadamente a una reducción de {abs(modelo['umbral_cambio_pct']):.2f} %. "
        "Es un criterio analítico del proyecto, no una definición oficial de SEP ni de otra institución. La probabilidad mostrada es una estimación del modelo.")
    st.caption("Señal de apoyo para priorizar revisiones. No recomienda asignar, eliminar ni modificar plazas docentes.")
    consulta, nueva = st.tabs(["Consultar municipio", "Nueva evaluación"])
    with consulta:
        consultar_municipio(datos, modelo, historico)
    with nueva:
        nueva_evaluacion(modelo, datos, historico)
    st.divider()
    st.subheader("¿Cómo interpretar la probabilidad estimada?")
    st.write("La probabilidad estimada indica qué tan compatible es el municipio con el grupo de casos que históricamente presentaron una disminución relevante de matrícula.")
    st.write("No representa el porcentaje exacto en que disminuirá la matrícula.")
    st.write("Por ejemplo, una probabilidad de 10.3 % no significa que la matrícula disminuirá 10.3 %. Significa que, según el modelo, el municipio presenta una señal de riesgo de 10.3 % de pertenecer al grupo de disminución relevante.")
    a, b = st.columns(2)
    with a.container(border=True, key="tarjeta_priorizacion"):
        st.caption("UMBRAL DE PRIORIZACIÓN")
        st.markdown(f"### {modelo['umbral_probabilidad']:.0%}")
        st.caption("Valor mínimo de probabilidad utilizado para generar una alerta.")
    with b.container(border=True, key="tarjeta_definicion"):
        st.caption("DISMINUCIÓN RELEVANTE")
        st.markdown(f"### ≈ {modelo['umbral_cambio_pct']:.2f} %")
        st.caption("Criterio histórico utilizado para definir qué casos se consideraron disminuciones relevantes durante el entrenamiento.")
    st.caption(f"{modelo['umbral_probabilidad']:.0%} y {modelo['umbral_cambio_pct']:.2f} % representan conceptos distintos.")
    st.write(f"Para este proyecto se estableció un umbral de priorización de {modelo['umbral_probabilidad']:.0%}. "
        f"Si la probabilidad es igual o superior a {modelo['umbral_probabilidad']:.0%}, el municipio se marca como ‘Riesgo identificado’. "
        f"Si es menor a {modelo['umbral_probabilidad']:.0%}, aparece como ‘Sin alerta’.")
    st.caption("Esta probabilidad es generada por un modelo Random Forest a partir de características educativas, demográficas y de contexto municipal. La clasificación se basa en patrones observados en datos históricos y no implica una relación causal.")
    with st.expander("Acerca del modelo"):
        st.write(f"**Modelo:** Random Forest · **Variables:** {len(modelo['variables'])}")
        st.write("**Entrenamiento:** tres pares históricos de ciclos escolares.")
        for periodo in modelo.get("periodos_entrenamiento", []):
            st.caption(periodo.replace("->", "→"))
        st.write("**Prueba histórica:** " + modelo.get("periodo_prueba_historica", "2023-2024 -> 2024-2025").replace("->", "→"))
        st.write("**Recall histórico:** 75.2 % · **Balanced accuracy:** 62.8 % · **ROC-AUC:** 0.668")
        st.write(f"**Umbral de priorización:** {modelo['umbral_probabilidad']:.0%} · "
            f"**Definición analítica de disminución relevante:** cambio porcentual ≤ {modelo['umbral_cambio_pct']:.2f} %.")
        st.caption("Estas métricas corresponden a una prueba histórica y no garantizan el mismo desempeño en datos futuros.")
    with st.expander("Limitaciones"):
        st.markdown("""- Es un modelo experimental y puede generar falsas alertas o dejar municipios sin detectar.
- Utiliza contexto INEGI 2020 como información estructural.
- Las relaciones pueden cambiar con el tiempo; debe reevaluarse al incorporar nuevos ciclos.
- No realiza inferencia causal ni sustituye decisiones educativas o administrativas.
- Las probabilidades estimadas no garantizan que ocurra una disminución.""")
    st.caption("Proyecto académico de Ingeniería de Datos II. Fuentes principales: SEP, CONAPO e INEGI.")


if __name__ == "__main__":
    main()
