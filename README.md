# Riesgo de disminución relevante de matrícula EMS

Aplicación desarrollada como parte del Proyecto Integrador de **Ingeniería de Datos II**.

El objetivo es identificar municipios que podrían requerir una revisión más detallada de la evolución de su matrícula de Educación Media Superior (EMS), utilizando información educativa, demográfica y de contexto municipal.

La aplicación funciona como una **herramienta experimental de apoyo para priorizar revisiones**. No determina decisiones educativas ni administrativas y no debe utilizarse de manera automática para asignar, eliminar o modificar plazas docentes.

## Fuentes de información

El proyecto integra información pública de:

- **SEP**: matrícula, docentes y escuelas de Educación Media Superior.
- **CONAPO**: población municipal de 15 a 19 años.
- **INEGI, Censo 2020**: grado promedio de escolaridad, acceso a internet y disponibilidad de computadora en las viviendas.

La unidad de análisis es el municipio.

## Modelo de Machine Learning

Después de comparar distintas formulaciones del problema, el modelo final experimental se construyó para identificar municipios con riesgo de presentar una **disminución relevante de matrícula** en el siguiente ciclo escolar.

Se utilizó un:

**Random Forest Classifier**

con 11 variables relacionadas con:

- matrícula actual;
- cambio de matrícula respecto al ciclo anterior;
- cambio porcentual anterior;
- docentes;
- cambio de docentes;
- escuelas;
- cambio de escuelas;
- población de 15 a 19 años;
- grado promedio de escolaridad;
- porcentaje de viviendas con internet;
- porcentaje de viviendas con computadora.

Para este proyecto, una disminución relevante corresponde a un criterio histórico aproximado de **−5.14 %**. Este valor es una definición analítica del proyecto y no una clasificación oficial de SEP u otra institución.

El modelo utiliza un **umbral de priorización de 20 %**. Cuando la probabilidad estimada es igual o superior a este valor, el municipio se marca como **“Riesgo identificado”**.

## Resultados de la prueba histórica

En el periodo reservado para prueba, correspondiente a **2023–2024 → 2024–2025**, el modelo obtuvo:

- Recall: **75.2 %**
- Balanced accuracy: **62.8 %**
- ROC-AUC: **0.668**
- F2: **0.541**

El recall indica que el modelo logró identificar aproximadamente tres de cada cuatro municipios que realmente presentaron una disminución relevante en el periodo de prueba.

Estas métricas corresponden a una prueba histórica y no garantizan el mismo desempeño con información futura.

## ¿Cómo interpretar la probabilidad?

La probabilidad estimada indica qué tan compatible es un municipio con el grupo de casos que históricamente presentaron una disminución relevante de matrícula.

Por ejemplo:

**Una probabilidad de 10.3 % NO significa que la matrícula disminuirá 10.3 %.**

Significa que el modelo asigna al municipio una probabilidad estimada de 10.3 % de pertenecer al grupo de riesgo definido en el proyecto.

Como el umbral de priorización es 20 %:

- probabilidad **≥ 20 %** → Riesgo identificado;
- probabilidad **< 20 %** → Sin alerta.

## Aplicación Streamlit

La aplicación contiene dos formas de uso.

### Consultar municipio

Permite seleccionar una entidad y un municipio.

La aplicación muestra:

- matrícula actual;
- cambio respecto al ciclo anterior;
- docentes;
- escuelas;
- población de 15 a 19 años;
- evolución histórica reciente de la matrícula;
- probabilidad estimada por el modelo;
- señal de riesgo.

La gráfica histórica utiliza los seis ciclos disponibles entre **2019–2020 y 2024–2025**.

### Nueva evaluación

Permite evaluar información nueva con el mismo modelo.

El usuario puede:

- partir de los datos de un municipio existente;
- utilizar un ejemplo precargado de Ciudad de México;
- modificar los valores cargados;
- o capturar completamente los datos de forma manual.

La aplicación calcula automáticamente las variables de cambio entre el ciclo actual y el ciclo inmediatamente anterior y posteriormente genera una nueva estimación.

## Estructura utilizada para la aplicación

```text
prediccion_matricula_ems/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── modelo_disminucion_relevante.joblib
│
└── data/
    └── curated/
        ├── base_inferencia_disminucion_relevante.csv
        └── historico_matricula_municipal.csv
```

## Ejecución local

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python -m streamlit run app.py
```

El modelo fue guardado utilizando:

**scikit-learn 1.9.1**

por lo que esta versión se conserva en `requirements.txt`.

## Limitaciones

El modelo debe interpretarse como una herramienta experimental.

Entre sus principales limitaciones:

- puede generar falsas alertas;
- puede dejar sin detectar algunos municipios con disminución relevante;
- utiliza información estructural del Censo 2020;
- las relaciones entre las variables pueden cambiar con el tiempo;
- requiere reevaluación al incorporar nuevos ciclos escolares;
- no realiza inferencia causal;
- no sustituye el análisis de especialistas ni las decisiones de las autoridades educativas.

## Uso académico

Proyecto académico desarrollado para la materia **Ingeniería de Datos II**.

Fuentes principales: **SEP, CONAPO e INEGI**.
