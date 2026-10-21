# Proyecto Integrador — Ingeniería de Datos II, Reto 3–4

El proyecto integra educación media superior de SEP, población municipal de CONAPO y contexto del Censo 2020 de INEGI para estudiar su relación a escala municipal. El dataset analítico para predecir matrícula del siguiente ciclo ya está preparado; Machine Learning sigue pendiente.

## Organización

- `data/raw/`: Bronze, fuentes originales y material de referencia.
- `data/processed/`: Silver, tablas limpias y homologadas por fuente.
- `data/curated/`: Gold, base municipal integrada para análisis.
- `database/sql/`: SQLite, esquema y consultas existentes.
- `notebooks/`: revisión, ETL, SQL, integración y análisis.
- `outputs/tables/` y `outputs/figures/`: controles, resúmenes y figuras.
- `docs/`: inventario, arquitectura, reproducibilidad y documentación previa.

Bronze, Silver y Gold son clasificaciones conceptuales; se mantienen las rutas existentes.

## Entorno y ejecución

Se utiliza Python con pandas, NumPy, openpyxl, Matplotlib, SciPy e IPython; SQLite forma parte de la biblioteca estándar. Instalar las dependencias con `python -m pip install -r requirements.txt` en un entorno aislado. El lector de notebooks y su kernel deben estar disponibles en el editor elegido; no se ha comprobado aquí un kernel Jupyter. Las versiones registradas corresponden al entorno consultado, no a una reconstrucción del entorno histórico.

Abrir el proyecto completo y ejecutar desde su raíz o desde `notebooks/`, según la detección de raíz que ya incluye el código. Usar rutas relativas y conservar CVEGEO como texto para mantener ceros iniciales. El notebook 12 puede anteponer `.exploratorio_deps` si existe: revisar esa carpeta al configurar el entorno, pues tiene prioridad sobre paquetes instalados.

## Orden actual de notebooks

1. `01_revision_fuente_sep.ipynb`
2. `02_revision_fuente_conapo.ipynb`
3. `03_revision_fuente_inegi.ipynb`
4. `04_etl_sep.ipynb`
5. `05_etl_conapo.ipynb`
6. `06_etl_inegi.ipynb`
7. `07_revision_compatibilidad_territorial.ipynb`
8. `08_creacion_base_sqlite.ipynb`
9. `09_carga_base_sqlite.ipynb`
10. `10_consultas_sql.ipynb`
11. `11_integracion_fuentes.ipynb`
12. `12_analisis_exploratorio.ipynb`

La fase de formalización es documental: no requiere volver a ejecutar estos notebooks. Para reconstruir productos en otra copia, seguir el orden anterior y consultar [reproducibilidad](docs/reproducibilidad.md).

## Política de notebooks

La formalización del Data Lake no creó notebooks. La continuación queda limitada a estos tres notebooks; 13, 14 y 15 están ejecutados; el 15 es el último notebook técnico nuevo del Reto 3–4. Los notebooks 01–13 se conservaron durante la fase PySpark:

- `13_mongodb_y_datawarehouse.ipynb`: conexión/carga y lectura MongoDB, función arquitectónica, modelo dimensional, carga, Data Warehouse y consultas analíticas.
- `14_pyspark_integracion.ipynb`: lectura Silver, transformaciones, integración, variables, escritura Gold y comparación con pandas.
- `15_validacion_y_dataset_ia.ipynb`: validaciones finales, conteos, faltantes, granularidad, consistencia entre capas, dataset IA y reproducibilidad.

Metadatos, tablas, SQL, diagramas y documentación se guardan en los formatos que correspondan, sin un notebook por tarea. Antes de un cuarto notebook adicional se debe justificar por qué no cabe con claridad en los tres previstos, qué problema resuelve separarlo y qué ventaja aporta a la reproducibilidad o comprensión; no crearlo automáticamente.

Consultar [inventario](docs/inventario_datos.csv), [arquitectura](docs/arquitectura_datalake.md) y [reproducibilidad](docs/reproducibilidad.md).

## MongoDB y Data Warehouse

Después de 01–12 sigue `notebooks/13_mongodb_y_datawarehouse.ipynb`. Ya se comprobaron sus entradas SQL (13,508), CSV CONAPO (14,850) e INEGI Silver (2,469). Están preparados `database/warehouse/schema_dw.sql` y `database/warehouse/consultas_dw.sql`. La extracción real MongoDB y la carga del DW se completaron con salidas guardadas en el notebook.

Instalar PyMongo en el entorno del kernel antes de la conexión y configurar `MONGODB_URI` fuera del notebook, o utilizar la entrada oculta durante su ejecución. No pegar la conexión en celdas. El notebook utiliza la base del proyecto `reto34_educacion` y únicamente su colección `contexto_municipal_2020`.

Ya se incluyen `database/mongodb/contexto_municipal_2020.json`, `database/warehouse/dw_educacion.db` y `outputs/tables/control_datawarehouse.csv`. El DW contiene 2,266 municipios, 6 ciclos y 13,508 hechos. La integridad y la conciliación por ciclo aprobaron. El modo JSON permite reproducir la extracción guardada y deja visible ese origen; la ejecución de entrega utilizó MongoDB. Se registró `pymongo==4.19.0` en requirements.

## Integración PySpark

Después del 13 sigue `notebooks/14_pyspark_integracion.ipynb`, ejecutado con PySpark 4.2.0 y Java Temurin 25.0.4.1 en `local[*]`. Lee SEP y CONAPO Silver y el JSON ya extraído de MongoDB, sin conexiones ni credenciales. Se revisó el ejemplo disponible en `clase/ejemplo_ventas_pyspark_validado.ipynb`.

Productos: `data/curated/base_integrada_pyspark.parquet/` (carpeta nativa con partes), `data/curated/base_integrada_pyspark.csv` y `outputs/tables/control_pyspark.csv`. Conserva 13,508 filas, 2,266 municipios y 6 ciclos; los 78 controles aprobaron. La matrícula coincide con pandas en todos los ciclos; las diferencias decimales máximas son 7.1e-15.

En Windows la escritura requiere las utilidades nativas Hadoop 3.5.0 locales en `.exploratorio_deps/hadoop-3.5.0/bin/`; el notebook configura solo su proceso. La procedencia y huellas están en reproducibilidad. No se modificaron Java, JAVA_HOME ni variables persistentes. La copia pandas para CSV ocupó aproximadamente 2.99 MiB; en un volumen grande se conservaría la escritura desde Spark. La validación final y el dataset IA se completaron en el notebook 15; el modelado sigue pendiente.

## Validación final y dataset IA

El último notebook técnico es `notebooks/15_validacion_y_dataset_ia.ipynb`. Produce `data/curated/dataset_ia_matricula.csv`, `docs/diccionario_dataset_ia.csv` y `outputs/tables/control_dataset_ia.csv`, sin modificar los notebooks 01–14 ni los productos anteriores.

El dataset contiene 8,996 pares municipio–ciclo, 2,255 municipios, cuatro transiciones y 20 columnas. Usa variables de t para la matrícula observada en t+1; excluye la transición 2019–2020 → 2020–2021. Conserva cuatro filas sin contexto INEGI y excluye tres observaciones sin registro en el ciclo siguiente, sin asignar objetivos cero. Los 80 controles aprobaron.

Las diez variables predictoras son candidatas, todavía sin selección definitiva. Se excluyen nuevo_ingreso por definición pendiente y registros_origen por describir el proceso. Entrenamiento propuesto: primeras tres transiciones (2020–2021 → 2021–2022 hasta 2022–2023 → 2023–2024); prueba propuesta: 2023–2024 → 2024–2025. No se generan particiones ni modelos. El baseline futuro será matrícula t. Antes de modelar se revisarán disponibilidad efectiva de fuentes, faltantes y comparabilidad territorial.

## Aplicación Streamlit

La aplicación usa el modelo experimental ya guardado; no lo entrena ni modifica sus umbrales. Permite consultar municipios y evaluar un registro nuevo como apoyo para priorizar revisiones, sin tomar decisiones sobre plazas docentes.

Con Python 3.12 o un entorno compatible con scikit-learn 1.9.1, instala las dependencias y ejecuta desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

El requirements.txt de la raíz contiene las dependencias de la aplicación; los entornos de los notebooks siguen documentados en sus secciones anteriores.

Para desplegar posteriormente se necesitan estos cuatro archivos, conservando sus carpetas:

- app.py
- requirements.txt
- models/modelo_disminucion_relevante.joblib
- data/curated/base_inferencia_disminucion_relevante.csv

El README y .gitignore pueden acompañarlos en GitHub. El modelo es experimental y sus probabilidades son estimaciones; las métricas históricas no garantizan el desempeño futuro. Esta etapa no publica ni despliega la aplicación.
