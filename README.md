# Perfiles financieros de las empresas del S&P 500
### Análisis de Componentes Principales (ACP) + K-Means · Streamlit

**Taller individual — Machine Learning aplicado a Economía y Finanzas**
Pontificia Universidad Javeriana Cali · Hector William Hernández Abadía · 2026

**Aplicación desplegada:** `PEGA_AQUÍ_EL_ENLACE_DE_STREAMLIT`
**Video de sustentación:** `PEGA_AQUÍ_EL_ENLACE_DEL_VIDEO`

---

## 1. Contexto y pregunta

El S&P 500 suele tratarse como un solo bloque, pero agrupa empresas con perfiles muy distintos:
tecnológicas que cotizan a múltiplos altos, *utilities* que reparten dividendos estables, petroleras,
bancos y compañías que vienen de caídas fuertes. La clasificación por **sector** no siempre captura esas
diferencias.

**Preguntas:**
1. ¿Qué dimensiones resumen la mayor parte de las diferencias de valoración, rentabilidad, tamaño y
   comportamiento del precio entre las empresas del índice?
2. ¿Existen grupos de empresas con perfiles financieros parecidos?
3. ¿Esos grupos coinciden con los sectores o los atraviesan?

**¿Por qué ACP y K-Means?** Los múltiplos financieros están correlacionados (una acción "barata" lo suele
ser en utilidades, ventas y valor en libros a la vez). El ACP condensa esa redundancia en pocas
dimensiones interpretables antes de medir distancias. K-Means busca grupos sin etiquetas previas, lo que
permite contrastar una segmentación construida desde los datos con la clasificación sectorial que el
mercado ya usa: útil para diversificar portafolios y para elegir empresas comparables en valoración.

## 2. Datos

| | |
|---|---|
| **Fuente** | Repositorio público [datasets/s-and-p-500-companies-financials](https://github.com/datasets/s-and-p-500-companies-financials) (Open Knowledge Foundation, licencia ODC-PDDL). Indicadores de **Yahoo Finance** (vía `yfinance`) y sectores GICS de **Wikipedia** ([datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)). |
| **Archivos** | `data/constituents-financials.csv` (indicadores) y `data/constituents.csv` (sector GICS). Copia descargada el 17-nov-2026 para que el análisis sea reproducible. |
| **Cobertura** | 503 registros (tickers) del S&P 500; corte transversal de un momento del mercado. |
| **Unidad de observación** | Una empresa. |

**Variables construidas** (los múltiplos como P/E o P/B no están definidos o explotan cuando hay pérdidas o
patrimonio negativo, por eso se usan sus inversos):

| Variable | Fórmula | Lectura |
|---|---|---|
| Tamaño | log10(Market Cap) | Tamaño en bolsa |
| Rendimiento de utilidades (E/P) | EPS ÷ Price | Inverso del P/E; negativo = pérdidas |
| Rendimiento por dividendo | Dividend Yield | Reparto de caja |
| Ventas / Capitalización | 1 ÷ (P/S) | Ventas por dólar de capitalización |
| Libro / Mercado (B/M) | 1 ÷ (P/B) | Perfil *value* |
| EBITDA / Capitalización | EBITDA ÷ Market Cap | Caja operativa por dólar de capitalización |
| Posición en rango 52 semanas | (P − Mín) ÷ (Máx − Mín) | 0 = en mínimo anual; 1 = en máximo |
| Amplitud del rango 52 semanas | (Máx − Mín) ÷ P | Proxy de volatilidad |

Ticker, nombre, sector y enlace SEC **no entran al modelo** (identificadores/categorías); el sector se
usa después para interpretar los grupos.

**Limitaciones de la fuente:** es una foto de un solo momento; Yahoo Finance no reporta EBITDA para
bancos; hay errores puntuales de captura (p. ej. Paramount tras su fusión).

## 3. Metodología

1. **Calidad de datos** (sección 3 de la app, con bitácora):
   - 0 duplicados por ticker; 3 empresas con dos clases de acción (GOOG, FOX, NWS) → se deja una.
   - 17 registros sin precio ni capitalización → eliminados.
   - 1 registro inconsistente (PARA: capitalización de USD 0,2 M y máximo anual ~4.000 veces el precio) → eliminado.
   - 26 empresas sin EBITDA, **todas financieras** (bancos, brókers, gestoras) → excluidas, porque el
     EBITDA no aplica a su modelo de negocio (si se quita esa variable en la app, vuelven a entrar).
   - Winsorización al 1 % y 99 % (los extremos se recortan, no se eliminan).
   - **Muestra final: 456 empresas, 8 variables.**
2. **Estandarización** (media 0, desviación 1) con `StandardScaler`.
3. **ACP** con `sklearn.decomposition.PCA`: varianza explicada, acumulada, sedimentación, cargas y biplot.
   Número de componentes por criterio de Kaiser (configurable: 80 % de varianza o manual).
4. **K-Means** (`n_init=20`, `random_state=42`) sobre los componentes conservados, comparando K = 2…10
   con método del codo y coeficiente de silueta.
5. **Interpretación** de los grupos con las variables originales (medianas) y cruce con sectores GICS.

## 4. Resultados principales (configuración por defecto)

**ACP:** 3 componentes (valores propios 2,58 / 1,56 / 1,07) que conservan el **65 %** de la varianza.

| Componente | Varianza | Interpretación |
|---|---|---|
| CP1 | 32,2 % | **Barata vs. cara**: B/M, EBITDA/Cap, S/P y dividendo (+); tamaño (−) |
| CP2 | 19,5 % | **Corrección reciente del precio**: amplitud 52s (+), posición 52s (−) |
| CP3 | 13,3 % | **Rentabilidad contable**: E/P (+0,91) |

**K-Means:** la silueta es casi plana entre K = 2 y 5 (0,33–0,35); se eligió **K = 4** (silueta 0,348,
empatada con el máximo) por ser la solución con lectura financiera más clara.

| Grupo | n | Perfil | Ejemplos |
|---|---|---|---|
| A — Calidad con impulso | 227 | P/E ≈ 28, dividendo ≈ 1,1 %, cerca del máximo anual, baja oscilación | Apple, Microsoft, Amazon, Meta, Eli Lilly |
| B — Valor y dividendos | 134 | P/E ≈ 17, dividendo ≈ 2,8 %, P/B ≈ 1,9 | Verizon, AT&T, ConocoPhillips, Pfizer |
| C — Volátiles en corrección | 84 | Rango anual ≈ 84 % del precio, en el 22 % inferior de su rango | Nvidia, Alphabet, Broadcom, Micron |
| D — En dificultades | 11 | 91 % con pérdidas, dividendo ≈ 5,6 % por caída del precio | Ford, Kraft Heinz, Dow, General Mills |

*Las letras solo identifican los grupos (ordenados por tamaño); no implican orden.*

**Grupos vs. sectores:** *utilities* (71 %) y energía (68 %) caen sobre todo en el grupo B; tecnología es el
único sector donde predomina el grupo C (51 %). Ningún sector cae completo en un solo grupo: la
segmentación aporta información que el sector no da.

**Limitaciones:** corte de un solo momento (el CP2 cambia rápido); banca excluida; estructura de grupos
moderada (silueta ≈ 0,35, ~30 empresas con silueta negativa); errores puntuales de la fuente.

## 5. Estructura del repositorio

```
├── app.py                  # Aplicación Streamlit (6 secciones)
├── src/
│   └── procesamiento.py    # Carga, limpieza, variables, ACP y K-Means
├── data/
│   ├── constituents-financials.csv
│   └── constituents.csv
├── .streamlit/config.toml  # Tema visual
├── requirements.txt
└── README.md
```

Todas las tablas, métricas y gráficos se **calculan en vivo** dentro de la app a partir de los CSV; no hay
resultados precargados. Los parámetros (variables, winsorización, criterio de componentes, K, semilla) se
cambian en la barra lateral.

## 6. Cómo ejecutar

```bash
git clone https://github.com/TU_USUARIO/acp-kmeans-sp500.git
cd acp-kmeans-sp500
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Actualizar los datos (opcional):** descarga de nuevo los dos CSV desde los repositorios de la fuente y
reemplázalos en `data/`. Los resultados cambiarán porque los precios cambian.
