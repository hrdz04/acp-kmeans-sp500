"""
Funciones de carga, preparación, ACP y K-Means.

Todo el procesamiento de la aplicación ocurre aquí: la app (app.py) solo
llama estas funciones con los parámetros que el usuario elige en pantalla.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Diccionario de variables construidas
# ---------------------------------------------------------------------------
VARIABLES = {
    "tamano_log": {
        "nombre": "Tamaño (log10 capitalización)",
        "formula": "log10(Market Cap)",
        "lectura": "Qué tan grande es la empresa en bolsa. Se usa logaritmo porque "
                   "la capitalización va de unos pocos miles de millones a más de 5 billones de USD.",
    },
    "rend_utilidades": {
        "nombre": "Rendimiento de utilidades (E/P)",
        "formula": "Earnings/Share ÷ Price",
        "lectura": "Utilidad por cada dólar pagado por la acción (inverso del P/E). "
                   "Alto = acción 'barata' frente a sus utilidades; negativo = empresa con pérdidas.",
    },
    "rend_dividendo": {
        "nombre": "Rendimiento por dividendo",
        "formula": "Dividend Yield",
        "lectura": "Dividendo anual ÷ precio. Alto = empresa madura que reparte caja.",
    },
    "ventas_precio": {
        "nombre": "Ventas / Capitalización (S/P)",
        "formula": "1 ÷ (Price/Sales)",
        "lectura": "Ventas por cada dólar de capitalización. Alto = el mercado paga poco por cada dólar vendido.",
    },
    "libro_mercado": {
        "nombre": "Valor en libros / Mercado (B/M)",
        "formula": "1 ÷ (Price/Book)",
        "lectura": "Patrimonio contable frente al valor de mercado. Alto = perfil 'value'; "
                   "negativo = patrimonio contable negativo (p. ej. por recompras de acciones).",
    },
    "ebitda_cap": {
        "nombre": "EBITDA / Capitalización",
        "formula": "EBITDA ÷ Market Cap",
        "lectura": "Generación operativa de caja por cada dólar de capitalización.",
    },
    "posicion_52s": {
        "nombre": "Posición en rango de 52 semanas",
        "formula": "(Price − Low52) ÷ (High52 − Low52)",
        "lectura": "0 = la acción está en su mínimo del año; 1 = en su máximo. Aproxima el momentum reciente.",
    },
    "amplitud_52s": {
        "nombre": "Amplitud del rango de 52 semanas",
        "formula": "(High52 − Low52) ÷ Price",
        "lectura": "Qué tanto osciló el precio en el último año frente al precio actual. Aproxima volatilidad.",
    },
}

VARS_DEFECTO = list(VARIABLES.keys())


def nombre(v: str) -> str:
    return VARIABLES.get(v, {}).get("nombre", v)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    """Une los indicadores financieros con el sector GICS de cada empresa."""
    fin = pd.read_csv(DATA_DIR / "constituents-financials.csv")
    cons = pd.read_csv(DATA_DIR / "constituents.csv")

    df = fin.merge(cons[["Symbol", "GICS Sector"]], on="Symbol", how="left")
    # Algunas empresas ya no aparecen en el listado actual: se completa el sector
    # usando la sub-industria (columna 'Sector' del archivo financiero).
    sub_a_sector = (cons.drop_duplicates("GICS Sub-Industry")
                        .set_index("GICS Sub-Industry")["GICS Sector"])
    df["GICS Sector"] = df["GICS Sector"].fillna(df["Sector"].map(sub_a_sector))
    manual = {"Home Furnishings": "Consumer Discretionary", "Brewers": "Consumer Staples",
              "Drug Retail": "Consumer Staples"}
    df["GICS Sector"] = df["GICS Sector"].fillna(df["Sector"].map(manual)).fillna("Sin clasificar")
    df = df.rename(columns={"Sector": "Sub-industria", "GICS Sector": "Sector GICS"})
    return df


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los múltiplos de Yahoo Finance en variables comparables."""
    X = pd.DataFrame(index=df.index)
    X["tamano_log"] = np.log10(df["Market Cap"])
    X["rend_utilidades"] = df["Earnings/Share"] / df["Price"]
    X["rend_dividendo"] = df["Dividend Yield"]
    X["ventas_precio"] = 1 / df["Price/Sales"]
    X["libro_mercado"] = 1 / df["Price/Book"]
    X["ebitda_cap"] = df["EBITDA"] / df["Market Cap"]
    rango = df["52 Week High"] - df["52 Week Low"]
    X["posicion_52s"] = (df["Price"] - df["52 Week Low"]) / rango
    X["amplitud_52s"] = rango / df["Price"]
    return X.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame, variables: list[str], winsor: float,
             excluir_inconsistentes: bool = True):
    """
    Aplica las reglas de limpieza y devuelve:
      base      -> filas que entran al modelo (datos originales + sector)
      X         -> variables construidas (sin winsorizar)
      Xw        -> variables winsorizadas (las que se estandarizan)
      bitacora  -> tabla con cada decisión y cuántas filas afectó
    """
    bitacora = []
    n0 = len(df)
    bitacora.append(("Registros iniciales", n0, "Archivo constituents-financials.csv"))

    dup = df.duplicated("Symbol").sum()
    df = df.drop_duplicates("Symbol")
    bitacora.append(("Duplicados por ticker eliminados", int(dup), "Se conserva la primera aparición"))

    # Algunas empresas cotizan con dos clases de acción (p. ej. Alphabet A y C):
    # son la misma compañía y contarla dos veces le daría doble peso.
    empresa = df["Name"].str.replace(r"\s*\(Class [A-Z]\)", "", regex=True).str.strip()
    dup_emp = empresa.duplicated()
    bitacora.append(("Segunda clase de acción de una misma empresa", int(dup_emp.sum()),
                     "Se conserva una sola clase: " + ", ".join(df.loc[dup_emp, "Symbol"].tolist())))
    df = df[~dup_emp]

    sin_precio = df["Price"].isna() | df["Market Cap"].isna()
    df = df[~sin_precio]
    bitacora.append(("Sin precio ni capitalización", int(sin_precio.sum()),
                     "Yahoo Finance no reportó datos (empresas fusionadas o retiradas del índice)"))

    if excluir_inconsistentes:
        incons = (df["Market Cap"] < 1e8) | (df["52 Week High"] > 50 * df["52 Week Low"]) | \
                 (df["52 Week High"] <= df["52 Week Low"])
        bitacora.append(("Datos inconsistentes", int(incons.sum()),
                         "Capitalización < USD 100 M (imposible para el S&P 500) o máximo de 52 semanas "
                         "más de 50 veces el mínimo (error de unidades tras una fusión): "
                         + ", ".join(df.loc[incons, "Symbol"].tolist())))
        df = df[~incons]

    X = construir_variables(df)[variables]
    faltan = X.isna().any(axis=1)
    if faltan.any():
        detalle = df.loc[faltan, "Sub-industria"].value_counts().head(4)
        txt = "; ".join(f"{k} ({v})" for k, v in detalle.items())
        bitacora.append(("Faltantes en alguna variable elegida", int(faltan.sum()),
                         f"Principalmente: {txt}"))
    X = X[~faltan]
    base = df.loc[X.index].copy()

    if winsor > 0:
        lo, hi = X.quantile(winsor), X.quantile(1 - winsor)
        Xw = X.clip(lo, hi, axis=1)
        afectadas = int(((X < lo) | (X > hi)).any(axis=1).sum())
        bitacora.append((f"Winsorización {winsor:.1%} – {1 - winsor:.1%}", afectadas,
                         "Empresas con al menos un valor recortado al percentil límite (no se eliminan)"))
    else:
        Xw = X.copy()
    bitacora.append(("Observaciones finales para el modelo", len(X), f"{len(variables)} variables cuantitativas"))

    bit = pd.DataFrame(bitacora, columns=["Paso", "Empresas", "Detalle"])
    return base, X, Xw, bit


def estandarizar(Xw: pd.DataFrame):
    scaler = StandardScaler()
    Z = scaler.fit_transform(Xw)
    return pd.DataFrame(Z, index=Xw.index, columns=Xw.columns), scaler


# ---------------------------------------------------------------------------
# ACP
# ---------------------------------------------------------------------------
def acp_completo(Z: pd.DataFrame):
    pca = PCA().fit(Z)
    n = len(pca.explained_variance_)
    tabla = pd.DataFrame({
        "Componente": [f"CP{i + 1}" for i in range(n)],
        "Valor propio": pca.explained_variance_,
        "Varianza explicada": pca.explained_variance_ratio_,
        "Varianza acumulada": np.cumsum(pca.explained_variance_ratio_),
    })
    cargas = pd.DataFrame(pca.components_.T, index=Z.columns,
                          columns=[f"CP{i + 1}" for i in range(n)])
    # Orientación: que la variable de mayor carga absoluta en cada CP sea positiva
    for c in cargas.columns:
        if cargas[c].loc[cargas[c].abs().idxmax()] < 0:
            cargas[c] *= -1
    puntajes = pd.DataFrame(Z.values @ cargas.values, index=Z.index, columns=cargas.columns)
    return tabla, cargas, puntajes


def sugerir_componentes(tabla: pd.DataFrame, criterio: str, umbral: float = 0.80) -> int:
    if criterio == "Kaiser (valor propio > 1)":
        return max(2, int((tabla["Valor propio"] > 1).sum()))
    if criterio.startswith("Varianza acumulada"):
        return max(2, int((tabla["Varianza acumulada"] < umbral).sum() + 1))
    return 2


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def evaluar_k(S: np.ndarray, k_min: int = 2, k_max: int = 10, semilla: int = 42) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(S)
        filas.append((k, km.inertia_, silhouette_score(S, km.labels_)))
    return pd.DataFrame(filas, columns=["K", "Inercia (WCSS)", "Silueta promedio"])


def ajustar_kmeans(S: np.ndarray, k: int, semilla: int = 42):
    """Ajusta K-Means y renombra los grupos con letras ordenadas por tamaño.

    Las letras solo identifican; no implican orden ni jerarquía entre grupos.
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(S)
    tam = pd.Series(km.labels_).value_counts()
    letras = {lab: f"Grupo {chr(65 + i)}" for i, lab in enumerate(tam.index)}
    etiquetas = np.array([letras[l] for l in km.labels_])
    sil = silhouette_samples(S, km.labels_)
    return km, etiquetas, sil


def describir_grupo(z_medias: pd.Series, n: int = 2) -> str:
    """Texto corto con los rasgos que más distinguen a un grupo (en desviaciones estándar)."""
    orden = z_medias.sort_values()
    altos = [f"{nombre(v).split(' (')[0].lower()} alto" for v, val in orden[::-1].items() if val > 0.35][:n]
    bajos = [f"{nombre(v).split(' (')[0].lower()} bajo" for v, val in orden.items() if val < -0.35][:n]
    partes = altos + bajos
    return ", ".join(partes) if partes else "cercano al promedio en todas las variables"
