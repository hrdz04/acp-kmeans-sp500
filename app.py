"""
Perfiles financieros de las empresas del S&P 500
Análisis de Componentes Principales (ACP) + K-Means
Machine Learning aplicado a Economía y Finanzas — PUJ Cali
Autor: Hector William Hernández Abadía
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import procesamiento as pr

st.set_page_config(page_title="S&P 500 · ACP y K-Means", page_icon=":material/analytics:", layout="wide")

PALETA = ["#1f4e79", "#e07b39", "#2a9d8f", "#c44e52", "#8172b2", "#937860", "#da8bc3", "#5b8e7d"]
st.markdown("""
<style>
h1, h2, h3 {color:#1f4e79;}
div[data-testid="stMetricValue"] {color:#1f4e79;}
.nota {background:#f3f6fa;border-left:4px solid #1f4e79;padding:0.8rem 1rem;border-radius:4px;margin:0.4rem 0 1rem 0;}
</style>""", unsafe_allow_html=True)


def nota(texto: str):
    st.markdown(f"<div class='nota'>{texto}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos y parámetros (barra lateral)
# ---------------------------------------------------------------------------
@st.cache_data
def datos():
    return pr.cargar_datos()


df_raw = datos()

st.sidebar.title("S&P 500 · ACP + K-Means")
pagina = st.sidebar.radio("Secciones", [
    "1. Contexto y datos",
    "2. Exploración",
    "3. Preparación",
    "4. Componentes principales",
    "5. Segmentación K-Means",
    "6. Resultados y conclusiones",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros del análisis")
st.sidebar.caption("Todo se recalcula al cambiar un parámetro.")
variables = st.sidebar.multiselect(
    "Variables del modelo (mínimo 4)", pr.VARS_DEFECTO, default=pr.VARS_DEFECTO,
    format_func=pr.nombre)
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()
winsor = st.sidebar.slider("Winsorización por cola (%)", 0.0, 5.0, 1.0, 0.5,
                           help="Recorta los valores extremos al percentil elegido en cada cola.") / 100
excluir = st.sidebar.checkbox("Excluir registros inconsistentes", value=True)
criterio = st.sidebar.selectbox("Criterio para el número de componentes",
                                ["Kaiser (valor propio > 1)", "Varianza acumulada ≥ 80%", "Manual"])
semilla = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)

base, X, Xw, bitacora = pr.preparar(df_raw, variables, winsor, excluir)
Z, _ = pr.estandarizar(Xw)
tabla_var, cargas, puntajes = pr.acp_completo(Z)

n_sug = pr.sugerir_componentes(tabla_var, criterio)
if criterio == "Manual":
    n_comp = st.sidebar.slider("Componentes a conservar", 2, len(variables), 3)
else:
    n_comp = n_sug
    st.sidebar.info(f"Componentes conservados: **{n_comp}** "
                    f"({tabla_var['Varianza acumulada'].iloc[n_comp - 1]:.1%} de la varianza)")

S = puntajes.iloc[:, :n_comp].values


@st.cache_data
def tabla_k(S_bytes, shape, semilla):
    S_ = np.frombuffer(S_bytes).reshape(shape)
    return pr.evaluar_k(S_, 2, 10, semilla)


eval_k = tabla_k(S.tobytes(), S.shape, int(semilla))
k_sug = int(eval_k.loc[eval_k["Silueta promedio"].idxmax(), "K"])
k = st.sidebar.slider("Número de grupos K", 2, 10, 4,
                      help=f"La silueta máxima se obtiene con K = {k_sug}. Ver sección 5.")
km, grupos, sil = pr.ajustar_kmeans(S, k, int(semilla))
base["Grupo"] = grupos
base["Silueta"] = sil
orden_grupos = sorted(base["Grupo"].unique())
color_grupo = {g: PALETA[i % len(PALETA)] for i, g in enumerate(orden_grupos)}

st.sidebar.markdown("---")
st.sidebar.caption("Hector William Hernández Abadía · PUJ Cali · 2026")

ETQ = {v: pr.nombre(v) for v in pr.VARIABLES}
pct = tabla_var["Varianza explicada"]


# ---------------------------------------------------------------------------
# 1. Contexto
# ---------------------------------------------------------------------------
if pagina.startswith("1"):
    st.title("¿Qué perfiles financieros conviven dentro del S&P 500?")
    st.markdown("""
El **S&P 500** suele tratarse como un bloque: "el mercado". Pero adentro hay empresas muy distintas:
gigantes tecnológicas que cotizan a múltiplos altos, *utilities* que reparten dividendos estables,
bancos, petroleras y compañías que vienen de una caída fuerte en su precio.

Para un inversionista o un analista, la clasificación tradicional por **sector** no siempre captura
esas diferencias: dos empresas del mismo sector pueden tener perfiles de valoración y de riesgo opuestos.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Preguntas que orientan el análisis")
        st.markdown("""
1. ¿Qué **dimensiones** resumen la mayor parte de las diferencias de valoración, rentabilidad,
   tamaño y comportamiento del precio entre las empresas del índice?
2. ¿Existen **grupos** de empresas con perfiles financieros parecidos?
3. ¿Esos grupos **coinciden con los sectores** o los atraviesan?
""")
        st.subheader("¿Por qué ACP y K-Means?")
        st.markdown("""
- Los múltiplos financieros están **correlacionados entre sí** (una acción "barata" lo suele ser en
  utilidades, ventas y valor en libros a la vez). El **ACP** condensa esas variables redundantes en
  pocas dimensiones interpretables (p. ej. "valor vs. crecimiento") y elimina la redundancia antes
  de medir distancias.
- **K-Means** busca grupos sin usar etiquetas previas. Esto permite contrastar una segmentación
  construida *desde los datos* con la clasificación sectorial que el mercado ya usa, que es justamente
  lo que interesa para armar portafolios diversificados o identificar pares comparables.
""")
    with c2:
        st.subheader("Fuente y cobertura")
        st.markdown("""
- **Fuente:** repositorio público *datasets/s-and-p-500-companies-financials* (Open Knowledge
  Foundation, licencia ODC-PDDL). Los indicadores provienen de **Yahoo Finance** (librería `yfinance`)
  y el listado de empresas y sectores GICS de **Wikipedia**.
- **Cobertura:** las ~500 empresas que componen el índice; es una **foto de un momento** (corte
  descargado el 17 de noviembre de 2026), no una serie de tiempo.
- **Unidad de observación:** una empresa (ticker).
""")
        st.subheader("Limitaciones de la fuente")
        st.markdown("""
- Es un corte transversal: los resultados describen un momento del mercado y pueden cambiar.
- Yahoo Finance no reporta algunos campos (p. ej. **EBITDA de los bancos**, que no aplica a su modelo de negocio).
- Hay errores puntuales de captura (p. ej. empresas fusionadas con datos de precio incoherentes).
- Los múltiplos dependen del precio del día, que incluye ruido de corto plazo.
""")

    st.subheader("Base de datos")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Empresas en el archivo", len(df_raw))
    m2.metric("Columnas originales", df_raw.shape[1])
    m3.metric("Sectores GICS", df_raw["Sector GICS"].nunique())
    m4.metric("Empresas en el modelo", len(base))
    st.dataframe(df_raw.drop(columns=["SEC Filings"]).head(15), width="stretch")

    st.subheader("Variables construidas para el modelo")
    st.markdown("Los múltiplos como P/E o P/B **no están definidos o se vuelven absurdos** cuando la "
                "empresa tiene pérdidas o patrimonio negativo (un P/E de 4.000 es un problema de "
                "denominador, no una empresa cara). Por eso se usan sus **inversos** (rendimientos), "
                "que se comportan de forma continua, más dos medidas del precio en el último año.")
    dic = pd.DataFrame([{"Variable": d["nombre"], "Fórmula": d["formula"], "Cómo leerla": d["lectura"]}
                        for d in pr.VARIABLES.values()])
    st.dataframe(dic, width="stretch", hide_index=True)
    nota("El <b>ticker</b>, el <b>nombre</b>, el <b>sector</b> y el enlace SEC no entran al modelo: son "
         "identificadores o categorías. El sector se usa <i>después</i>, para interpretar los grupos.")


# ---------------------------------------------------------------------------
# 2. Exploración
# ---------------------------------------------------------------------------
elif pagina.startswith("2"):
    st.title("Exploración de los datos")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Estructura y tipos")
        info = pd.DataFrame({"Tipo": df_raw.dtypes.astype(str),
                             "No nulos": df_raw.notna().sum(),
                             "Faltantes": df_raw.isna().sum()})
        st.dataframe(info, width="stretch")
        st.caption(f"Duplicados por ticker: {df_raw.duplicated('Symbol').sum()}")
    with c2:
        st.subheader("Valores faltantes por columna")
        falt = df_raw.isna().sum().loc[lambda s: s > 0].sort_values()
        fig = px.bar(falt, orientation="h", labels={"value": "Registros faltantes", "index": ""},
                     color_discrete_sequence=[PALETA[1]])
        fig.update_layout(showlegend=False, height=350, title="Faltantes en el archivo original")
        st.plotly_chart(fig, width="stretch")
        nota("17 empresas no tienen ningún dato de mercado (fusionadas o retiradas del índice). "
             "EBITDA falta además en 26 empresas, <b>todas financieras</b> (bancos, brókers, gestoras): "
             "para un banco el EBITDA no tiene sentido porque los intereses son su operación.")

    st.subheader("Estadísticas descriptivas de las variables originales")
    num = ["Price", "Price/Earnings", "Dividend Yield", "Earnings/Share", "52 Week Low",
           "52 Week High", "Market Cap", "EBITDA", "Price/Sales", "Price/Book"]
    st.dataframe(df_raw[num].describe().T.style.format("{:,.3f}"), width="stretch")
    nota("Observa los extremos: P/E máximo de más de 4.000, P/B entre −565 y 442, EPS de −80. "
         "Son señales de denominadores cercanos a cero y de errores de captura, y justifican "
         "transformar las variables y tratar valores extremos antes de modelar.")

    st.subheader("Distribución de las variables construidas (antes de winsorizar)")
    Xall = pr.construir_variables(df_raw.dropna(subset=["Price"]))
    var = st.selectbox("Variable", pr.VARS_DEFECTO, format_func=pr.nombre)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(Xall, x=var, nbins=60, color_discrete_sequence=[PALETA[0]],
                           labels={var: pr.nombre(var)}, title=f"Histograma — {pr.nombre(var)}")
        fig.update_layout(yaxis_title="Número de empresas")
        st.plotly_chart(fig, width="stretch")
    with c2:
        tmp = Xw.assign(Sector=base["Sector GICS"])
        fig = px.box(tmp, x="Sector", y=var if var in tmp else None, points=False,
                     color_discrete_sequence=[PALETA[2]],
                     title=f"{pr.nombre(var)} por sector (datos ya winsorizados)",
                     labels={var: pr.nombre(var), "Sector": ""}) if var in tmp else None
        if fig:
            fig.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Esta variable no está seleccionada para el modelo.")
    st.caption(pr.VARIABLES[var]["lectura"])

    st.subheader("Matriz de correlaciones")
    metodo = st.radio("Método", ["spearman", "pearson"], horizontal=True,
                      help="Spearman es robusta a valores extremos (usa rangos).")
    corr = X.corr(method=metodo)
    fig = px.imshow(corr.rename(index=ETQ, columns=ETQ), text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title=f"Correlación de {metodo.capitalize()} entre las variables del modelo")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")
    pares = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    top = "; ".join(f"{pr.nombre(a).split(' (')[0]} – {pr.nombre(b).split(' (')[0]} ({v:+.2f})"
                    for (a, b), v in pares.head(3).items())
    nota(f"Pares más correlacionados: {top}. Las medidas de 'valor' (B/M, EBITDA/Cap, S/P, dividendo) "
         "se mueven juntas y en contra del tamaño; la posición en el rango de 52 semanas va en contra de su "
         "amplitud. Esa redundancia es la que el ACP aprovecha.")

    st.subheader("Relación entre dos variables")
    c1, c2 = st.columns(2)
    vx = c1.selectbox("Eje X", variables, index=0, format_func=pr.nombre)
    vy = c2.selectbox("Eje Y", variables, index=min(4, len(variables) - 1), format_func=pr.nombre)
    fig = px.scatter(Xw.assign(Empresa=base["Name"], Sector=base["Sector GICS"]), x=vx, y=vy,
                     color="Sector", hover_name="Empresa", labels=ETQ, opacity=0.75,
                     title=f"{pr.nombre(vy)} vs. {pr.nombre(vx)} (winsorizadas)")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 3. Preparación
# ---------------------------------------------------------------------------
elif pagina.startswith("3"):
    st.title("Preparación de los datos")
    st.subheader("Bitácora de decisiones")
    st.dataframe(bitacora, width="stretch", hide_index=True)
    nota("""<b>Por qué cada decisión:</b><br>
• <b>Sin datos de mercado:</b> no hay forma de calcular ninguna variable; imputarlas inventaría empresas.<br>
• <b>Inconsistentes:</b> Paramount Global aparece con capitalización de USD 0,2 millones y un máximo de
52 semanas casi 4.000 veces su precio: es un error de la fuente tras su fusión, no un dato real.<br>
• <b>Dos clases de acción:</b> Alphabet, Fox y News Corp cotizan con dos tickers; se deja uno para no
contar dos veces a la misma empresa.<br>
• <b>Bancos sin EBITDA:</b> se excluyen en lugar de imputar, porque el EBITDA no describe su negocio y
cualquier valor inventado los ubicaría artificialmente en un grupo. Si se quita la variable
<i>EBITDA / Capitalización</i> en la barra lateral, los bancos vuelven a entrar (pruébalo).<br>
• <b>Winsorización:</b> los valores extremos no se eliminan; se recortan al percentil 1 y 99. K-Means y
el ACP trabajan con distancias y varianzas, así que una sola empresa a más de 20 desviaciones estándar dominaría el
primer componente.<br>
• <b>Estandarización:</b> las variables tienen escalas muy distintas (log de capitalización ≈ 10,
dividendo ≈ 0,02). Sin llevarlas a media 0 y desviación 1, la de mayor escala dominaría la distancia.""")

    st.subheader("Efecto de la winsorización")
    zx = (X - X.mean()) / X.std()
    zw = (Xw - Xw.mean()) / Xw.std()
    largo = pd.concat([zx.melt(var_name="v", value_name="z").assign(Datos="Original"),
                       zw.melt(var_name="v", value_name="z").assign(Datos="Winsorizado")])
    largo["Variable"] = largo["v"].map(ETQ)
    fig = px.box(largo, x="Variable", y="z", color="Datos", points="outliers",
                 color_discrete_sequence=[PALETA[3], PALETA[0]],
                 title="Distribución en desviaciones estándar, antes y después de winsorizar")
    fig.update_layout(yaxis_title="Valor estandarizado (z)", xaxis_title="", xaxis_tickangle=-25, height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("Antes de winsorizar hay valores a más de 20 desviaciones estándar de la media; después, "
               "las distribuciones quedan comparables.")

    st.subheader("Datos estandarizados que entran al ACP")
    resumen = Z.describe().T[["mean", "std", "min", "max"]].rename(
        columns={"mean": "Media", "std": "Desv. estándar", "min": "Mínimo", "max": "Máximo"})
    resumen.index = resumen.index.map(ETQ)
    st.dataframe(resumen.style.format("{:.2f}"), width="stretch")
    st.dataframe(Z.assign(Empresa=base["Name"]).set_index("Empresa").rename(columns=ETQ).head(10)
                 .style.format("{:.2f}"), width="stretch")


# ---------------------------------------------------------------------------
# 4. ACP
# ---------------------------------------------------------------------------
elif pagina.startswith("4"):
    st.title("Análisis de Componentes Principales")
    st.subheader("Varianza explicada")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(tabla_var.style.format({"Valor propio": "{:.3f}", "Varianza explicada": "{:.1%}",
                                             "Varianza acumulada": "{:.1%}"}),
                     width="stretch", hide_index=True)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=tabla_var["Componente"], y=tabla_var["Valor propio"], name="Valor propio",
                    marker_color=PALETA[0])
        fig.add_scatter(x=tabla_var["Componente"], y=tabla_var["Valor propio"], mode="lines+markers",
                        name="Sedimentación", line=dict(color=PALETA[1]))
        fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Criterio de Kaiser (λ = 1)")
        fig.update_layout(title="Gráfico de sedimentación", yaxis_title="Valor propio (varianza del componente)",
                          xaxis_title="Componente", height=360)
        st.plotly_chart(fig, width="stretch")
    fig = px.line(tabla_var, x="Componente", y="Varianza acumulada", markers=True,
                  title="Varianza acumulada", color_discrete_sequence=[PALETA[2]])
    fig.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="80 %")
    fig.update_layout(yaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, width="stretch")

    kaiser = int((tabla_var["Valor propio"] > 1).sum())
    n80 = int((tabla_var["Varianza acumulada"] < 0.8).sum() + 1)
    nota(f"<b>¿Cuántos componentes conservar?</b> Con {len(variables)} variables, el criterio de Kaiser "
         f"sugiere <b>{kaiser}</b> (los que explican más que una variable original) y el de 80 % de "
         f"varianza sugiere <b>{n80}</b>. El codo del gráfico de sedimentación aparece después del CP{kaiser}. "
         f"Se conservan <b>{n_comp}</b> componentes, que resumen el "
         f"<b>{tabla_var['Varianza acumulada'].iloc[n_comp - 1]:.1%}</b> de la información. "
         "Que no se alcance el 80 % con 2 o 3 componentes indica que las variables capturan aspectos "
         "genuinamente distintos de las empresas (valoración, precio reciente, tamaño), y no una sola idea.")

    st.subheader("Cargas: qué significa cada componente")
    cg = cargas.iloc[:, :max(n_comp, 2)].rename(index=ETQ)
    fig = px.imshow(cg, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                    title="Cargas de las variables en los componentes conservados")
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")
    for c in cargas.columns[:n_comp]:
        s = cargas[c].sort_values()
        pos = ", ".join(pr.nombre(v) for v, x in s[::-1].items() if x > 0.3)
        neg = ", ".join(pr.nombre(v) for v, x in s.items() if x < -0.3)
        st.markdown(f"- **{c}** ({pct[cargas.columns.get_loc(c)]:.1%}): valores altos ↔ "
                    f"{pos or '—'}" + (f"; valores bajos ↔ {neg}" if neg else ""))

    st.subheader("Empresas en el plano de los dos primeros componentes")
    c1, c2, c3 = st.columns(3)
    ejes = list(cargas.columns[:max(n_comp, 2)])
    ex = c1.selectbox("Eje X", ejes, index=0)
    ey = c2.selectbox("Eje Y", ejes, index=1)
    colorear = c3.selectbox("Color", ["Sector GICS", "Grupo K-Means"])
    escala = st.slider("Longitud de las flechas (biplot)", 1.0, 8.0, 4.0, 0.5)
    dfp = puntajes.assign(Empresa=base["Name"], Ticker=base["Symbol"],
                          **{"Sector GICS": base["Sector GICS"], "Grupo K-Means": base["Grupo"]})
    fig = px.scatter(dfp, x=ex, y=ey, color=colorear, hover_name="Empresa", hover_data=["Ticker"],
                     opacity=0.7,
                     color_discrete_map=color_grupo if colorear == "Grupo K-Means" else None,
                     category_orders={"Grupo K-Means": orden_grupos})
    for v in cargas.index:
        x1, y1 = cargas.loc[v, ex] * escala, cargas.loc[v, ey] * escala
        fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#333")
        fig.add_annotation(x=x1 * 1.12, y=y1 * 1.12, text=pr.nombre(v).split(" (")[0],
                           showarrow=False, font=dict(size=11, color="#333"))
    i, j = ejes.index(ex), ejes.index(ey)
    fig.update_layout(height=650, title=f"Biplot {ex} vs {ey} — resume el {pct[i] + pct[j]:.1%} de la varianza total",
                      xaxis_title=f"{ex} ({pct[i]:.1%})", yaxis_title=f"{ey} ({pct[j]:.1%})")
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada punto es una empresa. Las flechas muestran hacia dónde 'empuja' cada variable: "
               "empresas ubicadas en la dirección de una flecha tienen valores altos en esa variable.")
    if n_comp >= 3:
        with st.expander("Ver los tres primeros componentes en 3D"):
            fig = px.scatter_3d(dfp, x="CP1", y="CP2", z="CP3", color=colorear, hover_name="Empresa",
                                opacity=0.7, color_discrete_map=color_grupo if colorear == "Grupo K-Means" else None)
            fig.update_layout(height=650, title=f"CP1–CP3: {tabla_var['Varianza acumulada'].iloc[2]:.1%} de la varianza")
            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 5. K-Means
# ---------------------------------------------------------------------------
elif pagina.startswith("5"):
    st.title("Segmentación con K-Means")
    nota(f"K-Means se aplica sobre los <b>{n_comp} componentes conservados</b> (no sobre las 8 variables): "
         "así las distancias no se inflan por variables redundantes y se descarta la parte de la "
         "variación que es más ruido que estructura.")
    st.subheader("¿Cuántos grupos?")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(eval_k, x="K", y="Inercia (WCSS)", markers=True, color_discrete_sequence=[PALETA[0]],
                      title="Método del codo")
        fig.add_vline(x=k, line_dash="dash", line_color=PALETA[1], annotation_text=f"K elegido = {k}")
        fig.update_layout(yaxis_title="Suma de distancias² dentro de los grupos")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(eval_k, x="K", y="Silueta promedio", color_discrete_sequence=[PALETA[2]],
                     title="Coeficiente de silueta promedio", text_auto=".3f")
        fig.add_vline(x=k, line_dash="dash", line_color=PALETA[1])
        fig.update_layout(yaxis_title="Silueta (−1 a 1; más alto = grupos más separados)")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(eval_k.style.format({"Inercia (WCSS)": "{:,.1f}", "Silueta promedio": "{:.3f}"}),
                 width="stretch", hide_index=True)
    sil_k = eval_k.set_index("K")["Silueta promedio"]
    nota(f"El codo no es nítido: la inercia cae de forma gradual. La silueta máxima se da en <b>K = {k_sug}</b> "
         f"({sil_k.max():.3f}), pero varios valores de K quedan muy cerca (rango entre K=2 y K=5: "
         f"{sil_k.loc[2:5].min():.3f}–{sil_k.loc[2:5].max():.3f}). Cuando las métricas no deciden, manda la "
         "<b>interpretabilidad</b>: se elige el K cuyos grupos tienen un significado financiero claro y "
         "distinto entre sí (ver la discusión en la sección 6). Una silueta de 0,3–0,4 indica <b>estructura moderada</b>: hay perfiles reales, "
         "pero con fronteras difusas.")

    st.subheader(f"Grupos obtenidos con K = {k}")
    tam = base["Grupo"].value_counts().reindex(orden_grupos)
    cols = st.columns(len(orden_grupos))
    for c, g in zip(cols, orden_grupos):
        c.metric(g, f"{tam[g]} empresas", f"silueta {base.loc[base.Grupo == g, 'Silueta'].mean():.2f}",
                 delta_color="off")
    st.caption("Las letras solo identifican a los grupos (ordenados por tamaño); no indican orden ni jerarquía.")

    fig = px.scatter(puntajes.assign(Grupo=base["Grupo"], Empresa=base["Name"], Sector=base["Sector GICS"]),
                     x="CP1", y="CP2", color="Grupo", hover_name="Empresa", hover_data=["Sector"],
                     color_discrete_map=color_grupo, category_orders={"Grupo": orden_grupos}, opacity=0.75)
    cent = pd.DataFrame(km.cluster_centers_[:, :2], columns=["CP1", "CP2"])
    fig.add_scatter(x=cent["CP1"], y=cent["CP2"], mode="markers", name="Centroides",
                    marker=dict(symbol="x", size=16, color="black", line=dict(width=2)))
    fig.update_layout(height=580, title=f"Grupos en el plano CP1–CP2 ({pct[0] + pct[1]:.1%} de la varianza; "
                                        f"el modelo usa {n_comp} componentes)",
                      xaxis_title=f"CP1 ({pct[0]:.1%})", yaxis_title=f"CP2 ({pct[1]:.1%})")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Silueta por empresa")
    ss = base[["Name", "Grupo", "Silueta"]].sort_values(["Grupo", "Silueta"], ascending=[True, False]).reset_index(drop=True)
    fig = px.bar(ss, x=ss.index, y="Silueta", color="Grupo", color_discrete_map=color_grupo,
                 hover_name="Name", category_orders={"Grupo": orden_grupos})
    fig.add_hline(y=base["Silueta"].mean(), line_dash="dash", annotation_text="Promedio")
    fig.update_layout(bargap=0, height=360, xaxis_title="Empresas (ordenadas por grupo y silueta)",
                      yaxis_title="Silueta individual", title="Gráfico de silueta")
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(fig, width="stretch")
    mal = (base["Silueta"] < 0).sum()
    st.caption(f"{mal} empresas ({mal / len(base):.1%}) tienen silueta negativa: están más cerca, en promedio, "
               "de otro grupo que del propio. Son los casos frontera.")

    st.subheader("¿Cómo son los grupos? (variables originales)")
    Xfull = pr.construir_variables(base)
    perfil = pd.DataFrame({
        "Empresas": base.groupby("Grupo").size(),
        "Capitalización mediana (USD mil M)": base.groupby("Grupo")["Market Cap"].median() / 1e9,
        "P/E mediano": base.groupby("Grupo")["Price/Earnings"].median(),
        "% con pérdidas": base.assign(p=base["Earnings/Share"] < 0).groupby("Grupo")["p"].mean() * 100,
        "Dividend yield mediano (%)": base.groupby("Grupo")["Dividend Yield"].median() * 100,
        "P/B mediano": base.groupby("Grupo")["Price/Book"].median(),
        "P/S mediano": base.groupby("Grupo")["Price/Sales"].median(),
        "Posición 52s mediana": Xfull.groupby(base["Grupo"])["posicion_52s"].median(),
        "Amplitud 52s mediana": Xfull.groupby(base["Grupo"])["amplitud_52s"].median(),
    }).reindex(orden_grupos)
    total = pd.DataFrame({
        "Empresas": [len(base)], "Capitalización mediana (USD mil M)": [base["Market Cap"].median() / 1e9],
        "P/E mediano": [base["Price/Earnings"].median()], "% con pérdidas": [(base["Earnings/Share"] < 0).mean() * 100],
        "Dividend yield mediano (%)": [base["Dividend Yield"].median() * 100],
        "P/B mediano": [base["Price/Book"].median()], "P/S mediano": [base["Price/Sales"].median()],
        "Posición 52s mediana": [Xfull["posicion_52s"].median()], "Amplitud 52s mediana": [Xfull["amplitud_52s"].median()],
    }, index=["Todo el S&P 500"])
    tabla_perfil = pd.concat([perfil, total]).round(2)
    tabla_perfil["Empresas"] = tabla_perfil["Empresas"].astype(int)
    st.dataframe(tabla_perfil, width="stretch")
    st.caption("Se usan medianas porque son robustas a los valores extremos de los múltiplos. "
               "El P/E solo existe para empresas con utilidades positivas.")

    zmed = Z.groupby(base["Grupo"]).mean().reindex(orden_grupos)
    fig = px.imshow(zmed.rename(columns=ETQ), text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1.5, zmax=1.5, aspect="auto",
                    title="Perfil de cada grupo: promedio estandarizado (0 = promedio del índice)")
    fig.update_layout(height=320 + 30 * k)
    st.plotly_chart(fig, width="stretch")
    for g in orden_grupos:
        st.markdown(f"- **{g}** ({tam[g]} empresas): {pr.describir_grupo(zmed.loc[g])}.")

    st.subheader("Grupos vs. sectores")
    ct = pd.crosstab(base["Sector GICS"], base["Grupo"], normalize="index") * 100
    fig = px.imshow(ct.round(0), text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                    title="% de las empresas de cada sector que cae en cada grupo")
    fig.update_layout(height=520, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Ver conteos absolutos"):
        st.dataframe(pd.crosstab(base["Sector GICS"], base["Grupo"], margins=True, margins_name="Total"),
                     width="stretch")

    st.subheader("Buscar una empresa")
    tabla_emp = base[["Symbol", "Name", "Sector GICS", "Grupo", "Silueta", "Market Cap", "Price/Earnings",
                      "Dividend Yield"]].copy()
    tabla_emp["Market Cap"] = tabla_emp["Market Cap"] / 1e9
    tabla_emp = tabla_emp.rename(columns={"Symbol": "Ticker", "Name": "Empresa", "Market Cap": "Cap. (USD mil M)"})
    q = st.text_input("Ticker o nombre (p. ej. AAPL, Exxon, Coca)")
    if q:
        tabla_emp = tabla_emp[tabla_emp["Ticker"].str.contains(q, case=False) |
                              tabla_emp["Empresa"].str.contains(q, case=False)]
    st.dataframe(tabla_emp.sort_values("Cap. (USD mil M)", ascending=False)
                 .style.format({"Silueta": "{:.2f}", "Cap. (USD mil M)": "{:,.1f}", "Price/Earnings": "{:.1f}",
                                "Dividend Yield": "{:.2%}"}),
                 width="stretch", hide_index=True, height=380)
    st.download_button("Descargar asignación de grupos (CSV)",
                       base[["Symbol", "Name", "Sector GICS", "Grupo", "Silueta"]].to_csv(index=False).encode("utf-8"),
                       "grupos_sp500.csv", "text/csv")


# ---------------------------------------------------------------------------
# 6. Conclusiones
# ---------------------------------------------------------------------------
else:
    st.title("Resultados y conclusiones")
    defecto = (variables == pr.VARS_DEFECTO and abs(winsor - 0.01) < 1e-9 and excluir
               and criterio == "Kaiser (valor propio > 1)" and k == 4 and int(semilla) == 42)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas analizadas", len(base))
    c2.metric("Componentes conservados", n_comp)
    c3.metric("Varianza conservada", f"{tabla_var['Varianza acumulada'].iloc[n_comp - 1]:.1%}")
    c4.metric("Silueta promedio", f"{base['Silueta'].mean():.3f}")

    if not defecto:
        st.warning("Cambiaste algún parámetro: las cifras de arriba y de la sección 5 están actualizadas, "
                   "pero la lectura escrita de abajo corresponde a la **configuración por defecto** "
                   "(8 variables, winsorización 1 %, Kaiser, K = 4, semilla 42).")

    st.subheader("1. ¿Qué dimensiones resumen las diferencias entre empresas?")
    st.markdown("""
- **CP1 (32 %) — "Barata vs. cara" (valor frente a crecimiento).** Cargas positivas de EBITDA/capitalización,
  libro/mercado, ventas/capitalización y dividendo; carga negativa del tamaño. Un CP1 alto describe a una
  empresa por la que el mercado paga poco frente a lo que vende, tiene y genera; un CP1 bajo, a una empresa
  grande por la que el mercado paga múltiplos altos.
- **CP2 (20 %) — "Corrección reciente del precio".** Lo definen la amplitud del rango de 52 semanas (+0,72)
  y la posición dentro de ese rango (−0,58): separa a las acciones que oscilaron mucho y hoy están lejos de
  su máximo de las que suben con estabilidad y cotizan cerca de él.
- **CP3 (13 %) — "Rentabilidad contable".** Dominado por el rendimiento de utilidades E/P (+0,91): distingue
  a las empresas con utilidades altas frente a su precio de las que reportan pérdidas.
- Los tres componentes conservan el **65 %** de la variación (criterio de Kaiser; el CP4 tiene valor propio
  0,90 y se descarta). No es una reducción espectacular, y eso también es un hallazgo: valoración,
  comportamiento del precio y rentabilidad son **dimensiones bastante independientes** que no caben en un
  solo indicador.
""")

    st.subheader("2. ¿Qué grupos aparecen?")
    st.markdown("""
| Grupo | Empresas | Lectura | Rasgos (medianas) | Ejemplos |
|---|---|---|---|---|
| **A — Calidad con impulso** | 227 | Grandes, caras y estables | P/E ≈ 28, dividendo ≈ 1,1 %, cotizan al 80 % de su rango anual, baja oscilación | Apple, Microsoft, Amazon, Meta, Eli Lilly, Walmart |
| **B — Valor y dividendos** | 134 | Baratas y defensivas | P/E ≈ 17, dividendo ≈ 2,8 %, P/B ≈ 1,9, P/S ≈ 1,3 | Verizon, AT&T, ConocoPhillips, Pfizer, CVS |
| **C — Volátiles en corrección** | 84 | Mucha oscilación, lejos de su máximo | Rango anual ≈ 84 % del precio, cotizan al 22 % de su rango, casi sin dividendo | Nvidia, Alphabet, Broadcom, Micron, Oracle, Intel |
| **D — En dificultades** | 11 | Pérdidas y precio deprimido | 91 % reporta pérdidas, dividendo ≈ 5,6 % (alto porque el precio cayó), P/B ≈ 1,1 | Ford, Kraft Heinz, Dow, General Mills, International Paper |
""")
    st.caption("Los nombres son interpretaciones de los perfiles de la sección 5; las letras solo identifican "
               "(ordenadas por tamaño) y no indican orden ni calidad.")

    st.subheader("3. ¿Coinciden con los sectores?")
    st.markdown("""
Solo en parte. *Utilities* (71 %) y energía (68 %) caen mayoritariamente en el grupo de valor y dividendos,
y tecnología es el único sector donde predomina el grupo volátil (51 %, sobre todo semiconductores). Pero
**ningún sector cae completo en un solo grupo**: el 45 % de las tecnológicas está en el grupo de calidad
con impulso, y salud, consumo básico, inmobiliario e industriales se reparten entre tres o más grupos. La
segmentación aporta información que el sector no da: dos empresas del mismo sector pueden estar en
situaciones de mercado opuestas.
""")

    st.subheader("4. ¿Son grupos claramente distinguibles?")
    st.markdown("""
No del todo. La silueta promedio (≈ 0,35) indica **estructura moderada**: los centros de los grupos son
distintos, pero hay una franja de empresas frontera (unas 30 tienen silueta negativa). El grupo A es el más
compacto (silueta ≈ 0,48); B y C se tocan entre sí (≈ 0,21–0,22).

**Elección de K.** El codo no es nítido y la silueta es casi plana entre K = 2 y K = 5 (0,33–0,35). Se
eligió **K = 4** porque empata con el máximo de silueta y cada grupo tiene una lectura financiera propia:
con K = 3 las 11 empresas en dificultades se diluyen en otros grupos; con K = 5 (misma silueta) el grupo de
valor se parte en dos sin aportar una idea nueva. El grupo D es pequeño, pero no es un error: es el reflejo
del CP3 (pérdidas) y funciona como una **alerta**.
""")

    st.subheader("5. ¿Para qué sirve y qué limitaciones tiene?")
    st.markdown("""
**Utilidad.** Para diversificar un portafolio no basta con repartir por sectores: conviene combinar
empresas de grupos distintos, porque responden a factores diferentes (valor, crecimiento, momentum). La
segmentación también ayuda a elegir **empresas comparables** para valoración por múltiplos, y el grupo D
señala posibles *trampas de valor*: un dividendo alto que en realidad refleja una caída del precio.

**Limitaciones.**
- Es una **foto de un solo momento**: el CP2 (posición y amplitud en 52 semanas) cambia con el mercado, así
  que una empresa puede pasar de grupo en pocas semanas.
- Se **excluyeron los bancos** por no tener EBITDA: los resultados no describen a la banca.
- K-Means supone grupos compactos y de tamaño parecido; con fronteras difusas como estas, la asignación de
  las empresas cercanas al límite es frágil (depende de la semilla y de la winsorización).
- Los datos de Yahoo Finance tienen errores puntuales; la winsorización mitiga, pero no corrige, esos errores.
""")
