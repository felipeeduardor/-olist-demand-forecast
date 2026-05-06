"""
=============================================================================
ANALISE DE DEMANDA — OLIST E-COMMERCE  |  v2 — Modelos Corrigidos
=============================================================================
Melhorias v2:
  - Prophet com hiperparametros conservadores (changepoint_prior_scale 0.05)
  - Feriados brasileiros adicionados ao modelo
  - Tratamento de outliers (IQR cap) antes do fit
  - Fallback Holt-Winters quando Prophet MAPE > 40%
  - Sanity check: previsao minima = 15% da media historica (elimina colapsos)
  - Cross-validation com janela inicial maior (548 dias)
Etapas:
  1. EDA — tendencia geral, sazonalidade, categorias
  2. EDA — evolucao por categoria
  3. Analise de cohort de crescimento
  4. Decomposicao da serie temporal
  5. Modelo (Prophet ou HW) com backtesting + previsao 3 meses
  6. Metricas de validacao
  7. Ranking de crescimento e insights de negocio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os

warnings.filterwarnings("ignore")

BASE   = r"C:\Users\nelzi\Downloads\archive"
OUTPUT = os.path.join(BASE, "forecast_output")
os.makedirs(OUTPUT, exist_ok=True)

PALETTE = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0",
           "#00BCD4","#FF5722","#795548","#607D8B","#FFEB3B"]

def fmt_k(x, _): return f"{int(x/1000)}k" if x >= 1000 else str(int(x))

FLOOR_RATIO   = 0.15   # previsao minima = 15% da media historica
MAPE_THRESHOLD = 40.0  # acima disso, usa Holt-Winters

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA E LIMPEZA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(" OLIST — ANALISE DE DEMANDA & PREVISAO  v2")
print("=" * 65)
print("\n[1/7] Carregando dados...")

orders   = pd.read_csv(os.path.join(BASE, "olist_orders_dataset.csv"),
                       parse_dates=["order_purchase_timestamp",
                                    "order_delivered_customer_date",
                                    "order_estimated_delivery_date"])
items    = pd.read_csv(os.path.join(BASE, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(BASE, "olist_products_dataset.csv"),
                       usecols=["product_id","product_category_name"])
trans    = pd.read_csv(os.path.join(BASE, "product_category_name_translation.csv"))
trans.columns = ["product_category_name","category_en"]
payments = pd.read_csv(os.path.join(BASE, "olist_order_payments_dataset.csv"))
reviews  = pd.read_csv(os.path.join(BASE, "olist_order_reviews_dataset.csv"),
                       usecols=["order_id","review_score"])
sellers  = pd.read_csv(os.path.join(BASE, "olist_sellers_dataset.csv"))
customers_ds = pd.read_csv(os.path.join(BASE, "olist_customers_dataset.csv"),
                           usecols=["customer_id","customer_unique_id"])

# Apenas entregues
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

# Remover primeiro e ultimo mes (dados incompletos)
min_m = delivered["month"].min()
max_m = delivered["month"].max()
delivered = delivered[(delivered["month"] > min_m) & (delivered["month"] < max_m)]

# Dataset mestre
df = (delivered
      .merge(items[["order_id","product_id","seller_id","price","freight_value"]], on="order_id")
      .merge(products, on="product_id")
      .merge(trans, on="product_category_name", how="left")
      .merge(payments.groupby("order_id")["payment_value"].sum().reset_index(), on="order_id", how="left")
      .merge(reviews.groupby("order_id")["review_score"].mean().reset_index(), on="order_id", how="left"))

df["revenue"]    = df["price"] + df["freight_value"]
df["delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days

print(f"   Pedidos entregues: {delivered['order_id'].nunique():,}")
print(f"   Itens             : {len(df):,}")
print(f"   Periodo           : {delivered['month'].min().strftime('%b/%Y')} -> {delivered['month'].max().strftime('%b/%Y')}")
print(f"   Categorias        : {df['category_en'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA — PAINEL GERAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Gerando EDA...")

monthly_total = df.groupby("month").agg(
    pedidos   = ("order_id","nunique"),
    receita   = ("revenue","sum"),
    ticket    = ("payment_value","mean"),
    nota_media= ("review_score","mean"),
).reset_index()

top10_cats = (df.groupby("category_en")["order_id"].nunique()
               .sort_values(ascending=False).head(10))

monthly_cat = (df[df["category_en"].isin(top10_cats.index)]
               .groupby(["category_en","month"])["order_id"]
               .nunique().reset_index(name="pedidos"))

# ── Fig 1: EDA Geral ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EDA — Visao Geral do E-commerce Olist", fontsize=15, fontweight="bold")

ax = axes[0,0]
ax.bar(monthly_total["month"], monthly_total["pedidos"], color="#2196F3", alpha=0.8, width=20)
ax.plot(monthly_total["month"], monthly_total["pedidos"].rolling(3).mean(),
        color="#E91E63", lw=2, label="Media 3 meses")
ax.set_title("Volume de Pedidos por Mes")
ax.set_ylabel("Pedidos")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[0,1]
ax.fill_between(monthly_total["month"], monthly_total["receita"]/1e6,
                color="#4CAF50", alpha=0.5)
ax.plot(monthly_total["month"], monthly_total["receita"]/1e6, color="#4CAF50", lw=2)
ax.set_title("Receita Mensal (R$ milhoes)")
ax.set_ylabel("R$ Milhoes")
ax.grid(axis="y", alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[0,2]
ax.plot(monthly_total["month"], monthly_total["ticket"], color="#FF9800", lw=2, marker="o", markersize=4)
ax.set_title("Ticket Medio por Pedido (R$)")
ax.set_ylabel("R$")
ax.grid(axis="y", alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1,0]
bars = ax.barh(top10_cats.index[::-1], top10_cats.values[::-1], color=PALETTE)
ax.set_title("Top 10 Categorias por Volume")
ax.set_xlabel("Pedidos")
for bar, val in zip(bars, top10_cats.values[::-1]):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)

df["mes_ano"] = df["order_purchase_timestamp"].dt.month
meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
saz = df.groupby("mes_ano")["order_id"].nunique()
ax = axes[1,1]
ax.bar(saz.index, saz.values, color=PALETTE[:12], alpha=0.85)
ax.set_xticks(range(1,13))
ax.set_xticklabels(meses_nomes)
ax.set_title("Sazonalidade — Pedidos por Mes do Ano")
ax.set_ylabel("Total de Pedidos")
ax.grid(axis="y", alpha=0.3)
pico = saz.idxmax()
ax.get_children()[pico-1].set_color("#E91E63")
ax.annotate(f"Pico: {meses_nomes[pico-1]}", xy=(pico, saz[pico]),
            xytext=(pico+0.5, saz[pico]+200), fontsize=8, color="#E91E63",
            arrowprops=dict(arrowstyle="->", color="#E91E63"))

delay_score = df.dropna(subset=["delay_days","review_score"]).copy()
delay_bins = pd.cut(delay_score["delay_days"], bins=[-60,-14,-7,-1,0,7,14,30,90],
                    labels=["<-14","-14a-7","-7a-1","No dia","1a7","7a14","14a30",">30"])
ax = axes[1,2]
ds = delay_score.groupby(delay_bins, observed=True)["review_score"].mean()
def _label_is_early(lbl):
    s = str(lbl).split("a")[0].replace("<","").replace(">","").replace("No di","0").strip()
    try: return float(s) <= 0
    except: return False
colors_ds = ["#4CAF50" if _label_is_early(l) else "#E91E63" for l in ds.index]
ax.bar(ds.index.astype(str), ds.values, color=colors_ds, alpha=0.85)
ax.axhline(3.0, color="gray", linestyle="--", lw=1)
ax.set_title("Nota Media vs Atraso na Entrega")
ax.set_xlabel("Dias (negativo = adiantado)")
ax.set_ylabel("Nota Media (1-5)")
ax.set_ylim(1, 5)
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "01_eda_geral.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   01_eda_geral.png salvo")

# ── Fig 2: Evolucao por categoria ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
for i, cat in enumerate(top10_cats.index):
    data = monthly_cat[monthly_cat["category_en"] == cat]
    ax.plot(data["month"], data["pedidos"], marker="o", markersize=3,
            lw=1.8, label=cat.replace("_"," ").title(), color=PALETTE[i])

ax.set_title("Evolucao Mensal das Top 10 Categorias", fontsize=13, fontweight="bold")
ax.set_ylabel("Pedidos / mes")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "02_evolucao_categorias.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   02_evolucao_categorias.png salvo")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DECOMPOSICAO DA SERIE TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Decomposicao da serie temporal...")

serie_total = monthly_total.set_index("month")["pedidos"]
trend     = serie_total.rolling(window=3, center=True).mean()
detrended = serie_total / trend
seasonal  = detrended.groupby(detrended.index.month).transform("mean")
residual  = serie_total / (trend * seasonal)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Decomposicao da Serie Temporal — Volume Total de Pedidos",
             fontsize=13, fontweight="bold")

for ax, data, label, color in zip(
    axes,
    [serie_total, trend, seasonal, residual],
    ["Original", "Tendencia (media movel 3m)", "Sazonalidade", "Residuo"],
    ["#2196F3","#E91E63","#4CAF50","#FF9800"]
):
    ax.plot(data.index, data.values, color=color, lw=1.8)
    ax.fill_between(data.index, data.values, alpha=0.15, color=color)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(alpha=0.3)
    if label == "Tendencia (media movel 3m)":
        x_num = np.arange(len(trend.dropna()))
        y_num = trend.dropna().values
        coef  = np.polyfit(x_num, y_num, 1)
        ax.plot(trend.dropna().index,
                np.poly1d(coef)(x_num),
                "--", color="black", lw=1, label=f"Crescimento: +{coef[0]:.0f} ped/mes")
        ax.legend(fontsize=8)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "03_decomposicao.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   03_decomposicao.png salvo")

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCOES DE MODELAGEM
# ═══════════════════════════════════════════════════════════════════════════════

def cap_outliers(serie: pd.Series) -> pd.Series:
    """Limita outliers superiores em Q3 + 1.5*IQR para nao distorcer o fit."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return serie.clip(upper=upper)


def sanity_floor(yhat: pd.Series, hist_mean: float) -> pd.Series:
    """Garante previsao minima de FLOOR_RATIO * media historica."""
    floor = max(1.0, FLOOR_RATIO * hist_mean)
    return yhat.clip(lower=floor)


def fit_prophet(y_df: pd.DataFrame) -> tuple:
    """
    Treina Prophet conservador com feriados BR.
    Retorna (forecast_df, mape, mae).
    """
    model = Prophet(
        yearly_seasonality     = True,
        weekly_seasonality     = False,
        daily_seasonality      = False,
        seasonality_mode       = "multiplicative",
        changepoint_prior_scale = 0.05,   # conservador (era 0.15)
        seasonality_prior_scale = 5.0,    # regulariza sazonalidade
        n_changepoints          = 10,
        interval_width          = 0.90,
    )
    model.add_country_holidays(country_name="BR")
    model.fit(y_df)

    # Cross-validation com janela inicial maior e poucos folds
    n_meses = len(y_df)
    initial_days = max(365, int(n_meses * 0.6) * 30)  # pelo menos 60% dos dados
    try:
        cv = cross_validation(
            model,
            initial = f"{initial_days} days",
            period  = "90 days",
            horizon = "90 days",
            disable_tqdm = True,
        )
        pm   = performance_metrics(cv)
        mape = pm["mape"].mean() * 100
        mae  = pm["mae"].mean()
    except Exception:
        mape, mae = 999.0, 0.0

    return model, mape, mae


def fit_holtwinters(y_df: pd.DataFrame, meses_prev: int) -> tuple:
    """
    Treina Holt-Winters (ETS) como fallback.
    Retorna (forecast_series, mape_estimado, mae_estimado).
    """
    ts = y_df.set_index("ds")["y"]
    n  = len(ts)

    # Escolhe tendencia/sazonalidade baseado no tamanho da serie
    seasonal_periods = 12
    use_seasonal = n >= seasonal_periods * 2

    try:
        hw = ExponentialSmoothing(
            ts,
            trend           = "add",
            seasonal        = "add" if use_seasonal else None,
            seasonal_periods = seasonal_periods if use_seasonal else None,
            damped_trend    = True,
        ).fit(optimized=True)

        # MAPE estimado em holdout simples (ultimos 3 meses)
        split    = max(3, int(n * 0.8))
        hw_train = ExponentialSmoothing(
            ts.iloc[:split],
            trend            = "add",
            seasonal         = "add" if split >= seasonal_periods * 2 else None,
            seasonal_periods = seasonal_periods if split >= seasonal_periods * 2 else None,
            damped_trend     = True,
        ).fit(optimized=True)
        preds  = hw_train.forecast(n - split)
        actual = ts.iloc[split:].values
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.nanmean(np.abs((actual - preds.values) / actual)) * 100)
        mae  = float(np.nanmean(np.abs(actual - preds.values)))

        # Previsao futura
        future_vals = hw.forecast(meses_prev)
        future_idx  = pd.date_range(
            ts.index[-1] + pd.DateOffset(months=1),
            periods = meses_prev,
            freq    = "MS",
        )
        forecast_series = pd.Series(future_vals.values, index=future_idx)
        return forecast_series, mape, mae

    except Exception as e:
        # Fallback ultimo recurso: media movel simples
        mean_val        = ts.tail(6).mean()
        future_idx      = pd.date_range(
            ts.index[-1] + pd.DateOffset(months=1),
            periods = meses_prev,
            freq    = "MS",
        )
        forecast_series = pd.Series([mean_val] * meses_prev, index=future_idx)
        return forecast_series, 999.0, 0.0


def forecast_categoria(serie_df: pd.DataFrame, meses_prev: int = 3) -> dict:
    """
    Tenta Prophet; se MAPE > MAPE_THRESHOLD usa Holt-Winters.
    Retorna dict com previsoes, metricas e modelo usado.
    """
    y_df = (serie_df[["month","pedidos"]]
            .rename(columns={"month":"ds","pedidos":"y"})
            .sort_values("ds"))

    # Cap de outliers nos dados de treino
    y_df["y"] = cap_outliers(y_df["y"]).values
    hist_mean = float(y_df["y"].tail(6).mean())

    model_prophet, mape_p, mae_p = fit_prophet(y_df)

    if mape_p <= MAPE_THRESHOLD:
        # Prophet aprovado
        future   = model_prophet.make_future_dataframe(periods=meses_prev, freq="MS")
        forecast = model_prophet.predict(future)
        fut      = forecast[forecast["ds"] > y_df["ds"].max()].copy()

        yhat       = sanity_floor(fut["yhat"].clip(0),        hist_mean)
        yhat_lower = sanity_floor(fut["yhat_lower"].clip(0),  hist_mean)
        yhat_upper = fut["yhat_upper"].clip(0)

        return {
            "modelo"     : "Prophet",
            "mape"       : round(mape_p, 1),
            "mae"        : round(mae_p, 1),
            "datas"      : fut["ds"].tolist(),
            "yhat"       : yhat.round(0).astype(int).tolist(),
            "yhat_lower" : yhat_lower.round(0).astype(int).tolist(),
            "yhat_upper" : yhat_upper.round(0).astype(int).tolist(),
            "prophet_fc" : forecast,       # para plot
            "y_df"       : y_df,
        }
    else:
        # Fallback Holt-Winters
        hw_fc, mape_hw, mae_hw = fit_holtwinters(y_df, meses_prev)
        hw_fc_clipped = sanity_floor(hw_fc.clip(0), hist_mean)

        # Intervalo de confianca estimado: ±15% para HW (sem IC nativo)
        ic_margin = hw_fc_clipped * 0.15

        # Montar forecast completo para plot (historico + futuro)
        hist = y_df.copy()
        fut_df = pd.DataFrame({
            "ds"         : hw_fc_clipped.index,
            "yhat"       : hw_fc_clipped.values,
            "yhat_lower" : (hw_fc_clipped - ic_margin).clip(0).values,
            "yhat_upper" : (hw_fc_clipped + ic_margin).values,
        })
        hist_renamed = y_df[["ds","y"]].rename(columns={"y":"yhat"})
        prophet_like = pd.concat([
            hist_renamed,
            fut_df[["ds","yhat"]],
        ], ignore_index=True)
        prophet_like["yhat_lower"] = prophet_like["yhat"] * 0.85
        prophet_like["yhat_upper"] = prophet_like["yhat"] * 1.15

        return {
            "modelo"     : "Holt-Winters",
            "mape"       : round(mape_hw, 1),
            "mae"        : round(mae_hw, 1),
            "datas"      : hw_fc_clipped.index.tolist(),
            "yhat"       : hw_fc_clipped.round(0).astype(int).tolist(),
            "yhat_lower" : (hw_fc_clipped - ic_margin).clip(0).round(0).astype(int).tolist(),
            "yhat_upper" : (hw_fc_clipped + ic_margin).round(0).astype(int).tolist(),
            "prophet_fc" : prophet_like,
            "y_df"       : y_df,
            "is_hw"      : True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TREINAMENTO DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Treinando modelos (Prophet + HW fallback)...")

MESES_PREV = 3
resultados_fc = {}
metricas      = []

for cat in top10_cats.index:
    serie = monthly_cat[monthly_cat["category_en"] == cat]
    fc    = forecast_categoria(serie, MESES_PREV)
    resultados_fc[cat] = fc
    metricas.append({
        "categoria": cat,
        "modelo"   : fc["modelo"],
        "MAPE_%"   : fc["mape"],
        "MAE"      : fc["mae"],
    })
    status = "OK" if fc["mape"] <= MAPE_THRESHOLD else "HW"
    print(f"   [{list(top10_cats.index).index(cat)+1:02d}/10] {cat:<35} "
          f"[{fc['modelo']:<13}]  MAPE={fc['mape']:>6.1f}%  MAE={fc['mae']:>7.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRAFICOS DE FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Gerando graficos de previsao...")

fig_grid, axes_g = plt.subplots(5, 2, figsize=(18, 24))
fig_grid.suptitle("Previsao de Demanda por Categoria — Proximos 3 Meses (v2)",
                   fontsize=15, fontweight="bold", y=1.005)
axes_g = axes_g.flatten()

for idx, cat in enumerate(top10_cats.index):
    fc     = resultados_fc[cat]
    y_df   = fc["y_df"]
    fcast  = fc["prophet_fc"]
    is_hw  = fc.get("is_hw", False)

    ax = axes_g[idx]

    if is_hw:
        # Plot HW: historico + previsao separados
        ax.plot(y_df["ds"], y_df["y"], "o", color="#333333", markersize=4, label="Real", zorder=5)
        fut_dates = pd.to_datetime(fc["datas"])
        ax.fill_between(fut_dates,
                        fc["yhat_lower"], fc["yhat_upper"],
                        alpha=0.15, color=PALETTE[idx], label="IC ~85%")
        ax.plot(fut_dates, fc["yhat"], color=PALETTE[idx], lw=2, label="Previsao HW", marker="s", markersize=5)
        ax.axvspan(fut_dates.min(), fut_dates.max(), alpha=0.07, color="green")
        for d, v in zip(fut_dates, fc["yhat"]):
            ax.annotate(f"{int(v)}", xy=(d, v), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=8, color="darkgreen", fontweight="bold")
    else:
        ax.fill_between(fcast["ds"],
                        fcast["yhat_lower"].clip(0), fcast["yhat_upper"].clip(0),
                        alpha=0.15, color=PALETTE[idx], label="IC 90%")
        ax.plot(fcast["ds"], fcast["yhat"].clip(0),
                color=PALETTE[idx], lw=2, label="Previsao Prophet")
        ax.plot(y_df["ds"], y_df["y"], "o", color="#333333",
                markersize=4, zorder=5, label="Real")
        fut_dates = pd.to_datetime(fc["datas"])
        ax.axvspan(fut_dates.min(), fut_dates.max(), alpha=0.07, color="green")
        for d, v in zip(fut_dates, fc["yhat"]):
            ax.annotate(f"{int(v)}", xy=(d, v), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=8, color="darkgreen", fontweight="bold")

    mape_color = "#4CAF50" if fc["mape"] <= 20 else "#FF9800" if fc["mape"] <= MAPE_THRESHOLD else "#E91E63"
    ax.set_title(f"{cat.replace('_',' ').title()} [{fc['modelo']}]",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Pedidos / mes", fontsize=8)
    ax.text(0.99, 0.04, f"MAPE={fc['mape']:.1f}%  MAE={fc['mae']:.0f}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=mape_color)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_grid.savefig(os.path.join(OUTPUT, "04_forecast_categorias.png"), dpi=150, bbox_inches="tight")
plt.close(fig_grid)
print("   04_forecast_categorias.png salvo")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRICAS DE VALIDACAO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Gerando relatorio de metricas...")

df_met = pd.DataFrame(metricas).sort_values("MAPE_%")
fig, ax = plt.subplots(figsize=(11, 5))
colors_met = ["#4CAF50" if m < 20 else "#FF9800" if m < MAPE_THRESHOLD else "#E91E63"
              for m in df_met["MAPE_%"]]
bars = ax.barh(
    df_met.apply(lambda r: f"{r['categoria'].replace('_',' ').title()} [{r['modelo'][:2]}]", axis=1),
    df_met["MAPE_%"], color=colors_met, alpha=0.85
)
for bar, val in zip(bars, df_met["MAPE_%"]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.axvline(20,              color="#4CAF50", linestyle="--", lw=1.5, label="Otimo (<20%)")
ax.axvline(MAPE_THRESHOLD,  color="#FF9800", linestyle="--", lw=1.5, label=f"Aceitavel (<{MAPE_THRESHOLD:.0f}%)")
ax.set_xlabel("MAPE — Erro Percentual Medio Absoluto")
ax.set_title("Qualidade do Modelo por Categoria (Cross-Validation / Holdout) — v2",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "05_metricas_modelo.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   05_metricas_modelo.png salvo")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RANKING DE CRESCIMENTO + RESUMO FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Gerando ranking e salvando resultados...")

# Salvar previsoes
resultado_rows = []
for cat, fc in resultados_fc.items():
    for d, yh, yl, yu in zip(fc["datas"], fc["yhat"], fc["yhat_lower"], fc["yhat_upper"]):
        resultado_rows.append({
            "categoria" : cat,
            "modelo"    : fc["modelo"],
            "mes"       : pd.Timestamp(d).strftime("%Y-%m"),
            "previsao"  : int(yh),
            "limite_inf": int(yl),
            "limite_sup": int(yu),
        })

resultado_final = pd.DataFrame(resultado_rows)
resultado_final.to_csv(os.path.join(OUTPUT,"previsao_demanda.csv"), index=False)
pd.DataFrame(metricas).to_csv(os.path.join(OUTPUT,"metricas_modelo.csv"), index=False)

# Ranking
ultimos3 = (monthly_cat[monthly_cat["category_en"].isin(top10_cats.index)]
            .sort_values("month").groupby("category_en").tail(3)
            .groupby("category_en")["pedidos"].mean().round(0))

meses_previstos = sorted(resultado_final["mes"].unique())
ranking = []
for cat in top10_cats.index:
    media_real = float(ultimos3.get(cat, 0))
    prev_vals  = resultado_final[resultado_final["categoria"] == cat]["previsao"].tolist()
    media_prev = float(np.mean(prev_vals)) if prev_vals else 0.0
    variacao   = (media_prev - media_real) / media_real * 100 if media_real > 0 else 0.0
    fc         = resultados_fc[cat]
    ranking.append((cat, int(media_real), prev_vals, variacao, fc["mape"], fc["modelo"]))

ranking.sort(key=lambda x: x[3], reverse=True)

# Fig 6 — Ranking visual
fig, ax = plt.subplots(figsize=(10, 6))
cats_r   = [r[0].replace("_"," ").title() for r in ranking]
vars_r   = [r[3] for r in ranking]
models_r = [r[5] for r in ranking]
colors_r = ["#4CAF50" if v >= 0 else "#E91E63" for v in vars_r]
bars = ax.barh(cats_r[::-1], vars_r[::-1], color=colors_r[::-1], alpha=0.85)
for bar, val, mod in zip(bars, vars_r[::-1], models_r[::-1]):
    sinal = "+" if val >= 0 else ""
    label = f"{sinal}{val:.1f}% [{mod[:2]}]"
    ax.text(val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height()/2,
            label, va="center",
            ha="left" if val >= 0 else "right", fontsize=9, fontweight="bold")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Variacao Prevista vs Media Ultimos 3 Meses (%)")
ax.set_title("Ranking de Crescimento de Demanda — Proximos 3 Meses (v2)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT,"06_ranking_crescimento.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   06_ranking_crescimento.png salvo")

# ── Console Report ────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(" RELATORIO FINAL — PREVISAO DE DEMANDA OLIST  v2")
print("=" * 75)
print(f"\n{'Categoria':<32} {'Atual':>7}", end="")
for m in meses_previstos: print(f"  {m:>8}", end="")
print(f"  {'Var%':>7}  {'MAPE':>6}  {'Modelo'}")
print("-" * 80)

for cat, media_real, prev_vals, variacao, mape_cat, modelo in ranking:
    sinal  = "^" if variacao >= 0 else "v"
    mape_s = f"{mape_cat:.0f}%"
    print(f"{cat.replace('_',' ').title():<32} {media_real:>7}", end="")
    for v in prev_vals: print(f"  {int(v):>8}", end="")
    print(f"  {sinal}{abs(variacao):>5.1f}%  {mape_s:>6}  {modelo}")

print("\n LEGENDA MAPE: Verde <20% (otimo)  |  Laranja <40% (aceitavel)  |  Vermelho >40%")
print(" LEGENDA MODELO: Pr=Prophet  |  Ho=Holt-Winters (fallback)")
print(f"\n ARQUIVOS GERADOS em: {OUTPUT}")
for f in ["01_eda_geral.png","02_evolucao_categorias.png","03_decomposicao.png",
          "04_forecast_categorias.png","05_metricas_modelo.png","06_ranking_crescimento.png",
          "previsao_demanda.csv","metricas_modelo.csv"]:
    print(f"   {f}")
print("\nConcluido! (v2)")
