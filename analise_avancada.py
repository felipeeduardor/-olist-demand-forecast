"""
=============================================================================
ANALISE AVANCADA — OLIST E-COMMERCE  |  Parte 2
=============================================================================
  A. Analise geografica por estado
  B. Previsao de receita (Prophet)
  C. Segmentacao de clientes RFM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from prophet import Prophet
import warnings
import os

warnings.filterwarnings("ignore")

BASE   = r"C:\Users\nelzi\Downloads\archive"
OUTPUT = os.path.join(BASE, "forecast_output")
os.makedirs(OUTPUT, exist_ok=True)

PALETTE = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0",
           "#00BCD4","#FF5722","#795548","#607D8B","#FFEB3B"]

def fmt_k(x, _): return f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1000)}k" if x >= 1000 else str(int(x))

print("=" * 65)
print(" OLIST — ANALISE AVANCADA (Estado + Receita + RFM)")
print("=" * 65)

# ── Carga ─────────────────────────────────────────────────────────────────────
print("\nCarregando dados...")
orders    = pd.read_csv(os.path.join(BASE, "olist_orders_dataset.csv"),
                        parse_dates=["order_purchase_timestamp",
                                     "order_delivered_customer_date"])
items     = pd.read_csv(os.path.join(BASE, "olist_order_items_dataset.csv"))
products  = pd.read_csv(os.path.join(BASE, "olist_products_dataset.csv"),
                        usecols=["product_id","product_category_name"])
trans     = pd.read_csv(os.path.join(BASE, "product_category_name_translation.csv"))
trans.columns = ["product_category_name","category_en"]
payments  = pd.read_csv(os.path.join(BASE, "olist_order_payments_dataset.csv"))
customers = pd.read_csv(os.path.join(BASE, "olist_customers_dataset.csv"))
reviews   = pd.read_csv(os.path.join(BASE, "olist_order_reviews_dataset.csv"),
                        usecols=["order_id","review_score"])

delivered = orders[orders["order_status"] == "delivered"].copy()
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

min_m = delivered["month"].min()
max_m = delivered["month"].max()
delivered = delivered[(delivered["month"] > min_m) & (delivered["month"] < max_m)]

pay_agg = payments.groupby("order_id")["payment_value"].sum().reset_index()

df = (delivered
      .merge(items[["order_id","product_id","price","freight_value"]], on="order_id")
      .merge(products, on="product_id")
      .merge(trans, on="product_category_name", how="left")
      .merge(pay_agg, on="order_id", how="left")
      .merge(customers[["customer_id","customer_unique_id","customer_state"]], on="customer_id", how="left"))

df["revenue"] = df["price"] + df["freight_value"]

print(f"   Dataset pronto: {len(df):,} linhas")

# =============================================================================
# A. ANALISE GEOGRAFICA POR ESTADO
# =============================================================================
print("\n[A] Analise geografica por estado...")

state_stats = df.groupby("customer_state").agg(
    pedidos  = ("order_id","nunique"),
    receita  = ("revenue","sum"),
    ticket   = ("payment_value","mean"),
    clientes = ("customer_unique_id","nunique"),
).reset_index().sort_values("pedidos", ascending=False)

state_stats["receita_por_cliente"] = state_stats["receita"] / state_stats["clientes"]

# Evolucao mensal por estado (top 8)
top8_states = state_stats.head(8)["customer_state"].tolist()
monthly_state = (df[df["customer_state"].isin(top8_states)]
                 .groupby(["customer_state","month"])["order_id"]
                 .nunique().reset_index(name="pedidos"))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Analise Geografica por Estado", fontsize=15, fontweight="bold")

# A1 — Ranking de estados por pedidos
ax = axes[0,0]
top15 = state_stats.head(15)
colors_state = ["#E91E63" if s == "SP" else "#2196F3" for s in top15["customer_state"]]
bars = ax.barh(top15["customer_state"][::-1], top15["pedidos"][::-1],
               color=colors_state[::-1], alpha=0.85)
for bar, val in zip(bars, top15["pedidos"][::-1]):
    ax.text(val + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=7.5)
ax.set_title("Top 15 Estados por Volume de Pedidos")
ax.set_xlabel("Pedidos")
ax.grid(axis="x", alpha=0.3)

# A2 — Receita por estado
ax = axes[0,1]
top15_rec = state_stats.sort_values("receita", ascending=False).head(15)
bars = ax.barh(top15_rec["customer_state"][::-1], top15_rec["receita"][::-1]/1e6,
               color="#4CAF50", alpha=0.85)
for bar, val in zip(bars, top15_rec["receita"][::-1]/1e6):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"R${val:.1f}M", va="center", fontsize=7.5)
ax.set_title("Top 15 Estados por Receita Total")
ax.set_xlabel("Receita (R$ Milhoes)")
ax.grid(axis="x", alpha=0.3)

# A3 — Ticket medio por estado
ax = axes[1,0]
ticket_order = state_stats.sort_values("ticket", ascending=False).head(15)
bars = ax.barh(ticket_order["customer_state"][::-1], ticket_order["ticket"][::-1],
               color="#FF9800", alpha=0.85)
for bar, val in zip(bars, ticket_order["ticket"][::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"R${val:.0f}", va="center", fontsize=7.5)
ax.set_title("Ticket Medio por Estado (R$)")
ax.set_xlabel("Ticket Medio (R$)")
ax.grid(axis="x", alpha=0.3)

# A4 — Evolucao top 8 estados
ax = axes[1,1]
for i, state in enumerate(top8_states):
    data = monthly_state[monthly_state["customer_state"] == state]
    ax.plot(data["month"], data["pedidos"], marker="o", markersize=3,
            lw=1.8, label=state, color=PALETTE[i])
ax.set_title("Evolucao Mensal — Top 8 Estados")
ax.set_ylabel("Pedidos / mes")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT,"07_analise_estados.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   07_analise_estados.png salvo")

state_stats.to_csv(os.path.join(OUTPUT,"estados_stats.csv"), index=False)
print("   estados_stats.csv salvo")

# Console
print("\n   TOP 10 ESTADOS:")
print(f"   {'Estado':<8} {'Pedidos':>8} {'Receita':>12} {'Ticket Medio':>14} {'Clientes':>10}")
print("   " + "-"*55)
for _, row in state_stats.head(10).iterrows():
    print(f"   {row['customer_state']:<8} {int(row['pedidos']):>8,} "
          f"  R${row['receita']/1e6:>7.1f}M  "
          f"R${row['ticket']:>10.0f}  "
          f"{int(row['clientes']):>10,}")

# =============================================================================
# B. PREVISAO DE RECEITA
# =============================================================================
print("\n[B] Previsao de receita...")

monthly_rev = df.groupby("month").agg(
    receita=("revenue","sum"),
    pedidos=("order_id","nunique"),
).reset_index()

# Receita por top 5 categorias
top5_cats = (df.groupby("category_en")["revenue"].sum()
               .sort_values(ascending=False).head(5).index.tolist())
monthly_rev_cat = (df[df["category_en"].isin(top5_cats)]
                   .groupby(["category_en","month"])["revenue"]
                   .sum().reset_index())

MESES_PREV = 3
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Previsao de Receita — Proximos 3 Meses", fontsize=15, fontweight="bold")

# B1 — Receita total
ax = axes[0,0]
serie = monthly_rev[["month","receita"]].rename(columns={"month":"ds","receita":"y"})
model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="multiplicative",
                interval_width=0.90, changepoint_prior_scale=0.15)
model.fit(serie)
future   = model.make_future_dataframe(periods=MESES_PREV, freq="MS")
forecast = model.predict(future)

ax.fill_between(forecast["ds"], forecast["yhat_lower"].clip(0)/1e6,
                forecast["yhat_upper"].clip(0)/1e6, alpha=0.15, color="#4CAF50")
ax.plot(forecast["ds"], forecast["yhat"].clip(0)/1e6, color="#4CAF50", lw=2, label="Previsao")
ax.plot(serie["ds"], serie["y"]/1e6, "o", color="#333", markersize=4, label="Real")

fut_total = forecast[forecast["ds"] > serie["ds"].max()]
if len(fut_total):
    ax.axvspan(fut_total["ds"].min(), fut_total["ds"].max(), alpha=0.07, color="green")
    for _, row in fut_total.iterrows():
        yval = max(0, row["yhat"])
        ax.annotate(f"R${yval/1e6:.2f}M",
                    xy=(row["ds"], yval/1e6),
                    xytext=(0,10), textcoords="offset points",
                    ha="center", fontsize=8, color="darkgreen", fontweight="bold")

ax.set_title("Receita Total", fontweight="bold")
ax.set_ylabel("R$ Milhoes")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

rev_resultados = []
for i, cat in enumerate(top5_cats):
    ax = axes.flatten()[i+1]
    data = monthly_rev_cat[monthly_rev_cat["category_en"] == cat][["month","revenue"]]
    serie = data.rename(columns={"month":"ds","revenue":"y"}).sort_values("ds")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative",
                    interval_width=0.90, changepoint_prior_scale=0.15)
    model.fit(serie)
    future   = model.make_future_dataframe(periods=MESES_PREV, freq="MS")
    forecast = model.predict(future)

    fut_rows = forecast[forecast["ds"] > serie["ds"].max()].copy()
    fut_rows["categoria"] = cat
    rev_resultados.append(fut_rows[["ds","yhat","yhat_lower","yhat_upper","categoria"]])

    color = PALETTE[i+1]
    ax.fill_between(forecast["ds"], forecast["yhat_lower"].clip(0)/1e3,
                    forecast["yhat_upper"].clip(0)/1e3, alpha=0.15, color=color)
    ax.plot(forecast["ds"], forecast["yhat"].clip(0)/1e3, color=color, lw=2)
    ax.plot(serie["ds"], serie["y"]/1e3, "o", color="#333", markersize=4)

    if len(fut_rows):
        ax.axvspan(fut_rows["ds"].min(), fut_rows["ds"].max(), alpha=0.07, color="green")
        for _, row in fut_rows.iterrows():
            yval = max(0, row["yhat"])
            ax.annotate(f"R${yval/1e3:.0f}k",
                        xy=(row["ds"], yval/1e3),
                        xytext=(0,10), textcoords="offset points",
                        ha="center", fontsize=8, color="darkgreen", fontweight="bold")

    ax.set_title(cat.replace("_"," ").title(), fontweight="bold")
    ax.set_ylabel("R$ Mil")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    print(f"   Categoria {cat} — OK")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT,"08_forecast_receita.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   08_forecast_receita.png salvo")

if rev_resultados:
    df_rev = pd.concat(rev_resultados, ignore_index=True)
    df_rev["mes"] = df_rev["ds"].dt.strftime("%Y-%m")
    df_rev["receita_prevista_R$"] = df_rev["yhat"].clip(0).round(2)
    df_rev["limite_inf"]          = df_rev["yhat_lower"].clip(0).round(2)
    df_rev["limite_sup"]          = df_rev["yhat_upper"].clip(0).round(2)
    df_rev[["categoria","mes","receita_prevista_R$","limite_inf","limite_sup"]].to_csv(
        os.path.join(OUTPUT,"previsao_receita.csv"), index=False)
    print("   previsao_receita.csv salvo")

# Console — receita prevista total
print(f"\n   RECEITA TOTAL PREVISTA:")
for _, row in fut_total.iterrows():
    print(f"   {row['ds'].strftime('%b/%Y')}  ->  R${max(0,row['yhat'])/1e6:.2f}M "
          f"  (IC: R${max(0,row['yhat_lower'])/1e6:.2f}M - R${max(0,row['yhat_upper'])/1e6:.2f}M)")

# =============================================================================
# C. SEGMENTACAO RFM
# =============================================================================
print("\n[C] Segmentacao RFM de clientes...")

# Data de referencia = dia seguinte ao ultimo pedido
ref_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

# Juntar pagamentos
df_rfm_base = (delivered
               .merge(pay_agg, on="order_id", how="left")
               .merge(customers[["customer_id","customer_unique_id","customer_state"]],
                      on="customer_id", how="left"))

rfm = df_rfm_base.groupby("customer_unique_id").agg(
    recency   = ("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
    frequency = ("order_id","nunique"),
    monetary  = ("payment_value","sum"),
    state     = ("customer_state","first"),
).reset_index()

# Scores 1-5 (quintis)
rfm["R_score"] = pd.qcut(rfm["recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"],  q=5, labels=[1,2,3,4,5]).astype(int)
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

# Segmentacao
def segmentar(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4:                    return "Champions"
    elif r >= 3 and f >= 3:                  return "Loyal"
    elif r >= 4 and f <= 2:                  return "Recent"
    elif r >= 3 and f <= 2 and m >= 3:       return "Potential Loyalist"
    elif r <= 2 and f >= 3:                  return "At Risk"
    elif r == 1 and f >= 3:                  return "Cant Lose"
    elif r <= 2 and f <= 2 and m >= 3:       return "Hibernating"
    else:                                     return "Lost"

rfm["segment"] = rfm.apply(segmentar, axis=1)

seg_counts = rfm["segment"].value_counts()
seg_stats  = rfm.groupby("segment").agg(
    clientes  = ("customer_unique_id","count"),
    rec_media = ("recency","mean"),
    freq_media= ("frequency","mean"),
    gmv_medio = ("monetary","mean"),
    gmv_total = ("monetary","sum"),
).round(1).reset_index().sort_values("gmv_total", ascending=False)

# Cores por segmento
SEG_COLORS = {
    "Champions":        "#4CAF50",
    "Loyal":            "#2196F3",
    "Recent":           "#00BCD4",
    "Potential Loyalist":"#8BC34A",
    "At Risk":          "#FF9800",
    "Cant Lose":        "#E91E63",
    "Hibernating":      "#9E9E9E",
    "Lost":             "#F44336",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Segmentacao RFM de Clientes", fontsize=15, fontweight="bold")

# C1 — Distribuicao dos segmentos
ax = axes[0,0]
colors_seg = [SEG_COLORS.get(s,"#999") for s in seg_counts.index]
wedges, texts, autotexts = ax.pie(
    seg_counts.values, labels=seg_counts.index,
    colors=colors_seg, autopct="%1.1f%%",
    startangle=140, pctdistance=0.8)
for t in autotexts: t.set_fontsize(8)
ax.set_title("Distribuicao de Clientes por Segmento")

# C2 — GMV total por segmento
ax = axes[0,1]
seg_gmv = seg_stats.sort_values("gmv_total")
colors_gmv = [SEG_COLORS.get(s,"#999") for s in seg_gmv["segment"]]
bars = ax.barh(seg_gmv["segment"], seg_gmv["gmv_total"]/1e6, color=colors_gmv, alpha=0.85)
for bar, val in zip(bars, seg_gmv["gmv_total"]/1e6):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"R${val:.1f}M", va="center", fontsize=8)
ax.set_title("GMV Total por Segmento (R$ Milhoes)")
ax.set_xlabel("R$ Milhoes")
ax.grid(axis="x", alpha=0.3)

# C3 — Scatter R vs M colorido por segmento
ax = axes[1,0]
for seg, color in SEG_COLORS.items():
    sub = rfm[rfm["segment"] == seg]
    ax.scatter(sub["recency"], sub["monetary"], c=color, alpha=0.4,
               s=15, label=seg, rasterized=True)
ax.set_xlabel("Recency (dias desde ultima compra)")
ax.set_ylabel("Monetary (R$ total gasto)")
ax.set_title("Recency vs Monetary por Segmento")
ax.legend(fontsize=7, markerscale=1.5)
ax.set_yscale("log")
ax.grid(alpha=0.2)

# C4 — Tabela resumo por segmento
ax = axes[1,1]
ax.axis("off")
table_data = [["Segmento","Clientes","Rec.Media","Freq.Media","GMV Medio"]]
for _, row in seg_stats.iterrows():
    table_data.append([
        row["segment"],
        f"{int(row['clientes']):,}",
        f"{int(row['rec_media'])}d",
        f"{row['freq_media']:.1f}x",
        f"R${row['gmv_medio']:.0f}",
    ])
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="center", loc="center",
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
for j in range(len(table_data[0])):
    table[0,j].set_facecolor("#37474F")
    table[0,j].set_text_props(color="white", fontweight="bold")
for i, row_d in enumerate(seg_stats.itertuples(), start=1):
    color = SEG_COLORS.get(row_d.segment,"#999")
    table[i,0].set_facecolor(color + "55")
ax.set_title("Resumo por Segmento", pad=12, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT,"09_rfm_segmentacao.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   09_rfm_segmentacao.png salvo")

rfm.to_csv(os.path.join(OUTPUT,"clientes_rfm.csv"), index=False)
seg_stats.to_csv(os.path.join(OUTPUT,"rfm_segmentos_stats.csv"), index=False)
print("   clientes_rfm.csv salvo")
print("   rfm_segmentos_stats.csv salvo")

# Console
print("\n   RESUMO RFM:")
print(f"   {'Segmento':<22} {'Clientes':>9} {'Rec.Media':>10} {'Freq':>6} {'GMV Medio':>11} {'GMV Total':>12}")
print("   " + "-"*75)
for _, row in seg_stats.sort_values("clientes", ascending=False).iterrows():
    print(f"   {row['segment']:<22} {int(row['clientes']):>9,} "
          f"{int(row['rec_media']):>8}d  "
          f"{row['freq_media']:>5.1f}x  "
          f"R${row['gmv_medio']:>8.0f}  "
          f"R${row['gmv_total']/1e6:>8.1f}M")

# ── Resumo final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" ARQUIVOS GERADOS EM:", OUTPUT)
print("=" * 65)
print("   07_analise_estados.png   — Ranking e evolucao por estado")
print("   08_forecast_receita.png  — Previsao de receita (Prophet)")
print("   09_rfm_segmentacao.png   — Segmentacao RFM de clientes")
print("   estados_stats.csv        — Metricas por estado")
print("   previsao_receita.csv     — Receita prevista por categoria")
print("   clientes_rfm.csv         — Score RFM de cada cliente")
print("   rfm_segmentos_stats.csv  — Resumo por segmento")
print("\nConcluido!")
