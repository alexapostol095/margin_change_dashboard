# margin_change_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# ---------------------------
# Page config & global styles
# ---------------------------
st.set_page_config(page_title="Margin Change Analyzer", layout="wide")

# Light theme + dark text everywhere
st.markdown("""
<style>
:root{
  --bg: #F5F8FF;
  --card: #FFFFFF;
  --text: #111827;      /* dark slate */
  --muted: #374151;
  --border: #E5E7EB;
  --accent: #3C37FF;
  color-scheme: light;  /* ensure light form controls */
}

/* App + default text */
html, body, .stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4, h5, h6, p, span, div, label,
.stMarkdown, .stText, .stSubheader, .stTitle, .stHeader, .stTable, .stCaption {
  color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--card) !important;
  color: var(--text) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* File uploader dropzone */
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 1px dashed var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] * {
  color: var(--text) !important;
}
div[data-testid="stFileUploader"] button {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border-radius: 10px !important;
  border: none !important;
}

/* Cards/blocks/KPIs */
.block, .kpi {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
}
.kpi b, .kpi span { color: var(--text) !important; }
.small { color: #6B7280 !important; font-size: 12px; }

/* Inputs & labels */
label, .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio > label, .stDateInput label {
  color: var(--text) !important;
}
details > summary { color: var(--text) !important; }

/* ---------- SELECTBOX / MULTISELECT FIX ---------- */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] * ,
.stMultiSelect [data-baseweb="select"] * {
  color: var(--text) !important;
  background: transparent !important;
}
[data-baseweb="popover"] {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul[role="listbox"] {
  background: var(--card) !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li[role="option"],
[data-baseweb="popover"] li {
  background: var(--card) !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] li[role="option"]:hover,
[data-baseweb="popover"] li:hover {
  background: var(--accent) !important;
  color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] li[aria-selected="true"] {
  background: var(--accent) !important;
  color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border-radius: 10px;
  border: none;
}
            
.stDownloadButton > button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] > a {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border-radius: 10px !important;
  border: none !important;
  padding: 0.6rem 1rem !important;
}
.stDownloadButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stDownloadButton"] > a:hover {
  filter: brightness(0.95);
}
.stDownloadButton > button:active,
div[data-testid="stDownloadButton"] > button:active,
div[data-testid="stDownloadButton"] > a:active {
  filter: brightness(0.9);
}
div[data-testid="stDownloadButton"] * {
  color: #FFFFFF !important;
}

/* Dataframe text & headers */
div[data-testid="stDataFrame"] * { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# Plotly defaults
px.defaults.template = "plotly_white"
pio.templates.default = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.D3

def style_fig(fig):
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
        title_font=dict(color="#111827"),
        legend_title_font=dict(color="#111827"),
        legend_font=dict(color="#111827"),
        xaxis=dict(
            title_font=dict(color="#111827"),
            tickfont=dict(color="#111827"),
            gridcolor="#E5E7EB",
            zerolinecolor="#E5E7EB"
        ),
        yaxis=dict(
            title_font=dict(color="#111827"),
            tickfont=dict(color="#111827"),
            gridcolor="#E5E7EB",
            zerolinecolor="#E5E7EB"
        ),
        margin=dict(l=30, r=20, t=60, b=40)
    )
    return fig

# ---------------------------
# Helpers
# ---------------------------
REQUIRED = ["ProductId", "Margin", "MarginPercent", "PriceChangePercent", "MarginChangePercent"]

@st.cache_data
def load_csv(f):
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.abs().max(skipna=True) > 1.5:
        s = s / 100.0
    return s

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_MarginChangePct"] = normalize_pct(out["MarginChangePercent"])
    out["_PriceChangePct"]  = normalize_pct(out["PriceChangePercent"])
    out["_MarginPct"]       = normalize_pct(out["MarginPercent"])
    out["Margin_After"]  = pd.to_numeric(out["Margin"], errors="coerce")
    out["Margin_Before"] = out["Margin_After"] * (1.0 - out["_MarginChangePct"])
    out["Margin_Δ"]      = out["Margin_After"] - out["Margin_Before"]
    out["Is_Increase"]   = out["Margin_Δ"] > 0
    out["MarginChangePercent_pct"] = out["_MarginChangePct"] * 100.0
    out["PriceChangePercent_pct"]  = out["_PriceChangePct"] * 100.0
    out["MarginPercent_pct"]       = out["_MarginPct"] * 100.0
    return out

def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = (~values.isna()) & (~weights.isna())
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=np.clip(weights[mask], 0, None))

def topn_control(label: str, available: int, default: int | None = None, *, key: str) -> int:
    """Safe slider for 'Top N' – avoids min==max errors."""
    available = int(max(0, available))
    if available <= 1:
        st.caption(f"{label}: " + ("no items" if available == 0 else "showing 1 item"))
        return available
    if default is None:
        default = min(20, available)
    default = int(min(max(1, default), available))
    return st.slider(label, min_value=1, max_value=available, value=default, key=key)

def build_filters_ui(df: pd.DataFrame):
    st.sidebar.header("🔎 Filters")
    st.sidebar.caption("Pick columns to filter. Numeric → range; categorical → multiselect.")

    drop_zero = st.sidebar.checkbox(
        "Drop 0% margin changes",
        value=False,
        help="Exclude rows where MarginChangePercent is exactly 0%."
    )

    cols = df.columns.tolist()
    exclude = {
        "_MarginChangePct","_PriceChangePct","_MarginPct",
        "Margin_After","Margin_Before","Margin_Δ","Is_Increase",
        "MarginChangePercent_pct","PriceChangePercent_pct","MarginPercent_pct"
    }
    filterable = [c for c in cols if c not in exclude]

    current = df.copy()
    if drop_zero:
        current = current[~np.isclose(current["_MarginChangePct"].fillna(0.0), 0.0)]

    picked = st.sidebar.multiselect("Columns to filter", options=sorted(filterable))
    for col in picked:
        s = current[col]
        if pd.api.types.is_numeric_dtype(s):
            mn = float(np.nanmin(s)); mx = float(np.nanmax(s))
            if not np.isfinite(mn) or not np.isfinite(mx):
                continue
            if mn == mx:
                st.sidebar.caption(f"{col}: single value ({mn})")
                a, b = mn, mx
            else:
                a, b = st.sidebar.slider(f"{col} range", min_value=float(mn), max_value=float(mx),
                                         value=(float(mn), float(mx)))
            current = current[(current[col] >= a) & (current[col] <= b)]
        else:
            vals = sorted(s.dropna().astype(str).unique())
            chosen = st.sidebar.multiselect(f"{col} values", options=vals)
            if chosen:
                current = current[current[col].astype(str).isin(chosen)]

    st.sidebar.caption(f"Filtered rows: **{len(current):,}** / {len(df):,}")
    return current

# ---------------------------
# App UI
# ---------------------------
st.title("📈 Margin Change Analyzer")

with st.sidebar:
    st.header("📁 Upload CSV")
    f = st.file_uploader("Choose a CSV file", type=["csv"])
    st.caption("Required columns: ProductId, Margin, MarginPercent, PriceChangePercent, MarginChangePercent")

if not f:
    st.info("Upload a CSV to begin.")
    st.stop()

raw = load_csv(f)
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"Your file is missing required columns: {', '.join(missing)}")
    st.stop()

df = add_derived(raw)
filtered = build_filters_ui(df)

# ---------------------------
# KPI row
# ---------------------------
simple_avg = filtered["MarginChangePercent_pct"].mean()
weighted_avg = weighted_mean(filtered["MarginChangePercent_pct"], filtered["Margin_After"])
increase_share = (filtered["Is_Increase"].mean() * 100.0) if len(filtered) else np.nan
median_change = filtered["MarginChangePercent_pct"].median()
count_products = len(filtered)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"<div class='kpi'><b>Avg Δ Margin %</b><br><span style='font-size:24px'>{simple_avg:,.2f}%</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><b>Weighted Avg Δ %</b><br><span style='font-size:24px'>{weighted_avg:,.2f}%</span><div class='small'>Weights = Margin_After</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi'><b>↑ Margin Share</b><br><span style='font-size:24px'>{increase_share:,.2f}%</span></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi'><b>Median Δ %</b><br><span style='font-size:24px'>{median_change:,.2f}%</span></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='kpi'><b>Products</b><br><span style='font-size:24px'>{int(count_products):,}</span></div>", unsafe_allow_html=True)

# ---------------------------
# Grouped analysis
# ---------------------------
st.subheader("Grouped Averages")

all_cols = [c for c in filtered.columns if c not in {
    "_MarginChangePct","_PriceChangePct","_MarginPct",
    "Margin_After","Margin_Before","Margin_Δ","Is_Increase",
    "MarginChangePercent_pct","PriceChangePercent_pct","MarginPercent_pct"
}]

grp_cols = st.multiselect(
    "Group by (choose one or more columns from your file)",
    options=sorted(all_cols)
)

if grp_cols:
    def wavg(group_s, w_series):
        return weighted_mean(group_s, w_series.loc[group_s.index])

    g = (
        filtered
        .groupby(grp_cols, dropna=False)
        .agg(
            Count=("ProductId","count"),
            Avg_MarginChangePct=("MarginChangePercent_pct","mean"),
            WAvg_MarginChangePct=("MarginChangePercent_pct", lambda s: wavg(s, filtered["Margin_After"])),
            Avg_PriceChangePct=("PriceChangePercent_pct","mean"),
            Avg_MarginPercent=("MarginPercent_pct","mean"),
            Sum_Margin_After=("Margin_After","sum"),
            Sum_Margin_Before=("Margin_Before","sum"),
        )
        .reset_index()
    )

    g = g.sort_values(by="WAvg_MarginChangePct", ascending=False)
    st.dataframe(g.round(2), use_container_width=True)

    # ── Focus value (only when a single group column is selected) ──
    if len(grp_cols) == 1 and not g.empty:
        group_col = grp_cols[0]
        unique_vals = sorted(g[group_col].astype(str).unique())
        focus_val = st.selectbox(
            f"Focus on a specific {group_col} (optional)",
            options=["(All)"] + unique_vals,
            index=0,
            key="focus_value"
        )
        if focus_val != "(All)":
            g_view = g[g[group_col].astype(str) == focus_val].copy()
            # products inside the focused group value
            products = filtered[filtered[group_col].astype(str) == focus_val].copy()
        else:
            g_view = g.copy()
            products = None
    else:
        g_view = g.copy()
        products = None

    # ── If focused: switch visuals to ProductId level ──
    if products is not None:
        st.markdown(f"### Product-level view for {group_col} = **{focus_val}**")

        # BAR (ProductId) – choose Δ Margin % or Δ Price %
        metric_map_prod = {
            "Δ Margin %": "MarginChangePercent_pct",
            "Δ Price %":  "PriceChangePercent_pct",
        }
        chosen_label_prod = st.selectbox(
            "Product bar metric",
            list(metric_map_prod.keys()),
            index=0,
            key="prod_bar_metric"
        )
        metric_col_prod = metric_map_prod[chosen_label_prod]

        prod_bar_candidates = (
            products[["ProductId", metric_col_prod]]
            .dropna(subset=[metric_col_prod])
            .copy()
        )
        # order by absolute change to surface largest movers
        prod_bar_candidates["abs_val"] = prod_bar_candidates[metric_col_prod].abs()
        top_n = topn_control(
            "How many products to show (bar)?",
            available=len(prod_bar_candidates),
            default=min(30, len(prod_bar_candidates)),
            key="prod_bar_topn"
        )
        plot_prod = prod_bar_candidates.sort_values("abs_val", ascending=False).head(top_n)

        if plot_prod.empty:
            st.info("No products to plot for the selected focus.")
        else:
            fig_prod_bar = px.bar(
                plot_prod,
                x="ProductId",
                y=metric_col_prod,
                title=f"{chosen_label_prod} by ProductId",
                labels={metric_col_prod: chosen_label_prod, "ProductId": "ProductId"}
            )
            st.plotly_chart(style_fig(fig_prod_bar), use_container_width=True)

        # DIVERGING (butterfly-style) per ProductId using Δ Margin %
        st.markdown("#### Increases vs Decreases (Product diverging bars)")
        if not products.empty:
            prod_div = products[["ProductId", "MarginChangePercent_pct"]].dropna().copy()
            prod_div["Direction"] = np.where(prod_div["MarginChangePercent_pct"] >= 0, "Increase", "Decrease")
            # pick top by absolute change to keep it readable
            prod_div["abs_val"] = prod_div["MarginChangePercent_pct"].abs()
            top_n_div = topn_control(
                "How many products to show (diverging)?",
                available=len(prod_div),
                default=min(40, len(prod_div)),
                key="prod_div_topn"
            )
            prod_div = prod_div.sort_values("abs_val", ascending=False).head(top_n_div)

            if prod_div.empty:
                st.info("No product changes to show.")
            else:
                maxshare = float(np.ceil(max(1.0, prod_div["abs_val"].max()) / 10.0) * 10.0)
                fig_div = px.bar(
                    prod_div,
                    x="MarginChangePercent_pct",
                    y="ProductId",
                    color="Direction",
                    orientation="h",
                    hover_data={"ProductId": True, "MarginChangePercent_pct": ":.2f", "Direction": True},
                    title="Δ Margin % by ProductId (left=decrease, right=increase)",
                    labels={"MarginChangePercent_pct": "Δ Margin %", "ProductId": "ProductId"}
                )
                fig_div.update_xaxes(range=[-maxshare, maxshare], tickformat=".0f")
                fig_div.update_layout(bargap=0.2)
                fig_div.for_each_trace(
                    lambda t: t.update(marker_color="#10B981") if t.name == "Increase" else t.update(marker_color="#EF4444")
                )
                st.plotly_chart(style_fig(fig_div), use_container_width=True)

        # SLOPEGRAPH per ProductId: Before → After (€)
        if not products.empty:
            st.markdown("#### Before → After Margin (€) by ProductId")
            prod2 = products[["ProductId", "Margin_Before", "Margin_After"]].dropna().copy()
            # top by After (or use absolute delta)
            prod2["Δ_Margin"] = prod2["Margin_After"] - prod2["Margin_Before"]
            top_n_slope = topn_control(
                "How many products in slopegraph?",
                available=len(prod2),
                default=min(20, len(prod2)),
                key="prod_slope_topn"
            )
            top_prod = prod2.sort_values("Margin_After", ascending=False).head(top_n_slope)

            if not top_prod.empty:
                slope_df = pd.melt(
                    top_prod[["ProductId", "Margin_Before", "Margin_After"]],
                    id_vars="ProductId", var_name="Period", value_name="Margin (€)"
                ).replace({"Period": {"Margin_Before": "Before", "Margin_After": "After"}})
                slope_df["Period"] = pd.Categorical(slope_df["Period"], ["Before", "After"], ordered=True)

                fig_prod_slope = px.line(
                    slope_df,
                    x="Period", y="Margin (€)",
                    color="ProductId",
                    markers=True,
                    title="Before → After Margin (€)"
                )
                fig_prod_slope.update_traces(mode="lines+markers", line=dict(width=2), opacity=0.9)
                fig_prod_slope.update_xaxes(showgrid=False)
                fig_prod_slope.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#E5E7EB")
                st.plotly_chart(style_fig(fig_prod_slope), use_container_width=True)

    # ── If NOT focused: keep the original group-level visuals ──
    if products is None:
        if len(grp_cols) == 1 and not g_view.empty:
            group_col = grp_cols[0]
            metric_map = {
                "Weighted Avg Δ Margin %": "WAvg_MarginChangePct",
                "Avg Δ Margin %":          "Avg_MarginChangePct",
                "Avg Δ Price %":           "Avg_PriceChangePct",
            }
            chosen_label = st.selectbox(
                "Bar chart metric",
                list(metric_map.keys()),
                index=0,
                key="bar_metric_select"
            )
            metric_col = metric_map[chosen_label]

            bar_candidates = g_view[[group_col, metric_col]].dropna(subset=[metric_col])
            top_n = topn_control(
                "How many groups to show (bar)?",
                available=len(bar_candidates),
                default=min(25, len(bar_candidates)),
                key="bar_topn"
            )
            plot_df = bar_candidates.sort_values(metric_col, ascending=False).head(top_n)

            if plot_df.empty:
                st.info("No data to plot for the selected metric/filters.")
            else:
                figg = px.bar(
                    plot_df,
                    x=group_col,
                    y=metric_col,
                    title=f"{chosen_label} by {group_col}",
                    labels={metric_col: chosen_label, group_col: group_col}
                )
                st.plotly_chart(style_fig(figg), use_container_width=True)

        # Group-level butterfly (increase vs decrease share by group label)
        st.markdown("#### Increase vs Decrease Share (Butterfly)")
        work = filtered.copy()
        if len(grp_cols) == 1:
            label_col = grp_cols[0]
            work[label_col] = work[label_col].astype(str)
        else:
            label_col = "__GroupLabel__"
            work[label_col] = work[grp_cols].astype(str).apply(
                lambda r: " | ".join(f"{c}={v}" for c, v in zip(grp_cols, r)),
                axis=1
            )

        inc = (work.groupby(label_col, dropna=False)["Is_Increase"]
               .agg(IncShare=lambda s: 100.0 * s.mean(), Count="count")
               .reset_index())
        inc["DecShare"] = 100.0 - inc["IncShare"]

        topn = topn_control(
            "How many groups to show (butterfly)?",
            available=len(inc),
            default=min(20, len(inc)),
            key="butterfly_topn"
        )
        inc = inc.sort_values("Count", ascending=False).head(topn)

        long = inc.melt(
            id_vars=[label_col, "Count"],
            value_vars=["IncShare", "DecShare"],
            var_name="Direction", value_name="Share"
        )
        if long.empty:
            st.info("No groups to show for the butterfly chart with the current filters.")
        else:
            long["SignedShare"] = np.where(long["Direction"] == "IncShare", long["Share"], -long["Share"])
            maxshare = float(np.ceil(max(1.0, long["Share"].max()) / 10.0) * 10.0)

            fig_butter = px.bar(
                long, x="SignedShare", y=label_col, color="Direction", orientation="h",
                hover_data={label_col: True, "Share": ":.1f", "Count": True, "SignedShare": False},
                title="Increase vs Decrease Share by Group",
                labels={"SignedShare": "Share of products (%)", label_col: "Group"}
            )
            fig_butter.update_xaxes(range=[-maxshare, maxshare], tickformat=".0f")
            fig_butter.update_yaxes(
                categoryorder="array",
                categoryarray=list(inc.sort_values("Count", ascending=False)[label_col])
            )
            fig_butter.update_layout(bargap=0.2)
            fig_butter.for_each_trace(
                lambda t: t.update(marker_color="#10B981") if t.name == "IncShare" else t.update(marker_color="#EF4444")
            )
            st.plotly_chart(style_fig(fig_butter), use_container_width=True)

        # Group-level slopegraph (single group column)
        if len(grp_cols) == 1 and not g_view.empty:
            group_col = grp_cols[0]
            g2 = g_view.copy()
            g2["Δ_Margin"]  = g2["Sum_Margin_After"] - g2["Sum_Margin_Before"]
            g2["%Δ_Margin"] = np.where(
                g2["Sum_Margin_Before"] != 0,
                100 * g2["Δ_Margin"] / g2["Sum_Margin_Before"],
                np.nan
            )

            top_n = topn_control(
                "How many groups in slopegraph?",
                available=len(g2),
                default=min(12, len(g2)),
                key="slope_topn"
            )
            top_groups = g2.sort_values("Sum_Margin_After", ascending=False).head(top_n)

            if top_groups.empty:
                st.info("No groups to show for the slopegraph with the current filters.")
            else:
                slope_df = pd.melt(
                    top_groups[[group_col, "Sum_Margin_Before", "Sum_Margin_After"]],
                    id_vars=group_col,
                    var_name="Period",
                    value_name="Margin (€)"
                ).replace({"Period": {"Sum_Margin_Before": "Before", "Sum_Margin_After": "After"}})
                slope_df["Period"] = pd.Categorical(slope_df["Period"], ["Before", "After"], ordered=True)

                fig_slope = px.line(
                    slope_df,
                    x="Period", y="Margin (€)",
                    color=group_col,
                    markers=True,
                    title=f"Before → After Margin (€) by {group_col}"
                )
                fig_slope.update_traces(mode="lines+markers", line=dict(width=2), opacity=0.9)
                pct_map = top_groups.set_index(group_col)["%Δ_Margin"].to_dict()
                for name, sub in slope_df.groupby(group_col):
                    y_after = sub.loc[sub["Period"] == "After", "Margin (€)"].values[0]
                    fig_slope.add_annotation(
                        x="After", y=y_after,
                        xanchor="left", yanchor="middle",
                        text=f"{pct_map.get(name, np.nan):+.1f}%",
                        showarrow=False, font=dict(size=10, color="#111827"),
                        align="left"
                    )
                fig_slope.update_xaxes(showgrid=False)
                fig_slope.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#E5E7EB")
                st.plotly_chart(style_fig(fig_slope), use_container_width=True)

    # Download grouped results
    csv = g.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download grouped results (CSV)", data=csv,
                       file_name="grouped_margin_change.csv", mime="text/csv")
else:
    st.info("Select one or more columns to group by and see segmented averages.")
