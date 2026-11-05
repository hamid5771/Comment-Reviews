import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROCESSED_PATH = Path("data/processed/analyzed_reviews.jsonl")

st.set_page_config(
    page_title="Premier Inn Evesham | Operational Intelligence",
    page_icon="🌙",
    layout="wide",
)

# Dark-mode & Light-mode safe styling for KPI cards and summary boxes
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #1E1E2E !important;
        border: 1px solid #3B3B54 !important;
        border-top: 4px solid #8B5CF6 !important;
        padding: 14px 18px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    .amz-box {
        background-color: #1E1E2E;
        border: 1px solid #3B3B54;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
        color: #F1F5F9;
    }
    .amz-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #FFFFFF;
    }
    .amz-body {
        font-size: 14px;
        line-height: 1.6;
        color: #CBD5E1;
        margin-bottom: 14px;
    }
    .amz-quote {
        font-size: 13px;
        font-style: italic;
        color: #94A3B8;
        border-left: 3px solid #8B5CF6;
        padding-left: 10px;
        margin-top: 10px;
    }
    .tag-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .tag-pos {
        background-color: rgba(22, 163, 74, 0.2);
        border: 1px solid #16A34A;
        color: #4ADE80;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .tag-neg {
        background-color: rgba(220, 38, 38, 0.2);
        border: 1px solid #DC2626;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .tag-neu {
        background-color: rgba(217, 119, 6, 0.2);
        border: 1px solid #D97706;
        color: #FBBF24;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return pd.DataFrame()

    records = []
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["month_year"] = df["published_date"].dt.strftime("%Y-%m")
    df["hotel_vs_rest_gap"] = df["hotel_score"] - df["restaurant_score"]
    return df


def build_amazon_style_summaries(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Branch-agnostic Amazon-style 'Customers Say' synthesis for HOTEL and RESTAURANT.
    Focuses strictly on universal operational sub-pillars (zero staff names or site-specific quirks).
    """
    hotel_themes = {
        "Reception & Check-In": (r"\b(staff|reception|receptionist|check.?in|welcome|welcoming|friendly|helpful|polite|efficient|team)\b", []),
        "Room & Bed Comfort": (r"\b(room|rooms|bed|beds|mattress|comfortable|comfy|spacious|quiet|sleep|pillows)\b", []),
        "Cleanliness & Housekeeping": (r"\b(clean|cleanliness|spotless|tidy|dirty|dusty|hoovered|smelt|smell|stained|housekeeping)\b", []),
        "Bathroom & Shower": (r"\b(shower|bathroom|ensuite|toilet|water|pressure|bath|towels)\b", []),
        "Temperature & AC": (r"\b(air.?con|ac|heating|temperature|cold|hot|stuffy|fan|ventilation|window)\b", []),
    }

    rest_themes = {
        "Breakfast Quality": (r"\b(breakfast|buffet|cooked|bacon|eggs|sausage|pastry|pastries|continental|toast|cereal|coffee|tea|juice)\b", []),
        "Dinner & Bar": (r"\b(dinner|evening meal|eatery|pub|starter|starters|main|mains|menu|drink|drinks|beer|bar)\b", []),
        "Wait Times & Staffing": (r"\b(wait|waited|waiting|slow|understaffed|staffing|rushed|queue|shambles|missing|lost|order|service)\b", []),
        "Food Quality & Portions": (r"\b(portion|portions|available|availability|run out|ran out|fresh|hot|cold|tasty|delicious|awful|inedible|value)\b", []),
    }

    for _, row in df.iterrows():
        text = str(row.get("raw_text", ""))
        h_score = row.get("hotel_score")
        r_score = row.get("restaurant_score")

        if pd.notna(h_score):
            for name, (pat, scores) in hotel_themes.items():
                if re.search(pat, text, re.I):
                    scores.append(float(h_score))

        if pd.notna(r_score):
            for name, (pat, scores) in rest_themes.items():
                if re.search(pat, text, re.I):
                    scores.append(float(r_score))

    def format_theme_stats(theme_dict):
        stats = []
        for name, (_, scores) in theme_dict.items():
            if not scores:
                continue
            stats.append({"name": name, "count": len(scores), "avg": sum(scores) / len(scores)})
        return sorted(stats, key=lambda x: x["count"], reverse=True)

    h_stats = format_theme_stats(hotel_themes)
    r_stats = format_theme_stats(rest_themes)

    # 1. Build Universal Hotel Narrative
    h_avg_overall = df["hotel_score"].mean()
    h_pos = [t["name"] for t in h_stats if t["avg"] >= 3.6]
    h_neg = [t["name"] for t in h_stats if t["avg"] < 3.2]

    hotel_narrative = (
        f"Customers rate core hotel operations at <b>{h_avg_overall:.2f} ★</b> across {len(df)} analyzed stays. "
    )
    if h_pos:
        hotel_narrative += f"Guests consistently praise <b>{', '.join(h_pos[:3])}</b>. "
    if h_neg:
        hotel_narrative += f"Areas requiring operational focus include <b>{', '.join(h_neg)}</b>."
    elif not h_pos:
        hotel_narrative += "Feedback across core accommodation themes remains mixed."

    best_h_rows = df[df["hotel_score"] >= 4.5].sort_values(by="credibility_weight", ascending=False)
    hotel_quote = best_h_rows.iloc[0]["raw_text"][:220] + "..." if not best_h_rows.empty else ""

    # 2. Build Universal Restaurant Narrative
    rest_df = df[df["restaurant_score"].notna()]
    if not rest_df.empty:
        r_avg_overall = rest_df["restaurant_score"].mean()
        r_pos = [t["name"] for t in r_stats if t["avg"] >= 3.6]
        r_neg = [t["name"] for t in r_stats if t["avg"] < 3.2]
        gap = h_avg_overall - r_avg_overall
        gap_str = (
            f"trailing hotel performance by <b>{gap:.2f} ★</b>"
            if gap > 0.15
            else f"outperforming hotel operations by <b>{abs(gap):.2f} ★</b>"
            if gap < -0.15
            else "tracking in line with hotel performance"
        )

        rest_narrative = (
            f"Across {len(rest_df)} reviews mentioning dining, food and beverage scores <b>{r_avg_overall:.2f} ★</b> ({gap_str}). "
        )
        if r_pos:
            rest_narrative += f"Diners respond positively to <b>{', '.join(r_pos[:2])}</b>. "
        if r_neg:
            rest_narrative += f"Key guest pain points center on <b>{', '.join(r_neg[:3])}</b>."

        worst_r_rows = rest_df.sort_values(by="hotel_vs_rest_gap", ascending=False)
        rest_quote = worst_r_rows.iloc[0]["raw_text"][:220] + "..." if not worst_r_rows.empty else ""
    else:
        rest_narrative = "No restaurant or breakfast mentions found in the currently selected filter window."
        rest_quote = ""

    return (
        {"narrative": hotel_narrative, "tags": h_stats, "quote": hotel_quote},
        {"narrative": rest_narrative, "tags": r_stats, "quote": rest_quote},
    )

df_raw = load_data()

if df_raw.empty:
    st.error("No processed data found at `data/processed/analyzed_reviews.jsonl`. Run `processor.py` first!")
    st.stop()

# ==========================================
# SIDEBAR SLICERS (EXCEL-STYLE FILTERS)
# ==========================================
st.sidebar.title("🌙 Whitbread Ops Slicers")
st.sidebar.caption("Premier Inn Evesham — Aspect & Credibility MVP")

window_options = {
    "Last 30 Days": 30,
    "Last 90 Days (Quarter)": 90,
    "Last 180 Days (6 Months)": 180,
    "Last 365 Days (Full Year)": 365,
    "All Available Data": 9999,
}
selected_window_label = st.sidebar.selectbox("📅 Date Range Window", list(window_options.keys()), index=3)
days_cutoff = window_options[selected_window_label]

sources = sorted(df_raw["source"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect("🌐 Review Platforms", sources, default=sources)

exclude_dups = st.sidebar.checkbox("Exclude Cross-Platform Duplicates", value=True)
use_credibility_weights = st.sidebar.checkbox("Apply Credibility Weighting (W_i)", value=True)
only_mixed_reviews = st.sidebar.checkbox("Show Only Reviews Mentioning F&B", value=False)

min_star, max_star = st.sidebar.slider("⭐ Filter by Raw Star Rating", 1.0, 5.0, (1.0, 5.0), step=1.0)

df = df_raw.copy()
if days_cutoff < 9999:
    max_dt = df["published_date"].max()
    if pd.notna(max_dt):
        cutoff_dt = max_dt - timedelta(days=days_cutoff)
        df = df[df["published_date"] >= cutoff_dt]

df = df[df["source"].isin(selected_sources)]
df = df[(df["raw_rating"] >= min_star) & (df["raw_rating"] <= max_star)]

if exclude_dups and "is_cross_platform_duplicate" in df.columns:
    df = df[~df["is_cross_platform_duplicate"]]

if only_mixed_reviews:
    df = df[df["restaurant_score"].notna()]

if df.empty:
    st.warning("No reviews match the current sidebar filters.")
    st.stop()

# ==========================================
# TOP EXECUTIVE KPI HEADER
# ==========================================
st.title("🌙 Premier Inn Evesham \nOperational Sentiment Dashboard")
st.markdown(
    f"**Showing {len(df)} unique stays** | Window: `{selected_window_label}` | "
    f"Sources: `{', '.join(selected_sources)}` | "
    f"Weighting Mode: `{'Credibility-Weighted (W_i)' if use_credibility_weights else 'Unweighted Raw'}`"
)

raw_mean = df["raw_rating"].mean()
w_sum = df["credibility_weight"].sum()
weighted_mean = (df["raw_rating"] * df["credibility_weight"]).sum() / w_sum if w_sum > 0 else raw_mean
rating_delta = weighted_mean - raw_mean

hotel_df = df[df["hotel_score"].notna()]
rest_df = df[df["restaurant_score"].notna()]

if use_credibility_weights:
    hotel_kpi = (hotel_df["hotel_score"] * hotel_df["credibility_weight"]).sum() / hotel_df["credibility_weight"].sum()
    rest_kpi = (
        (rest_df["restaurant_score"] * rest_df["credibility_weight"]).sum() / rest_df["credibility_weight"].sum()
        if not rest_df.empty
        else 0.0
    )
else:
    hotel_kpi = hotel_df["hotel_score"].mean()
    rest_kpi = rest_df["restaurant_score"].mean() if not rest_df.empty else 0.0

rage_posts_count = len(df[(df["author_contributions"] <= 1) & (df["raw_rating"] <= 2.0)])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("1. Raw Public Rating", f"{raw_mean:.2f} ★", f"{len(df)} total reviews", delta_color="off")
col2.metric(
    "2. Credibility-Weighted",
    f"{weighted_mean:.2f} ★",
    f"{rating_delta:+.2f} ★ vs Raw",
    delta_color="normal",
)
col3.metric(
    "3. Core HOTEL Score",
    f"{hotel_kpi:.2f} ★",
    f"{hotel_kpi - raw_mean:+.2f} ★ vs Public Rating",
    delta_color="normal",
)
col4.metric(
    "4. RESTAURANT (F&B) Score",
    f"{rest_kpi:.2f} ★" if not rest_df.empty else "N/A",
    f"{rest_kpi - hotel_kpi:+.2f} ★ vs Hotel Ops" if not rest_df.empty else "No F&B reviews",
    delta_color="normal",
)
col5.metric(
    "5. 1-Review Rage Posts",
    f"{rage_posts_count} flagged",
    "Down-weighted to W_i = 0.10",
    delta_color="off",
)

st.divider()

# ==========================================
# ROW 2: AMAZON-STYLE "CUSTOMERS SAY" AI SUMMARIES
# ==========================================
st.subheader("✨ Guests Say — Aspect Executive Summaries")
hotel_sum, rest_sum = build_amazon_style_summaries(df)

sum_col1, sum_col2 = st.columns(2)

def render_tag_pills(tags: list[dict]) -> str:
    pills = []
    for t in tags:
        if t["avg"] >= 3.6:
            cls, icon = "tag-pos", "🟢"
        elif t["avg"] >= 2.8:
            cls, icon = "tag-neu", "🟡"
        else:
            cls, icon = "tag-neg", "🔴"
        pills.append(f'<span class="{cls}">{icon} {t["name"]} ({t["count"]}) • {t["avg"]:.1f}★</span>')
    return '<div class="tag-wrap">' + "".join(pills) + "</div>"

with sum_col1:
    st.markdown(
        f"""
        <div class="amz-box">
            <div class="amz-title">🏨 Customers say about HOTEL Operations</div>
            <div class="amz-body">{hotel_sum['narrative']}</div>
            {render_tag_pills(hotel_sum['tags'])}
            <div class="amz-quote">"{hotel_sum['quote']}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sum_col2:
    st.markdown(
        f"""
        <div class="amz-box">
            <div class="amz-title">🍽️ Customers say about RESTAURANT & F&B</div>
            <div class="amz-body">{rest_sum['narrative']}</div>
            {render_tag_pills(rest_sum['tags'])}
            <div class="amz-quote">"{rest_sum['quote']}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ==========================================
# ROW 3: CHARTS (TREND LINE + QUADRANT SCATTER)
# ==========================================
chart_col1, chart_col2 = st.columns([1.35, 1.0])

with chart_col1:
    st.subheader("📈 Monthly Trend: Core Hotel vs. Restaurant (F&B) Performance")
    monthly_records = []
    for month, grp in df.groupby("month_year"):
        if month == "NaT" or pd.isna(month):
            continue
        if use_credibility_weights:
            h_val = (grp["hotel_score"] * grp["credibility_weight"]).sum() / grp["credibility_weight"].sum()
            r_grp = grp[grp["restaurant_score"].notna()]
            r_val = (
                (r_grp["restaurant_score"] * r_grp["credibility_weight"]).sum() / r_grp["credibility_weight"].sum()
                if not r_grp.empty
                else None
            )
            pub_val = (grp["raw_rating"] * grp["credibility_weight"]).sum() / grp["credibility_weight"].sum()
        else:
            h_val = grp["hotel_score"].mean()
            r_val = grp["restaurant_score"].mean()
            pub_val = grp["raw_rating"].mean()

        monthly_records.append(
            {
                "Month": str(month),
                "Core Hotel Score": round(h_val, 2),
                "Restaurant (F&B) Score": round(r_val, 2) if pd.notna(r_val) else None,
                "Public Star Rating": round(pub_val, 2),
                "Review Count": len(grp),
            }
        )

    df_monthly = pd.DataFrame(monthly_records).sort_values("Month")

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["Month"],
            y=df_monthly["Core Hotel Score"],
            mode="lines+markers",
            name="Core Hotel Ops (DeBERTa)",
            line=dict(color="#16A34A", width=3),
            marker=dict(size=8),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["Month"],
            y=df_monthly["Public Star Rating"],
            mode="lines+markers",
            name="Submitted Public Star Rating",
            line=dict(color="#94A3B8", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["Month"],
            y=df_monthly["Restaurant (F&B) Score"],
            mode="lines+markers",
            name="Restaurant / Breakfast (DeBERTa)",
            line=dict(color="#DC2626", width=3),
            marker=dict(size=8),
            connectgaps=True,
        )
    )
    fig_trend.update_layout(
        yaxis=dict(range=[0.8, 5.2], title="Star Score (1.0 - 5.0 ★)"),
        xaxis=dict(title="Month", type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.subheader("🎯 Decoupled Quadrant: Hotel vs. Restaurant")
    if not rest_df.empty:
        fig_scatter = px.scatter(
            rest_df,
            x="hotel_score",
            y="restaurant_score",
            color="source",
            size="credibility_weight",
            hover_name="author",
            hover_data=["raw_rating", "author_contributions", "published_date"],
            color_discrete_map={"Google": "#3B82F6", "TripAdvisor": "#10B981"},
            labels={
                "hotel_score": "Core Hotel Score (★)",
                "restaurant_score": "Restaurant / F&B Score (★)",
                "credibility_weight": "Weight (W_i)",
            },
        )
        fig_scatter.add_hline(y=3.0, line_dash="dash", line_color="#9CA3AF")
        fig_scatter.add_vline(x=3.0, line_dash="dash", line_color="#9CA3AF")
        fig_scatter.add_annotation(
            x=4.4,
            y=1.25,
            text="⚠️ Great Hotel / Bad F&B",
            showarrow=False,
            font=dict(color="#F87171", size=12),
        )
        fig_scatter.update_layout(
            xaxis=dict(range=[0.8, 5.2]),
            yaxis=dict(range=[0.8, 5.2]),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No reviews with F&B mentions in the selected filter window.")

st.divider()

# ==========================================
# ROW 4: EXCEL-STYLE INTERACTIVE DATA GRID & EXPORT
# ==========================================
st.subheader("📊 Interactive Review Spreadsheet & Aspect Inspector")
st.caption("Click any column header to sort (e.g., sort by `Gap (Hotel - F&B)` to see where F&B dragged down a great hotel stay).")

grid_cols = [
    "published_date",
    "source",
    "author",
    "author_contributions",
    "credibility_weight",
    "raw_rating",
    "hotel_score",
    "restaurant_score",
    "hotel_vs_rest_gap",
    "trip_type",
    "raw_text",
]

df_grid = df[grid_cols].sort_values(by="published_date", ascending=False).copy()
df_grid["published_date"] = df_grid["published_date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    df_grid,
    use_container_width=True,
    height=420,
    hide_index=True,
    column_config={
        "published_date": st.column_config.TextColumn("Date", width="small"),
        "source": st.column_config.TextColumn("Platform", width="small"),
        "author": st.column_config.TextColumn("Guest", width="medium"),
        "author_contributions": st.column_config.NumberColumn("Contribs (c_i)", format="%d"),
        "credibility_weight": st.column_config.NumberColumn("Weight (W_i)", format="%.2f"),
        "raw_rating": st.column_config.NumberColumn("Public ★", format="%.1f ★"),
        "hotel_score": st.column_config.ProgressColumn(
            "HOTEL Score",
            help="DeBERTa-v3 ABSA Hotel Score (1-5)",
            format="%.2f ★",
            min_value=1.0,
            max_value=5.0,
        ),
        "restaurant_score": st.column_config.ProgressColumn(
            "RESTAURANT Score",
            help="DeBERTa-v3 ABSA Restaurant Score (1-5)",
            format="%.2f ★",
            min_value=1.0,
            max_value=5.0,
        ),
        "hotel_vs_rest_gap": st.column_config.NumberColumn(
            "Gap (Hotel - F&B)",
            help="Positive means Hotel outperformed Restaurant",
            format="%+.2f ★",
        ),
        "trip_type": st.column_config.TextColumn("Trip Type", width="small"),
        "raw_text": st.column_config.TextColumn("Full Review Comment", width="large"),
    },
)

csv_bytes = df_grid.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Export Filtered Table to Excel (CSV)",
    data=csv_bytes,
    file_name=f"premier_inn_evesham_analyzed_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)