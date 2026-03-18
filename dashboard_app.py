import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Macro Intelligence Platform", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "daily_news_sentiment.csv"
REPORT_FILE = BASE_DIR / "daily_report.txt"
PRICE_REPORT_FILE = BASE_DIR / "market_sentiment_price_report.txt"
EMERGING_FILE = BASE_DIR / "emerging_themes_latest.json"
THEME_STRENGTH_V2_FILE = BASE_DIR / "theme_strength_latest.json"
NARRATIVE_V2_FILE = BASE_DIR / "narrative_latest.json"
EXPOSURE_FILE = BASE_DIR / "exposure_map_latest.json"
TRADE_IDEAS_V2_FILE = BASE_DIR / "trade_ideas_v2_latest.json"
ALERTS_V2_FILE = BASE_DIR / "alerts_v2_latest.json"
BACKTEST_FILE = BASE_DIR / "backtest_latest.json"

THEME_KEYWORDS = {
    "Geopolitics": ["war", "iran", "china", "russia", "conflict", "sanctions"],
    "Inflation": ["inflation", "cpi", "ppi", "price pressures"],
    "Energy Supply": ["oil", "gas", "opec", "crude"],
    "AI Boom": ["ai", "chip", "nvidia", "semiconductor"],
    "Economic Slowdown": ["recession", "slowdown", "job losses"],
}

ASSET_WEIGHTS = {
    "Oil": 1.4,
    "Gold": 1.2,
    "SP500": 1.5,
    "Nasdaq": 1.4,
    "USD": 1.3,
    "US10Y": 1.3,
    "Natural Gas": 1.2,
    "Nvidia": 1.5,
}


def parse_sentiment_text(sentiment_text: str):
    items = []
    if not isinstance(sentiment_text, str) or not sentiment_text.strip():
        return items
    for part in sentiment_text.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        asset, score = part.split(":", 1)
        try:
            items.append((asset.strip(), float(score.strip())))
        except ValueError:
            continue
    return items


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_news_data():
    if not CSV_FILE.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return df, pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["categories", "assets", "sentiment", "reasoning", "analysis_source"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    rows = []
    for _, row in df.iterrows():
        parsed = parse_sentiment_text(row["sentiment"])
        for asset, score in parsed:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "title": row["title"],
                    "categories": row["categories"],
                    "assets": row["assets"],
                    "asset": asset,
                    "sentiment_score": score,
                    "analysis_source": row.get("analysis_source", ""),
                }
            )
    sentiment_df = pd.DataFrame(rows)
    return df, sentiment_df


def summarize_categories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["category", "count"])
    categories = df["categories"].str.split(",").explode().dropna().str.strip()
    categories = categories[categories != ""]
    out = categories.value_counts().reset_index()
    out.columns = ["category", "count"]
    return out


def average_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df.empty:
        return pd.DataFrame(columns=["asset", "avg_sentiment"])
    out = (
        sentiment_df.groupby("asset", as_index=False)["sentiment_score"]
        .mean()
        .sort_values("sentiment_score", ascending=False)
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )
    return out


def detect_themes(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["theme", "count"])
    counts = defaultdict(int)
    for title in news_df["title"].astype(str):
        text = title.lower()
        for theme, words in THEME_KEYWORDS.items():
            if any(word in text for word in words):
                counts[theme] += 1
    if not counts:
        return pd.DataFrame(columns=["theme", "count"])
    return pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["theme", "count"])


def calculate_theme_strength(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["theme", "strength"])

    theme_strength = defaultdict(float)
    for _, row in news_df.iterrows():
        title = str(row["title"]).lower()
        sentiment_pairs = parse_sentiment_text(row.get("sentiment", ""))
        sentiment_intensity = sum(abs(score) for _, score in sentiment_pairs)
        asset_weight = 1.0
        for asset, _ in sentiment_pairs:
            asset_weight += ASSET_WEIGHTS.get(asset, 1.0)
        for theme, keywords in THEME_KEYWORDS.items():
            if any(k in title for k in keywords):
                theme_strength[theme] += sentiment_intensity * asset_weight

    if not theme_strength:
        return pd.DataFrame(columns=["theme", "strength"])
    return pd.DataFrame(sorted(theme_strength.items(), key=lambda x: x[1], reverse=True), columns=["theme", "strength"])


def build_narrative(news_df: pd.DataFrame):
    if news_df.empty:
        return ["No dominant macro narrative detected today."]

    titles = news_df["title"].astype(str).tolist()
    lines = []
    if any("oil" in t.lower() or "gas" in t.lower() for t in titles):
        lines.append("Energy markets are receiving significant attention due to oil and gas related developments.")
    if any("inflation" in t.lower() or "cpi" in t.lower() or "ppi" in t.lower() for t in titles):
        lines.append("Inflation concerns remain present in the macro narrative.")
    if any("ai" in t.lower() or "chip" in t.lower() or "nvidia" in t.lower() for t in titles):
        lines.append("AI and semiconductor news continues to influence equity sentiment.")
    if any("war" in t.lower() or "iran" in t.lower() or "sanctions" in t.lower() for t in titles):
        lines.append("Geopolitical risk remains an important market driver across commodities and risk assets.")
    if not lines:
        lines.append("No dominant macro narrative detected today.")
    return lines


def load_text_file(path: Path) -> str:
    if not path.exists():
        return "File not found. Run the pipeline first."
    return path.read_text(encoding="utf-8")


def format_signal_badge(signal: str) -> str:
    signal = (signal or "").lower()
    if signal == "strong":
        return "🔴 Strong"
    if signal == "medium":
        return "🟠 Medium"
    if signal == "weak":
        return "🟡 Weak"
    return "⚪ Unknown"


def format_regime_badge(regime: str) -> str:
    regime = (regime or "").lower()
    if regime == "risk_on":
        return "🟢 Risk-On"
    if regime == "risk_off":
        return "🔴 Risk-Off"
    if regime == "mixed":
        return "🟠 Mixed"
    return "⚪ Neutral"


def format_exposure_badge(view: str) -> str:
    view = (view or "").lower()
    if view == "strong_bullish":
        return "🟢 Strong Bullish"
    if view == "bullish":
        return "🟩 Bullish"
    if view == "strong_bearish":
        return "🔴 Strong Bearish"
    if view == "bearish":
        return "🟥 Bearish"
    return "⚪ Neutral"


def format_trade_badge(side: str, conviction: str) -> str:
    side = (side or "").upper()
    conviction = (conviction or "").lower()
    if side == "LONG" and conviction == "high":
        return "🟢 High Conviction Long"
    if side == "LONG":
        return "🟩 Long"
    if side == "SHORT" and conviction == "high":
        return "🔴 High Conviction Short"
    if side == "SHORT":
        return "🟥 Short"
    return "⚪ Watch"


def format_alert_badge(severity: str, alert_type: str) -> str:
    severity = (severity or "").lower()
    alert_type = (alert_type or "").lower()
    if severity == "high" and alert_type == "trade":
        return "🚨 High Trade Alert"
    if severity == "high" and alert_type == "theme":
        return "🔥 High Theme Alert"
    if severity == "high" and alert_type == "exposure":
        return "⚠️ High Exposure Alert"
    if severity == "medium":
        return "🟠 Monitor"
    return "⚪ Info"


news_df, sentiment_df = load_news_data()
category_df = summarize_categories(news_df)
avg_sent_df = average_sentiment(sentiment_df)
emerging_json = load_json(EMERGING_FILE, {})
theme_strength_v2_json = load_json(THEME_STRENGTH_V2_FILE, {})
narrative_v2_json = load_json(NARRATIVE_V2_FILE, {})
exposure_json = load_json(EXPOSURE_FILE, {})
trade_ideas_json = load_json(TRADE_IDEAS_V2_FILE, {})
alerts_v2_json = load_json(ALERTS_V2_FILE, {})
backtest_json = load_json(BACKTEST_FILE, {})

theme_df = detect_themes(news_df)
theme_strength_df = calculate_theme_strength(news_df)
narrative_lines = build_narrative(news_df)

st.title("AI Macro Intelligence Platform")
st.caption("Macro radar, smart alerts, emerging themes, signal strength, exposure mapping, trade ideas, backtest validation, AI narrative, and market intelligence dashboard")

if news_df.empty:
    st.error("daily_news_sentiment.csv not found or empty. Run run_daily_agent.py first.")
    st.stop()

latest_time = news_df["timestamp"].max()
scored_themes = theme_strength_v2_json.get("scored_themes", [])
emerging_themes = emerging_json.get("themes", [])
ai_narrative = narrative_v2_json.get("narrative", {})
theme_exposures = exposure_json.get("theme_exposures", [])
aggregate_exposures = exposure_json.get("aggregate_asset_exposures", [])
trade_ideas = trade_ideas_json.get("trade_ideas", [])
smart_alerts = alerts_v2_json.get("alerts", [])
backtest_summary = backtest_json.get("summary", {})
backtest_results = backtest_json.get("results", [])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total news", len(news_df))
col2.metric("Known themes", len(theme_df))
col3.metric("Emerging themes", len(emerging_themes))
col4.metric("Latest run", latest_time.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_time) else "N/A")

with st.sidebar:
    st.header("Filters")
    category_options = sorted(category_df["category"].tolist()) if not category_df.empty else []
    selected_categories = st.multiselect("Category", category_options)

    asset_options = sorted(avg_sent_df["asset"].tolist()) if not avg_sent_df.empty else []
    selected_assets = st.multiselect("Asset", asset_options)

    source_options = sorted([s for s in news_df["analysis_source"].dropna().unique().tolist() if str(s).strip()])
    selected_sources = st.multiselect("Analysis source", source_options)

    keyword = st.text_input("Keyword in title")

filtered_news = news_df.copy()
if selected_categories:
    filtered_news = filtered_news[
        filtered_news["categories"].apply(
            lambda x: any(cat in [i.strip() for i in str(x).split(",")] for cat in selected_categories)
        )
    ]
if selected_assets:
    filtered_news = filtered_news[
        filtered_news["assets"].apply(
            lambda x: any(asset in [i.strip() for i in str(x).split(",")] for asset in selected_assets)
        )
    ]
if selected_sources:
    filtered_news = filtered_news[filtered_news["analysis_source"].isin(selected_sources)]
if keyword.strip():
    filtered_news = filtered_news[filtered_news["title"].str.contains(keyword, case=False, na=False)]

filtered_sentiment = sentiment_df.copy()
if not filtered_news.empty:
    filtered_titles = set(filtered_news["title"].tolist())
    filtered_sentiment = filtered_sentiment[filtered_sentiment["title"].isin(filtered_titles)]
else:
    filtered_sentiment = filtered_sentiment.iloc[0:0]

filtered_avg_sent = average_sentiment(filtered_sentiment)
filtered_category_df = summarize_categories(filtered_news)
filtered_theme_df = detect_themes(filtered_news)
filtered_theme_strength_df = calculate_theme_strength(filtered_news)
filtered_narrative = build_narrative(filtered_news)

st.subheader("Smart Alerts")
if not smart_alerts:
    st.info("No smart alerts found yet. Run alert_engine_v2.py.")
else:
    alert_left, alert_right = st.columns([1.1, 1.3])
    with alert_left:
        st.markdown("**Alert overview**")
        alert_df = pd.DataFrame(smart_alerts)
        type_counts = alert_df["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        st.bar_chart(type_counts.set_index("type")["count"])
        display_df = alert_df[["type", "severity", "bucket", "priority_score", "title"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with alert_right:
        st.markdown("**Actionable alert cards**")
        for item in smart_alerts[:6]:
            st.markdown(
                f"**{item.get('title', 'Alert')}**  \n"
                f"Badge: {format_alert_badge(item.get('severity', 'medium'), item.get('type', 'theme'))}  \n"
                f"Bucket: {item.get('bucket', 'other')}  \n"
                f"Priority: {item.get('priority_score', 0):.2f}"
            )
            st.caption(item.get("message", ""))
            st.divider()

st.subheader("Macro Radar")
radar_left, radar_right = st.columns(2)
with radar_left:
    st.markdown("**Known themes today**")
    if filtered_theme_df.empty:
        st.info("No themes detected for current filters.")
    else:
        st.bar_chart(filtered_theme_df.set_index("theme")["count"])
        st.dataframe(filtered_theme_df, use_container_width=True, hide_index=True)

with radar_right:
    st.markdown("**Known theme strength**")
    if filtered_theme_strength_df.empty:
        st.info("No known theme strength available for current filters.")
    else:
        st.bar_chart(filtered_theme_strength_df.set_index("theme")["strength"])
        st.dataframe(filtered_theme_strength_df, use_container_width=True, hide_index=True)

st.subheader("Emerging Theme Radar")
if not scored_themes:
    st.info("No emerging theme strength file found yet. Run emerging_theme_detector.py and theme_strength_v2.py.")
else:
    emerg_df = pd.DataFrame(scored_themes)
    top_emerg_left, top_emerg_right = st.columns([1.2, 1])

    with top_emerg_left:
        chart_df = emerg_df[["theme", "strength_score"]].copy()
        st.bar_chart(chart_df.set_index("theme")["strength_score"])
        display_df = emerg_df[["theme", "strength_score", "signal_level", "momentum", "market_regime"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with top_emerg_right:
        st.markdown("**Signal cards**")
        for item in scored_themes[:5]:
            st.markdown(
                f"**{item.get('theme', 'Unknown')}**  \n"
                f"Score: {item.get('strength_score', 0):.2f}  \n"
                f"Signal: {format_signal_badge(item.get('signal_level', 'unknown'))}  \n"
                f"Momentum: {item.get('momentum', 'developing')}  \n"
                f"Regime: {format_regime_badge(item.get('market_regime', 'neutral'))}"
            )
            st.caption(item.get("description", ""))
            st.divider()

st.subheader("Theme → Asset Exposure Map")
if not theme_exposures and not aggregate_exposures:
    st.info("No exposure map found yet. Run exposure_engine_v1.py.")
else:
    exp_left, exp_right = st.columns([1.1, 1.3])

    with exp_left:
        st.markdown("**Aggregate asset exposures**")
        if aggregate_exposures:
            agg_df = pd.DataFrame(aggregate_exposures)
            chart_df = agg_df[["asset", "aggregate_score"]].copy()
            st.bar_chart(chart_df.set_index("asset")["aggregate_score"])
            display_df = agg_df.copy()
            display_df["view_badge"] = display_df["view"].apply(format_exposure_badge)
            st.dataframe(display_df[["asset", "aggregate_score", "view_badge"]], use_container_width=True, hide_index=True)
        else:
            st.info("No aggregate exposures available.")

    with exp_right:
        st.markdown("**Theme drilldown**")
        for block in theme_exposures[:5]:
            st.markdown(f"**{block.get('theme', 'Unknown Theme')}**")
            st.caption(
                f"Strength: {block.get('strength_score', 0):.2f} | "
                f"Signal: {block.get('signal_level', 'unknown')} | "
                f"Momentum: {block.get('momentum', 'developing')} | "
                f"Regime: {block.get('market_regime', 'neutral')}"
            )
            exposures_df = pd.DataFrame(block.get("exposures", []))
            if not exposures_df.empty:
                exposures_df["view_badge"] = exposures_df["view"].apply(format_exposure_badge)
                st.dataframe(exposures_df[["asset", "final_score", "view_badge"]], use_container_width=True, hide_index=True)
            st.divider()

st.subheader("Top Trade Ideas")
if not trade_ideas:
    st.info("No trade ideas found yet. Run trade_ideas_engine_v2.py.")
else:
    idea_left, idea_right = st.columns([1.1, 1.2])

    with idea_left:
        st.markdown("**Trade idea ranking**")
        ideas_df = pd.DataFrame(trade_ideas)
        chart_df = ideas_df[["asset", "score"]].copy()
        st.bar_chart(chart_df.set_index("asset")["score"])
        ideas_df["trade_badge"] = ideas_df.apply(lambda row: format_trade_badge(row.get("trade_side", "WATCH"), row.get("conviction", "low")), axis=1)
        st.dataframe(ideas_df[["asset", "trade_side", "conviction", "score", "trade_badge"]], use_container_width=True, hide_index=True)

    with idea_right:
        st.markdown("**Action cards**")
        for item in trade_ideas[:5]:
            st.markdown(
                f"**{item.get('trade_side', 'WATCH')} {item.get('asset', 'Unknown')}**  \n"
                f"Score: {item.get('score', 0):+.2f}  \n"
                f"Conviction: {item.get('conviction', 'low')}  \n"
                f"Signal: {format_trade_badge(item.get('trade_side', 'WATCH'), item.get('conviction', 'low'))}"
            )
            st.caption(item.get("reason", ""))
            st.divider()

st.subheader("Backtest / Hit-Rate")
if not backtest_summary:
    st.info("No backtest results found yet. Run backtest_engine_v1.py.")
else:
    bt1, bt2, bt3, bt4 = st.columns(4)
    bt1.metric("Total trades", backtest_summary.get("total_trades", 0))
    bt2.metric("Wins", backtest_summary.get("wins", 0))
    bt3.metric("Losses", backtest_summary.get("losses", 0))
    bt4.metric("Hit rate", f"{backtest_summary.get('hit_rate', 0)}%")

    back_left, back_right = st.columns([1.1, 1.4])
    with back_left:
        st.markdown("**Performance summary**")
        avg_return = backtest_summary.get("average_return_pct", 0)
        st.metric("Average return", f"{avg_return}%")
        perf_df = pd.DataFrame([
            {"metric": "Total trades", "value": backtest_summary.get("total_trades", 0)},
            {"metric": "Wins", "value": backtest_summary.get("wins", 0)},
            {"metric": "Losses", "value": backtest_summary.get("losses", 0)},
            {"metric": "Hit rate %", "value": backtest_summary.get("hit_rate", 0)},
            {"metric": "Average return %", "value": backtest_summary.get("average_return_pct", 0)},
        ])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with back_right:
        st.markdown("**Recent trade results**")
        if backtest_results:
            bt_df = pd.DataFrame(backtest_results)
            keep_cols = [c for c in ["date", "asset", "side", "status", "market_move_pct", "strategy_return_pct"] if c in bt_df.columns]
            st.dataframe(bt_df[keep_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No backtest trade rows available.")

st.subheader("AI Market Briefing")
if ai_narrative:
    top_left, top_right = st.columns([1.5, 1])
    with top_left:
        st.markdown(f"### {ai_narrative.get('headline', 'Daily Macro Briefing')}")
        st.markdown(f"**Regime:** {format_regime_badge(ai_narrative.get('market_regime', 'neutral'))}")
        st.write(ai_narrative.get("summary", ""))

        st.markdown("**Key drivers**")
        for driver in ai_narrative.get("key_drivers", []):
            st.write(f"- {driver}")

        st.markdown("**Watch items**")
        for item in ai_narrative.get("watch_items", []):
            st.write(f"- {item}")

    with top_right:
        st.markdown("**Asset implications**")
        asset_implications = ai_narrative.get("asset_implications", [])
        if asset_implications:
            implication_df = pd.DataFrame(asset_implications)
            st.dataframe(implication_df, use_container_width=True, hide_index=True)
        else:
            st.info("No asset implications found.")
else:
    st.markdown("**Fallback narrative**")
    for line in filtered_narrative:
        st.write(f"- {line}")

st.subheader("Asset Sentiment Overview")
over_left, over_right = st.columns(2)
with over_left:
    st.markdown("**Average sentiment by asset**")
    if filtered_avg_sent.empty:
        st.info("No sentiment data for current filters.")
    else:
        st.bar_chart(filtered_avg_sent.set_index("asset")["avg_sentiment"])
        st.dataframe(filtered_avg_sent, use_container_width=True, hide_index=True)
with over_right:
    st.markdown("**News count by category**")
    if filtered_category_df.empty:
        st.info("No category data for current filters.")
    else:
        st.bar_chart(filtered_category_df.set_index("category")["count"])
        st.dataframe(filtered_category_df, use_container_width=True, hide_index=True)

st.subheader("News Feed")
news_columns = ["timestamp", "title", "categories", "assets", "sentiment", "analysis_source", "reasoning"]
news_display = filtered_news[[c for c in news_columns if c in filtered_news.columns]].copy()
news_display = news_display.sort_values("timestamp", ascending=False)
st.dataframe(news_display, use_container_width=True, hide_index=True)

st.subheader("Generated Files")
rep1, rep2, rep3, rep4 = st.columns(4)
with rep1:
    st.markdown("**daily_report.txt**")
    st.text(load_text_file(REPORT_FILE))
with rep2:
    st.markdown("**market_sentiment_price_report.txt**")
    st.text(load_text_file(PRICE_REPORT_FILE))
with rep3:
    st.markdown("**narrative_latest.txt**")
    st.text(load_text_file(BASE_DIR / "narrative_latest.txt"))
with rep4:
    st.markdown("**trade_ideas_v2_latest.txt**")
    st.text(load_text_file(BASE_DIR / "trade_ideas_v2_latest.txt"))
