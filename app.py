import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import norm

st.set_page_config(page_title="LifeStat", page_icon="📊", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "lifestyle_data.csv"

COLUMNS = [
    "date", "sleep_hours", "exercise_minutes", "water_litres",
    "screen_hours", "study_hours", "energy_rating", "mood", "sleep_quality"
]

def sample_data(n=75, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n)
    sleep = np.clip(rng.normal(7.1, 1.0, n), 4, 10)
    exercise = np.clip(rng.normal(35, 18, n), 0, 100)
    water = np.clip(rng.normal(2.2, 0.55, n), 0.7, 4.5)
    screen = np.clip(rng.normal(5.0, 1.5, n), 1, 11)
    study = np.clip(rng.normal(4.0, 1.4, n), 0, 9)
    energy = np.clip(np.rint(3 + 0.55 * sleep + 0.025 * exercise - 0.18 * screen + rng.normal(0, 1.2, n)), 1, 10).astype(int)
    moods = np.where(energy >= 7, "Good", np.where(energy <= 4, "Low", "Okay"))
    quality = np.where(sleep >= 7.5, "Good", np.where(sleep >= 6, "Average", "Poor"))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sleep_hours": np.round(sleep, 1),
        "exercise_minutes": np.round(exercise, 0).astype(int),
        "water_litres": np.round(water, 1),
        "screen_hours": np.round(screen, 1),
        "study_hours": np.round(study, 1),
        "energy_rating": energy,
        "mood": moods,
        "sleep_quality": quality
    })[COLUMNS]

def clean_data(df):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = ["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours", "energy_rating"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"] + numeric)
    df = df[(df.sleep_hours >= 0) & (df.sleep_hours <= 24)]
    df = df[(df.exercise_minutes >= 0) & (df.exercise_minutes <= 1440)]
    df = df[(df.water_litres >= 0) & (df.water_litres <= 20)]
    df = df[(df.screen_hours >= 0) & (df.screen_hours <= 24)]
    df = df[(df.study_hours >= 0) & (df.study_hours <= 24)]
    df = df[df.energy_rating.between(1, 10)]
    df["energy_rating"] = df["energy_rating"].round().astype(int)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["mood"] = df["mood"].fillna("Not specified").astype(str)
    df["sleep_quality"] = df["sleep_quality"].fillna("Not specified").astype(str)
    return df.sort_values("date").reset_index(drop=True)

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def load_data():
    if DATA_FILE.exists():
        try:
            return clean_data(pd.read_csv(DATA_FILE))
        except Exception:
            pass
    df = sample_data()
    save_data(df)
    return df

if "data" not in st.session_state:
    st.session_state.data = load_data()

st.title("📊 LifeStat – Personal Lifestyle Statistical Analyzer")
st.caption("A Probability and Statistics project using your daily lifestyle observations.")
st.sidebar.title("LifeStat Menu")
page = st.sidebar.radio("Navigate", [
    "Dashboard", "Data Collection", "Joint Probability",
    "Discrete Distribution", "Continuous Distribution",
    "Regression Lines", "Pearson Correlation", "Reports"
])
st.sidebar.caption("The included sample data is illustrative. Replace it with your own entries or CSV.")

df = st.session_state.data.copy()

def require_data():
    if len(df) < 2:
        st.warning("Please add at least two valid observations in Data Collection.")
        return False
    return True

def event_probability(condition):
    return float(condition.sum() / len(df))

if page == "Dashboard":
    st.subheader("Your lifestyle overview")
    if df.empty:
        st.info("No data available. Add records in Data Collection.")
    else:
        metrics = [
            ("Recorded days", f"{len(df)}"),
            ("Avg. sleep", f"{df.sleep_hours.mean():.1f} h"),
            ("Avg. exercise", f"{df.exercise_minutes.mean():.0f} min"),
            ("Avg. water", f"{df.water_litres.mean():.1f} L"),
            ("Avg. screen time", f"{df.screen_hours.mean():.1f} h"),
            ("Avg. energy", f"{df.energy_rating.mean():.1f}/10")
        ]
        cols = st.columns(3)
        for i, (label, value) in enumerate(metrics):
            cols[i % 3].metric(label, value)
        st.markdown("#### Recent observations")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
        st.markdown("#### Lifestyle trends")
        metric = st.selectbox("Choose a measure", ["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours", "energy_rating"])
        chart = px.line(df, x="date", y=metric, markers=True, title=f"{metric.replace('_', ' ').title()} over time")
        st.plotly_chart(chart, use_container_width=True)

elif page == "Data Collection":
    st.subheader("Add or upload your lifestyle data")
    st.write("Enter one day's values manually, or upload a CSV containing the required columns.")
    with st.form("daily_entry", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Date", value=pd.Timestamp.today().date())
        sleep = c1.number_input("Sleep duration (hours)", 0.0, 24.0, 7.0, 0.1)
        exercise = c2.number_input("Exercise duration (minutes)", 0, 1440, 30)
        water = c2.number_input("Water intake (litres)", 0.0, 20.0, 2.0, 0.1)
        screen = c3.number_input("Screen time (hours)", 0.0, 24.0, 4.0, 0.1)
        study = c3.number_input("Study/work hours", 0.0, 24.0, 4.0, 0.1)
        energy = st.slider("Energy rating (1 = very low, 10 = very high)", 1, 10, 6)
        mood = st.selectbox("Mood", ["Good", "Okay", "Low", "Not specified"])
        quality = st.selectbox("Sleep quality", ["Good", "Average", "Poor", "Not specified"])
        submitted = st.form_submit_button("Save daily entry")
    if submitted:
        new_row = pd.DataFrame([{
            "date": str(date), "sleep_hours": sleep, "exercise_minutes": exercise,
            "water_litres": water, "screen_hours": screen, "study_hours": study,
            "energy_rating": energy, "mood": mood, "sleep_quality": quality
        }])
        st.session_state.data = clean_data(pd.concat([st.session_state.data, new_row], ignore_index=True))
        save_data(st.session_state.data)
        st.success("Daily entry saved.")

    st.markdown("#### Upload CSV")
    st.caption("Required headers: " + ", ".join(COLUMNS))
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    col_a, col_b = st.columns(2)
    if uploaded is not None:
        try:
            incoming = clean_data(pd.read_csv(uploaded))
            st.write(f"Valid rows found: {len(incoming)}")
            mode = st.radio("How should uploaded rows be handled?", ["Add to existing data", "Replace existing data"], horizontal=True)
            if st.button("Import CSV"):
                combined = pd.concat([st.session_state.data, incoming], ignore_index=True) if mode == "Add to existing data" else incoming
                st.session_state.data = clean_data(combined)
                save_data(st.session_state.data)
                st.success(f"Imported {len(incoming)} valid rows.")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")

    st.markdown("#### Current dataset")
    st.dataframe(st.session_state.data, use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("Download current data as CSV", st.session_state.data.to_csv(index=False), "lifestat_data.csv", "text/csv")
    with col_b:
        if st.button("Restore sample dataset"):
            st.session_state.data = sample_data()
            save_data(st.session_state.data)
            st.success("Sample dataset restored.")
    if st.button("Delete all records", type="secondary"):
        st.session_state.data = pd.DataFrame(columns=COLUMNS)
        save_data(st.session_state.data)
        st.warning("All records deleted.")

elif page == "Joint Probability":
    st.subheader("1. Joint Probability Analyzer")
    st.write("Find the probability that two selected events occur together.")
    if require_data():
        a_label = st.selectbox("Event A", ["Sleep at least 7 hours", "Exercise at least 30 minutes", "Screen time at most 4 hours", "Energy rating at least 7"], key="ja")
        b_label = st.selectbox("Event B", ["Energy rating at least 7", "Sleep at least 7 hours", "Exercise at least 30 minutes", "Screen time at most 4 hours"], key="jb")
        conditions = {
            "Sleep at least 7 hours": df.sleep_hours >= 7,
            "Exercise at least 30 minutes": df.exercise_minutes >= 30,
            "Screen time at most 4 hours": df.screen_hours <= 4,
            "Energy rating at least 7": df.energy_rating >= 7
        }
        A, B = conditions[a_label], conditions[b_label]
        both = A & B
        n = len(df)
        p = both.sum() / n
        x1, x2, x3 = st.columns(3)
        x1.metric("P(A)", f"{A.mean():.3f}")
        x2.metric("P(B)", f"{B.mean():.3f}")
        x3.metric("P(A ∩ B)", f"{p:.3f}")
        st.latex(r"P(A\\cap B)=\\frac{n(A\\cap B)}{N}")
        st.write(f"Both events occurred on **{int(both.sum())} out of {n}** recorded days.")
        fig = go.Figure(data=[go.Bar(x=["A only", "B only", "Both", "Neither"],
             y=[int((A & ~B).sum()), int((B & ~A).sum()), int(both.sum()), int((~A & ~B).sum())])])
        fig.update_layout(title="Frequency of event combinations", yaxis_title="Number of days")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Interpretation: In this dataset, the estimated probability of both “{a_label}” and “{b_label}” occurring on the same day is {p:.1%}.")

elif page == "Discrete Distribution":
    st.subheader("2. Discrete Probability Distribution")
    st.write("A discrete variable takes separate, countable values. Here we use the daily energy rating.")
    if require_data():
        variable = st.selectbox("Choose a discrete variable", ["energy_rating"])
        values = df[variable].astype(int)
        freq = values.value_counts().sort_index()
        dist = pd.DataFrame({"Value (x)": freq.index, "Frequency": freq.values, "P(X = x)": freq.values / len(values)})
        mean = (dist["Value (x)"] * dist["P(X = x)"]).sum()
        variance = (((dist["Value (x)"] - mean) ** 2) * dist["P(X = x)"]).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected value E(X)", f"{mean:.3f}")
        c2.metric("Variance", f"{variance:.3f}")
        c3.metric("Standard deviation", f"{np.sqrt(variance):.3f}")
        st.latex(r"E(X)=\\sum xP(X=x)")
        st.latex(r"Var(X)=E(X^2)-[E(X)]^2")
        st.dataframe(dist, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(dist, x="Value (x)", y="P(X = x)", title="Probability Mass Function (PMF)"), use_container_width=True)
        st.caption("The probabilities in the PMF add up to 1, apart from possible display rounding.")

elif page == "Continuous Distribution":
    st.subheader("3. Continuous Probability Distribution")
    st.write("Explore a continuous measurement and estimate the probability that it falls within an interval.")
    if require_data():
        options = {"Sleep duration (hours)": "sleep_hours", "Exercise duration (minutes)": "exercise_minutes",
                   "Water intake (litres)": "water_litres", "Screen time (hours)": "screen_hours",
                   "Study/work hours (hours)": "study_hours"}
        label = st.selectbox("Choose a continuous variable", list(options.keys()))
        col = options[label]
        vals = df[col].dropna().astype(float)
        mu, sd = vals.mean(), vals.std(ddof=1)
        c1, c2 = st.columns(2)
        low = c1.number_input("Interval lower limit", value=float(vals.min()), format="%.2f")
        high = c2.number_input("Interval upper limit", value=float(vals.max()), format="%.2f")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{mu:.3f}")
        c2.metric("Variance", f"{vals.var(ddof=1):.3f}")
        c3.metric("Standard deviation", f"{sd:.3f}")
        if high < low:
            st.error("Upper limit must be greater than or equal to lower limit.")
        else:
            empirical = ((vals >= low) & (vals <= high)).mean()
            st.metric("Observed interval proportion", f"{empirical:.1%}")
            st.write(f"{int(((vals >= low) & (vals <= high)).sum())} of {len(vals)} observations fall in the selected interval.")
            if sd > 0:
                normal_est = norm.cdf(high, mu, sd) - norm.cdf(low, mu, sd)
                st.metric("Normal-model estimated interval probability", f"{normal_est:.1%}")
                st.caption("The normal-model value is an approximation based on a fitted normal curve. Use the observed proportion as a direct summary of this dataset; the normal model may not fit every lifestyle variable.")
            fig = px.histogram(vals, nbins=15, histnorm="probability density", title=f"Probability density view: {label.lower()}")
            if sd > 0:
                xx = np.linspace(vals.min(), vals.max(), 200)
                fig.add_scatter(x=xx, y=norm.pdf(xx, mu, sd), mode="lines", name="Fitted normal PDF")
            st.plotly_chart(fig, use_container_width=True)
            sorted_vals = np.sort(vals.to_numpy())
            empirical_cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            cdf_fig = go.Figure()
            cdf_fig.add_scatter(x=sorted_vals, y=empirical_cdf, mode="lines+markers", name="Empirical CDF")
            if sd > 0:
                xx = np.linspace(vals.min(), vals.max(), 200)
                cdf_fig.add_scatter(x=xx, y=norm.cdf(xx, mu, sd), mode="lines", name="Fitted normal CDF")
            cdf_fig.update_layout(title="Cumulative Distribution Function (CDF)", xaxis_title=label, yaxis_title="F(x)", yaxis_range=[0, 1])
            st.plotly_chart(cdf_fig, use_container_width=True)
            st.latex(r"P(a\\leq X\\leq b)=F(b)-F(a)")

elif page == "Regression Lines":
    st.subheader("4. Regression Line Analyzer")
    st.write("Study the linear relationship between two numerical lifestyle variables.")
    if require_data():
        numeric_cols = ["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours", "energy_rating"]
        x_name = st.selectbox("X variable (predictor)", numeric_cols, index=0)
        y_name = st.selectbox("Y variable (response)", numeric_cols, index=5)
        if x_name == y_name:
            st.warning("Choose two different variables.")
        else:
            x, y = df[x_name].astype(float), df[y_name].astype(float)
            xbar, ybar = x.mean(), y.mean()
            sxx = ((x - xbar) ** 2).sum()
            syy = ((y - ybar) ** 2).sum()
            sxy = ((x - xbar) * (y - ybar)).sum()
            if sxx == 0 or syy == 0:
                st.error("Regression cannot be calculated because one variable has no variation.")
            else:
                b_yx = sxy / sxx
                a_yx = ybar - b_yx * xbar
                b_xy = sxy / syy
                a_xy = xbar - b_xy * ybar
                c1, c2 = st.columns(2)
                c1.markdown("**Regression of Y on X**")
                c1.latex(r"Y-\\bar Y=b_{yx}(X-\\bar X)")
                c1.write(f"Y = {a_yx:.4f} + ({b_yx:.4f})X")
                c2.markdown("**Regression of X on Y**")
                c2.latex(r"X-\\bar X=b_{xy}(Y-\\bar Y)")
                c2.write(f"X = {a_xy:.4f} + ({b_xy:.4f})Y")
                r = sxy / np.sqrt(sxx * syy)
                st.metric("Pearson r (for context)", f"{r:.4f}")
                fig = px.scatter(x=x, y=y, labels={"x": x_name, "y": y_name}, title="Scatter plot with regression lines")
                xx = np.linspace(x.min(), x.max(), 100)
                fig.add_scatter(x=xx, y=a_yx + b_yx * xx, mode="lines", name="Y on X")
                # Convert the X-on-Y regression equation into y-as-a-function-of-x for plotting.
                if abs(b_xy) > 1e-12:
                    fig.add_scatter(x=xx, y=(xx - a_xy) / b_xy, mode="lines", name="X on Y (rearranged)")
                fig.update_layout(xaxis_title=x_name, yaxis_title=y_name)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("#### What-if estimate using Y on X")
                chosen_x = st.slider(f"Select a value for {x_name}", float(x.min()), float(x.max()), float(x.mean()))
                st.success(f"Estimated {y_name} = {a_yx + b_yx * chosen_x:.2f}")
                st.caption("This is a statistical estimate from the fitted line, not a guarantee. Association does not prove causation.")

elif page == "Pearson Correlation":
    st.subheader("5. Pearson Correlation Coefficient")
    st.write("Pearson's r measures the strength and direction of a linear relationship between two numerical variables.")
    if require_data():
        numeric_cols = ["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours", "energy_rating"]
        x_name = st.selectbox("First variable", numeric_cols, index=0, key="px")
        y_name = st.selectbox("Second variable", numeric_cols, index=5, key="py")
        if x_name == y_name:
            st.warning("Choose two different variables.")
        else:
            x, y = df[x_name].astype(float), df[y_name].astype(float)
            r = x.corr(y, method="pearson")
            st.metric("Pearson correlation coefficient (r)", f"{r:.4f}")
            if r >= 0.7:
                meaning = "strong positive linear correlation"
            elif r >= 0.3:
                meaning = "moderate positive linear correlation"
            elif r > 0:
                meaning = "weak positive linear correlation"
            elif r <= -0.7:
                meaning = "strong negative linear correlation"
            elif r <= -0.3:
                meaning = "moderate negative linear correlation"
            elif r < 0:
                meaning = "weak negative linear correlation"
            else:
                meaning = "approximately no linear correlation"
            st.info(f"Interpretation: The selected variables show {meaning}.")
            st.latex(r"r=\\frac{\\sum(x-\\bar{x})(y-\\bar{y})}{\\sqrt{\\sum(x-\\bar{x})^2\\sum(y-\\bar{y})^2}}")
            st.plotly_chart(px.scatter(x=x, y=y, trendline="ols", labels={"x": x_name, "y": y_name},
                                       title="Scatter plot and linear trend"), use_container_width=True)
            st.caption("Pearson correlation describes linear association only. It does not prove that one factor causes the other.")

elif page == "Reports":
    st.subheader("Reports and export")
    if df.empty:
        st.info("There are no records to report.")
    else:
        st.write("Summary statistics for the current dataset")
        summary = df[["sleep_hours", "exercise_minutes", "water_litres", "screen_hours", "study_hours", "energy_rating"]].describe().T
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download lifestyle data (CSV)", df.to_csv(index=False), "lifestat_data.csv", "text/csv")
        report = [
            "LIFESTAT – PERSONAL LIFESTYLE STATISTICAL ANALYZER",
            f"Records analysed: {len(df)}",
            f"Average sleep: {df.sleep_hours.mean():.2f} hours",
            f"Average exercise: {df.exercise_minutes.mean():.2f} minutes",
            f"Average water intake: {df.water_litres.mean():.2f} litres",
            f"Average screen time: {df.screen_hours.mean():.2f} hours",
            f"Average energy rating: {df.energy_rating.mean():.2f}/10",
            "",
            "Note: This is an educational statistical summary, not medical advice."
        ]
        st.download_button("Download simple text summary", "\n".join(report), "lifestat_summary.txt", "text/plain")

st.sidebar.divider()
st.sidebar.caption("Educational project • Probability and Statistics")
