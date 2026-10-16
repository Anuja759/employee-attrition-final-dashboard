import json, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from pathlib import Path
from features import engineer

A = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Attrition Risk | Palo Alto Networks", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv(A/"scored_employees.csv"), pd.read_csv(A/"global_importance.csv"),
            pd.read_csv(A/"model_metrics.csv", index_col=0))
@st.cache_resource
def model(): return joblib.load(A/"final_model.joblib")

df, imp, metrics = load()
COL = {"Low":"#2e9e6b","Medium":"#e0a030","High":"#d64545"}
cat = lambda p: "Low" if p < lo else ("Medium" if p < hi else "High")

st.sidebar.title("Filters")
lo, hi = st.sidebar.slider("Risk thresholds (Low < a ≤ Medium < b ≤ High)", 0.0, 1.0, (0.30, 0.60), 0.05)
depts = st.sidebar.multiselect("Department", sorted(df.Department.unique()), sorted(df.Department.unique()))
roles = st.sidebar.multiselect("Job role", sorted(df.JobRole.unique()), sorted(df.JobRole.unique()))
view = df[df.Department.isin(depts) & df.JobRole.isin(roles)].copy()
view["Risk"] = view.AttritionProb.map(cat)
page = st.sidebar.radio("Module", ["Attrition Risk Dashboard","Employee Risk Profile","Department & Role View","Explainability & What-If"])
st.sidebar.caption("Scores are out-of-fold calibrated probabilities from the selected model, so no employee is scored by a model that saw their own outcome.")

st.title("Employee Attrition Risk — Palo Alto Networks")

if page == "Attrition Risk Dashboard":
    c = st.columns(4)
    c[0].metric("Employees in view", f"{len(view):,}")
    c[1].metric("High risk", int((view.Risk=="High").sum()))
    c[2].metric("Medium risk", int((view.Risk=="Medium").sum()))
    c[3].metric("Mean attrition probability", f"{view.AttritionProb.mean():.1%}")
    l, r = st.columns(2)
    cnt = view.Risk.value_counts().reindex(["Low","Medium","High"]).fillna(0).reset_index()
    cnt.columns = ["Risk","Employees"]
    l.plotly_chart(px.bar(cnt, x="Risk", y="Employees", color="Risk", color_discrete_map=COL, title="Risk distribution"), width="stretch")
    r.plotly_chart(px.histogram(view, x="AttritionProb", nbins=40, title="Distribution of attrition probability").add_vline(x=lo).add_vline(x=hi), width="stretch")
    st.subheader("Highest-risk employees")
    st.dataframe(view.sort_values("AttritionProb", ascending=False)[["EmployeeID","Department","JobRole","JobLevel","OverTime","MonthlyIncome","AttritionProb","Risk","TopDrivers"]].head(25)
                 .style.format({"AttritionProb":"{:.1%}"}), width="stretch", hide_index=True)
    st.download_button("Download filtered scores (CSV)", view.to_csv(index=False), "attrition_scores.csv")

elif page == "Employee Risk Profile":
    eid = st.selectbox("Employee ID", view.EmployeeID.tolist() or df.EmployeeID.tolist())
    e = df[df.EmployeeID==eid].iloc[0]; p = float(e.AttritionProb); rk = cat(p)
    c = st.columns(3)
    c[0].metric("Attrition probability", f"{p:.1%}"); c[1].metric("Risk category", rk)
    c[2].metric("Actual outcome (historical)", "Left" if e.Attrition==1 else "Stayed")
    st.progress(min(p,1.0))
    st.markdown(f"**{e.JobRole}**, {e.Department} · Level {e.JobLevel} · {e.YearsAtCompany} yrs at company · ${e.MonthlyIncome:,.0f}/month · OverTime: {e.OverTime}")
    st.markdown("**Key contributing factors (SHAP-based reason codes):** " + (e.TopDrivers if isinstance(e.TopDrivers,str) else "none material"))
    st.caption("Reason codes show which factors push this employee's risk up the most; they indicate association, not proven cause.")
    peers = df[df.JobRole==e.JobRole]
    st.write(f"Peer comparison ({e.JobRole}, n={len(peers)}): engagement {e.EngagementScore:.2f} vs peer mean {peers.EngagementScore.mean():.2f}; "
             f"income ${e.MonthlyIncome:,.0f} vs ${peers.MonthlyIncome.mean():,.0f}; years since promotion {e.YearsSinceLastPromotion} vs {peers.YearsSinceLastPromotion.mean():.1f}.")

elif page == "Department & Role View":
    g = view.groupby("Department").agg(Employees=("EmployeeID","size"), MeanRisk=("AttritionProb","mean"),
            HighRisk=("Risk", lambda s:(s=="High").sum())).reset_index()
    st.plotly_chart(px.bar(g, x="Department", y="MeanRisk", text=g.MeanRisk.map("{:.1%}".format), title="Mean attrition probability by department"), width="stretch")
    gr = view.groupby(["Department","JobRole"]).agg(Employees=("EmployeeID","size"), MeanRisk=("AttritionProb","mean"),
            HighRisk=("Risk", lambda s:(s=="High").sum())).reset_index().sort_values("MeanRisk")
    st.plotly_chart(px.bar(gr, x="MeanRisk", y="JobRole", color="Department", orientation="h", title="Mean attrition probability by role"), width="stretch")
    st.dataframe(gr.sort_values("MeanRisk", ascending=False).style.format({"MeanRisk":"{:.1%}"}), width="stretch", hide_index=True)

else:
    t1, t2 = st.tabs(["Feature importance","What-if scenario"])
    with t1:
        top = imp.head(15).iloc[::-1]
        st.plotly_chart(px.bar(top, x="mean_abs_shap", y="feature", orientation="h", title="Global drivers (mean |SHAP|, XGBoost explainer)"), width="stretch")
        st.caption("Dummy-encoded categories appear separately (e.g. OverTime_Yes / OverTime_No); read them together.")
        st.subheader("Model comparison (20% held-out test set)")
        st.dataframe(metrics.style.format("{:.3f}"), width="stretch")
    with t2:
        eid = st.selectbox("Employee ID", df.EmployeeID.tolist(), key="wi")
        base = df[df.EmployeeID==eid].iloc[0]
        raw_cols = [c for c in pd.read_csv(A.parent/"Palo_Alto_Networks.csv", nrows=1).columns if c!="Attrition"]
        r = base[raw_cols].to_frame().T.copy()
        for c in r.columns:
            if c not in ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]: r[c]=pd.to_numeric(r[c])
        a, b, c3 = st.columns(3)
        r["OverTime"] = a.selectbox("OverTime", ["Yes","No"], index=0 if base.OverTime=="Yes" else 1)
        r["MonthlyIncome"] = a.number_input("Monthly income", 1000, 30000, int(base.MonthlyIncome), 250)
        r["JobSatisfaction"] = b.slider("Job satisfaction", 1, 4, int(base.JobSatisfaction))
        r["WorkLifeBalance"] = b.slider("Work-life balance", 1, 4, int(base.WorkLifeBalance))
        r["EnvironmentSatisfaction"] = b.slider("Environment satisfaction", 1, 4, int(base.EnvironmentSatisfaction))
        r["StockOptionLevel"] = c3.slider("Stock option level", 0, 3, int(base.StockOptionLevel))
        r["YearsSinceLastPromotion"] = c3.slider("Years since last promotion", 0, 15, int(base.YearsSinceLastPromotion))
        r["BusinessTravel"] = c3.selectbox("Business travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"],
                                           index=["Non-Travel","Travel_Rarely","Travel_Frequently"].index(base.BusinessTravel))
        new = float(model().predict_proba(engineer(r))[0,1])
        old = float(model().predict_proba(engineer(base[raw_cols].to_frame().T.assign(**{c:pd.to_numeric(base[c]) for c in raw_cols if c not in ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]})))[0,1])
        m = st.columns(3)
        m[0].metric("Current (refit model) probability", f"{old:.1%}"); m[1].metric("Scenario probability", f"{new:.1%}", f"{new-old:+.1%}", delta_color="inverse")
        m[2].metric("Scenario category", cat(new))
        st.caption("What-if shows model-estimated association. It is a prioritisation aid, not a guarantee that an intervention will change behaviour.")
