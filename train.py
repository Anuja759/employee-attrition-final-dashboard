import json, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import *
from xgboost import XGBClassifier
from features import engineer, CAT

raw = pd.read_csv("Palo_Alto_Networks.csv")
raw.insert(0, "EmployeeID", np.arange(1, len(raw)+1))
df = engineer(raw)
y = df.Attrition.values
X = df.drop(columns=["Attrition","EmployeeID"])
NUM = [c for c in X.columns if c not in CAT]
pre = lambda: ColumnTransformer([("n", StandardScaler(), NUM),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
spw = (y==0).sum()/(y==1).sum()
models = {
 "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
 "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
 "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
 "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                          scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1),
}
mk = lambda m: Pipeline([("p", pre()), ("m", m)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
rows = {}
for name, m in models.items():
    cvauc = cross_val_predict(mk(m), Xtr, ytr, cv=skf, method="predict_proba")[:,1]
    p = mk(m).fit(Xtr, ytr)
    pr = p.predict_proba(Xte)[:,1]
    # threshold tuned for F1 on training CV predictions (no test leakage)
    ths = np.linspace(.1,.9,81); f1s=[f1_score(ytr, cvauc>=t) for t in ths]; t=float(ths[int(np.argmax(f1s))])
    pred = (pr>=t).astype(int)
    rows[name] = dict(Threshold=round(t,2), Accuracy=accuracy_score(yte,pred), Precision=precision_score(yte,pred),
        Recall=recall_score(yte,pred), F1=f1_score(yte,pred), ROC_AUC=roc_auc_score(yte,pr),
        CV_AUC_train=roc_auc_score(ytr,cvauc))
res = pd.DataFrame(rows).T.round(3); print(res)
res.to_csv("artifacts/model_metrics.csv")
best = res.ROC_AUC.astype(float).idxmax() if False else res.CV_AUC_train.astype(float).idxmax()
print("Selected (by train-CV AUC):", best)

# Calibrated, out-of-fold risk probabilities for EVERY employee (honest scores)
cal = CalibratedClassifierCV(mk(models[best]), method="sigmoid", cv=5)
oof = cross_val_predict(cal, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:,1]
final = CalibratedClassifierCV(mk(models[best]), method="sigmoid", cv=5).fit(X, y)
joblib.dump(final, "artifacts/final_model.joblib")
print("Calibrated OOF AUC", round(roc_auc_score(y, oof),3), "Brier", round(brier_score_loss(y,oof),3))

# Explainability: SHAP on a gradient-boosting model for drivers; LR coefs for direction
import shap
xgb = mk(models["XGBoost"]).fit(X, y)
Xt = xgb[:-1].transform(X); names = list(xgb[0].get_feature_names_out())
Xt = Xt.toarray() if hasattr(Xt,"toarray") else Xt
expl = shap.TreeExplainer(xgb[-1]); sv = expl.shap_values(Xt)
glob = pd.DataFrame({"feature": [n.split("__",1)[1] for n in names], "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
glob.to_csv("artifacts/global_importance.csv", index=False)
np.save("artifacts/shap_values.npy", sv); json.dump([n.split("__",1)[1] for n in names], open("artifacts/shap_names.json","w"))

# Reason codes: top positive SHAP drivers, mapped to human-readable
def pretty(f, v):
    if "_" in f and any(f.startswith(c+"_") for c in CAT):
        c = next(c for c in CAT if f.startswith(c+"_")); return f"{c} = {f[len(c)+1:]}"
    return f
fn = [n.split("__",1)[1] for n in names]
reasons=[]
for i in range(len(X)):
    idx = np.argsort(-sv[i])[:3]
    reasons.append("; ".join(f"{pretty(fn[j],0)}" for j in idx if sv[i][j]>0))
out = raw.copy()
out["AttritionProb"] = oof.round(4)
out["RiskCategory"] = pd.cut(oof, [-1,.3,.6,2], labels=["Low","Medium","High"])
out["TopDrivers"] = reasons
out = pd.concat([out, df[["IncomePerExpYear","PromotionDelayRatio","LongNoPromotion","EngagementScore","WorkloadStress","CompaniesPerYear"]]], axis=1)
out.to_csv("artifacts/scored_employees.csv", index=False)
print(out.RiskCategory.value_counts())
print(out.groupby("RiskCategory").Attrition.mean().round(3))
print(glob.head(12))
json.dump(dict(best=best, num=NUM), open("artifacts/meta.json","w"))
