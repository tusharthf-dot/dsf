"""
================================================================================
Data Science Lab (CS2311) -- Combined Solutions
================================================================================

This single file contains:

    PART 0  : Web scraping of the Campus Events HTML page with BeautifulSoup
    PART 1  : Assignment 2  (California Housing - regression)
                 Q1  EDA + preprocessing
                 Q2  Three Multiple Linear Regression models (A/B/C)
                 Q3  Feature selection (RFE) vs all-features model
    PART 2  : Assignment 3
                 Q1  Telco Customer Churn classification (5 models)
                 Q2  PCA & SVD dimensionality reduction + re-classification
                 Q3  House Sales regression: Linear / Ridge / Lasso / ElasticNet
    PART 3  : Assignment 4  (hypothesis testing)
                 Q1  Independent two-sample t-test
                 Q2  Paired t-test
                 Q3  Demonstrating the difference between the two tests

HOW TO RUN
----------
    python ds_lab_solutions.py

Each part is a self-contained function; scroll to the bottom (`main`) and
comment out whatever you don't want to run.

DATASETS
--------
  * California Housing  -> ships with scikit-learn, downloads automatically.
  * Telco Customer Churn -> download from Kaggle:
        "blastchar/telco-customer-churn"
        file: WA_Fn-UseC_-Telco-Customer-Churn.csv
  * House Sales (King County) -> download from Kaggle:
        "harlfoxem/housesalesprediction"
        file: kc_house_data.csv
  Put those two CSVs next to this script (or edit the paths in CONFIG below).
  The script checks whether each file exists and skips gracefully if not.

DEPENDENCIES
------------
    pip install beautifulsoup4 pandas numpy scikit-learn scipy matplotlib seaborn
"""

import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to disk without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# CONFIG : file paths (edit if your files live elsewhere)
# ---------------------------------------------------------------------------
HTML_FILE   = "Campus_Events_Practice_Harder.html"
TELCO_CSV   = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HOUSES_CSV  = "kc_house_data.csv"
FIG_DIR     = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _savefig(name):
    """Save the current matplotlib figure into FIG_DIR and close it."""
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"   [saved figure] {path}")


def banner(text):
    line = "=" * 78
    print(f"\n{line}\n{text}\n{line}")


# ===========================================================================
# PART 0 : WEB SCRAPING WITH BEAUTIFULSOUP
# ===========================================================================
from bs4 import BeautifulSoup


def scrape_campus_events(html_path=HTML_FILE):
    """
    Parse the Campus Events HTML page and return a tidy pandas DataFrame,
    one row per event. Demonstrates the core BeautifulSoup patterns:
    select_one / select, reading attributes, and stripping <strong> labels.
    """
    banner("PART 0 : Web scraping the Campus Events page (BeautifulSoup)")

    if not os.path.exists(html_path):
        print(f"   HTML file '{html_path}' not found -- skipping.")
        return None

    with open(html_path, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "html.parser")

    # Page-level information
    page_title = soup.title.get_text(strip=True) if soup.title else None
    intro      = soup.select_one("p.intro")
    notice     = soup.select_one("p.notice")
    updated    = soup.select_one("p.updated")
    print(f"   Page title : {page_title}")
    if intro:   print(f"   Intro      : {intro.get_text(strip=True)}")
    if notice:  print(f"   Notice     : {notice.get_text(strip=True)}")
    if updated: print(f"   {updated.get_text(strip=True)}")

    def label_value(card, css_class):
        """Return the text of <p class="css_class"> with its <strong> label removed."""
        el = card.select_one(f"p.{css_class}")
        if el is None:
            return None
        strong = el.find("strong")
        if strong:
            strong.extract()          # drop the "Venue:" / "Date:" label
        return el.get_text(strip=True)

    def tag_text(card, selector):
        el = card.select_one(selector)
        return el.get_text(strip=True) if el else None

    rows = []
    for card in soup.select("div.event"):           # every event block
        organizer = tag_text(card, "p.organizer")
        if organizer and ":" in organizer:          # "Organized by: X" -> "X"
            organizer = organizer.split(":", 1)[1].strip()

        seats = label_value(card, "seats")
        m_seats = re.search(r"\d+", seats) if isinstance(seats, str) else None
        if m_seats:
            seats = int(m_seats.group())          # "Seats: 30" -> 30

        rows.append({
            "event_id":    card.get("data-event-id"),
            "title":       tag_text(card, "h2"),
            "featured":    "featured" in (card.get("class") or []),
            "badge":       tag_text(card, "span.badge"),
            "description": tag_text(card, "p.description"),
            "venue":       label_value(card, "venue"),
            "date":        label_value(card, "date"),
            "time":        label_value(card, "time"),
            "category":    tag_text(card, "span.category"),
            "level":       tag_text(card, "span.level"),
            "fee":         tag_text(card, "span.fee"),
            "seats":       seats,
            "organizer":   organizer,
        })

    df = pd.DataFrame(rows)
    print(f"\n   Scraped {len(df)} events.\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df[["event_id", "title", "category", "date", "fee", "seats"]])

    # A couple of tiny analyses to show the data is usable
    print("\n   Events per category:")
    print(df["category"].value_counts().to_string())
    seats_num = pd.to_numeric(df["seats"], errors="coerce")   # "Limited" -> NaN (ignored)
    print(f"\n   Total available seats across all events: {int(seats_num.sum())}"
          f"  (non-numeric seat values ignored: {int(seats_num.isna().sum())})")
    free = df[df["fee"].str.lower().str.contains("free", na=False)]
    print(f"   Free events: {', '.join(free['title'])}")

    # Save to CSV so the scraped data can be reused
    df.to_csv("campus_events_scraped.csv", index=False)
    print("   [saved] campus_events_scraped.csv")
    return df


# ===========================================================================
# PART 1 : ASSIGNMENT 2  (California Housing - Regression)
# ===========================================================================
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


def _cap_outliers_iqr(df, cols):
    """Cap (winsorize) outliers of the given columns to the 1.5*IQR fences."""
    out = df.copy()
    for c in cols:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = out[c].clip(lo, hi)
    return out


def _load_california():
    """Try sklearn first (needs internet). If that fails, fall back to a local
    Kaggle 'housing.csv' and rename its columns to the sklearn names so the
    rest of the code is unchanged."""
    try:
        return fetch_california_housing(as_frame=True).frame.copy()
    except Exception as e:
        print(f"   sklearn download failed ({type(e).__name__}); trying local housing.csv")
    if not os.path.exists("housing.csv"):
        print("   'housing.csv' not found either -- skipping Assignment 2.")
        return None
    k = pd.read_csv("housing.csv")
    k["total_bedrooms"] = k["total_bedrooms"].fillna(k["total_bedrooms"].median())
    return pd.DataFrame({
        "MedInc": k["median_income"], "HouseAge": k["housing_median_age"],
        "AveRooms": k["total_rooms"] / k["households"],
        "AveBedrms": k["total_bedrooms"] / k["households"],
        "Population": k["population"], "AveOccup": k["population"] / k["households"],
        "Latitude": k["latitude"], "Longitude": k["longitude"],
        "MedHouseVal": k["median_house_value"] / 1e5})


def assignment2():
    banner("PART 1 : ASSIGNMENT 2  (California Housing Prices)")

    # ----- Load -----------------------------------------------------------
    df = _load_california()                      # features + MedHouseVal target
    if df is None:
        return None
    target = "MedHouseVal"
    features = [c for c in df.columns if c != target]

    # -------------------------------------------------------------------
    # Q1 : Exploratory Data Analysis + preprocessing
    # -------------------------------------------------------------------
    banner("A2-Q1 : Exploratory Data Analysis & Preprocessing")
    print(f"   Dataset shape      : {df.shape}  (rows x columns)")
    print(f"   Features           : {features}")
    print(f"   Target             : {target}")
    print("\n   Data types:")
    print(df.dtypes.to_string())
    print("\n   Summary statistics:")
    print(df.describe().T.round(3).to_string())
    print(f"\n   Missing values (total)   : {int(df.isnull().sum().sum())}")
    print(f"   Duplicate rows           : {int(df.duplicated().sum())}")

    # Outlier count per feature using the IQR rule
    print("\n   Outliers per feature (1.5*IQR rule):")
    for c in features:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        mask = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
        print(f"      {c:<12}: {int(mask.sum())}")

    # ---- Visualisations ----
    df[features + [target]].hist(bins=40, figsize=(14, 10))
    plt.suptitle("Histograms of all variables")
    _savefig("a2q1_histograms.png")

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    _savefig("a2q1_correlation_heatmap.png")

    plt.figure(figsize=(14, 7))
    df[features].plot(kind="box", subplots=True, layout=(2, 4),
                      figsize=(14, 7), sharex=False, sharey=False)
    plt.suptitle("Box plots of features (outlier view)")
    _savefig("a2q1_boxplots.png")

    # Strongest-correlated feature vs target scatter
    corr_target = df.corr()[target].drop(target).abs().sort_values(ascending=False)
    top_feat = corr_target.index[0]
    plt.figure(figsize=(7, 5))
    plt.scatter(df[top_feat], df[target], s=6, alpha=0.3)
    plt.xlabel(top_feat); plt.ylabel(target)
    plt.title(f"{target} vs {top_feat} (|corr|={corr_target.iloc[0]:.2f})")
    _savefig("a2q1_scatter_top_feature.png")

    # ---- Preprocessing ----
    # (1) No missing values here, but we show the handling step anyway.
    df = df.fillna(df.median(numeric_only=True))
    # (2) Treat outliers in the predictors by IQR capping.
    df = _cap_outliers_iqr(df, features)
    # (3) Feature scaling + (4) train/test split.
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=features, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=features, index=X_test.index)
    print(f"\n   Train set: {X_train_s.shape},  Test set: {X_test_s.shape}")
    print("   Preprocessing done: missing handled, outliers capped, features scaled.")

    # -------------------------------------------------------------------
    # Q2 : Three Multiple Linear Regression models (A / B / C)
    # -------------------------------------------------------------------
    banner("A2-Q2 : Three Multiple Linear Regression models")

    # Model A -> 3 most correlated features with the target
    featA = list(corr_target.index[:3])
    # Model B -> 5 selected features (top-5 by correlation here)
    featB = list(corr_target.index[:5])
    # Model C -> all predictors
    featC = features
    print(f"   Model A features (3): {featA}")
    print(f"   Model B features (5): {featB}")
    print(f"   Model C features ({len(featC)}): {featC}")

    def run_model(feat_list):
        model = LinearRegression()
        t0 = time.perf_counter()
        model.fit(X_train_s[feat_list], y_train)
        train_time = time.perf_counter() - t0
        pred = model.predict(X_test_s[feat_list])
        mae, mse, rmse, r2 = _regression_metrics(y_test, pred)
        return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, Time_s=train_time)

    results = {"Model A (3 feats)": run_model(featA),
               "Model B (5 feats)": run_model(featB),
               "Model C (all feats)": run_model(featC)}
    table = pd.DataFrame(results).T.round(4)
    print("\n   Comparative performance:")
    print(table.to_string())
    best = table["R2"].idxmax()
    print(f"\n   -> Best R2: {best} (R2={table.loc[best,'R2']:.4f})")
    print("   Trade-off: adding features raises R2 but also training cost;")
    print("   note how MAE/RMSE fall as more predictors are included.")

    # -------------------------------------------------------------------
    # Q3 : Feature selection (RFE) vs all-features model
    # -------------------------------------------------------------------
    banner("A2-Q3 : Feature selection with RFE")
    n_select = 4
    rfe = RFE(LinearRegression(), n_features_to_select=n_select)
    rfe.fit(X_train_s, y_train)
    selected = list(X_train_s.columns[rfe.support_])
    print(f"   RFE selected {n_select} features: {selected}")

    rfe_res = run_model(selected)
    all_res = run_model(featC)
    comp = pd.DataFrame({"RFE-selected features": rfe_res,
                         "All features": all_res}).T.round(4)
    print("\n   RFE vs All-features comparison:")
    print(comp.to_string())
    dr2 = all_res["R2"] - rfe_res["R2"]
    verdict = ("maintains" if abs(dr2) < 0.01
               else ("improves" if dr2 < 0 else "slightly degrades"))
    print(f"\n   -> Using {n_select} RFE features {verdict} accuracy "
          f"(delta R2 = {dr2:+.4f}) while using fewer predictors.")
    return table


# ===========================================================================
# PART 2 : ASSIGNMENT 3
# ===========================================================================
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve)


def _build_classifiers():
    """The five classification algorithms used throughout Assignment 3."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest":       RandomForestClassifier(n_estimators=200,
                                                     random_state=RANDOM_STATE),
        "KNN":                KNeighborsClassifier(n_neighbors=7),
        "GradientBoosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _evaluate_classifiers(X_train, X_test, y_train, y_test, tag, make_curves=True):
    """Train the 5 classifiers, return a metrics DataFrame, and (optionally)
    draw ROC and Precision-Recall curves for the set."""
    models = _build_classifiers()
    metrics, roc_data, pr_data = [], {}, {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        pred = model.predict(X_test)
        proba = (model.predict_proba(X_test)[:, 1]
                 if hasattr(model, "predict_proba") else model.decision_function(X_test))

        cm = confusion_matrix(y_test, pred)
        metrics.append({
            "Model": name,
            "Accuracy":  accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall":    recall_score(y_test, pred, zero_division=0),
            "F1":        f1_score(y_test, pred, zero_division=0),
            "ROC_AUC":   roc_auc_score(y_test, proba),
            "Time_s":    fit_time,
        })
        print(f"   [{name}] confusion matrix (tn fp / fn tp):")
        print("      ", cm.ravel())

        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)
        pr_data[name]  = (rec, prec)

    df_metrics = pd.DataFrame(metrics).set_index("Model").round(4)
    print(f"\n   Comparative metrics [{tag}]:")
    print(df_metrics.to_string())
    best = df_metrics["F1"].idxmax()
    print(f"   -> Best F1 model [{tag}]: {best}")

    if make_curves:
        plt.figure(figsize=(7, 6))
        for name, (fpr, tpr) in roc_data.items():
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC curves ({tag})"); plt.legend()
        _savefig(f"a3q1_roc_{tag}.png")

        plt.figure(figsize=(7, 6))
        for name, (rec, prec) in pr_data.items():
            plt.plot(rec, prec, label=name)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall curves ({tag})"); plt.legend()
        _savefig(f"a3q1_pr_{tag}.png")

    return df_metrics


def _load_telco(path=TELCO_CSV):
    """Load and preprocess the Telco Churn dataset. Returns scaled splits."""
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # TotalCharges has blank strings for brand-new customers -> numeric + fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop_duplicates()
    # Target
    y = (df["Churn"].map({"Yes": 1, "No": 0})).astype(int)
    df = df.drop(columns=["Churn"])
    # One-hot encode the categorical predictors
    X = pd.get_dummies(df, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train.values, y_test.values, X.columns, y


def assignment3_q1_q2():
    banner("PART 2 : ASSIGNMENT 3  (Q1 Churn classification, Q2 PCA/SVD)")

    if not os.path.exists(TELCO_CSV):
        print(f"   '{TELCO_CSV}' not found. Download it from Kaggle "
              f"(blastchar/telco-customer-churn) and place it here. Skipping.")
        return

    X_train, X_test, y_train, y_test, cols, y = _load_telco()
    print(f"   Features after encoding: {len(cols)}")
    print(f"   Churn rate: {y.mean():.3f}")

    # ---- Target-distribution + a feature plot (EDA) ----
    plt.figure(figsize=(5, 4))
    y.value_counts().rename({0: "No", 1: "Yes"}).plot(kind="bar", color=["#4c72b0", "#dd8452"])
    plt.title("Churn distribution"); plt.ylabel("count")
    _savefig("a3q1_target_distribution.png")

    # -------------------------------------------------------------------
    # Q1 : five classifiers on the full (original) feature set
    # -------------------------------------------------------------------
    banner("A3-Q1 : Classification with 5 algorithms (original features)")
    original = _evaluate_classifiers(X_train, X_test, y_train, y_test,
                                     tag="original", make_curves=True)

    # -------------------------------------------------------------------
    # Q2 : PCA and SVD dimensionality reduction, then re-classify
    # -------------------------------------------------------------------
    banner("A3-Q2 : PCA & SVD dimensionality reduction")

    # Choose #components to retain ~95% variance via PCA
    pca_full = PCA().fit(X_train)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= 0.95) + 1)
    print(f"   PCA components retaining >=95% variance: {n_comp} "
          f"(of {X_train.shape[1]})")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    plt.axhline(0.95, color="r", ls="--"); plt.axvline(n_comp, color="g", ls="--")
    plt.xlabel("number of components"); plt.ylabel("cumulative explained variance")
    plt.title("PCA explained variance")
    _savefig("a3q2_pca_variance.png")

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X_train)
    Xtr_pca, Xte_pca = pca.transform(X_train), pca.transform(X_test)

    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE).fit(X_train)
    Xtr_svd, Xte_svd = svd.transform(X_train), svd.transform(X_test)

    pca_res = _evaluate_classifiers(Xtr_pca, Xte_pca, y_train, y_test,
                                    tag="PCA", make_curves=False)
    svd_res = _evaluate_classifiers(Xtr_svd, Xte_svd, y_train, y_test,
                                    tag="SVD", make_curves=False)

    # ---- Compare original vs PCA vs SVD ----
    banner("A3-Q2 : Original vs PCA vs SVD")
    summary = pd.DataFrame({
        "Original (%d feats)" % X_train.shape[1]: original.mean(),
        "PCA (%d comps)" % n_comp:                pca_res.mean(),
        "SVD (%d comps)" % n_comp:                svd_res.mean(),
    }).round(4)
    print("   Mean metric across the 5 models for each feature set:")
    print(summary.to_string())

    # Side-by-side F1 bar chart
    f1_cmp = pd.DataFrame({"Original": original["F1"],
                           "PCA": pca_res["F1"],
                           "SVD": svd_res["F1"]})
    f1_cmp.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("F1 score"); plt.title("F1 by model: Original vs PCA vs SVD")
    plt.xticks(rotation=20)
    _savefig("a3q2_f1_comparison.png")
    d_f1  = summary.loc["F1"]
    d_t   = summary.loc["Time_s"]
    orig_c, pca_c, svd_c = summary.columns
    print(f"\n   -> Features reduced {X_train.shape[1]} -> {n_comp}. "
          f"Mean F1: original {d_f1[orig_c]:.4f}, PCA {d_f1[pca_c]:.4f}, SVD {d_f1[svd_c]:.4f}.")
    print("      Reduction " + ("IMPROVES" if d_f1[pca_c] > d_f1[orig_c] + 0.005 else
                                "slightly DECREASES" if d_f1[pca_c] < d_f1[orig_c] - 0.005 else
                                "MAINTAINS") + " classification performance.")
    print(f"      Mean training time: original {d_t[orig_c]:.3f}s vs PCA {d_t[pca_c]:.3f}s "
          "(tree models are slower on dense PCA components than on sparse 0/1 columns).")
    print("      PCA and SVD give near-identical results because the data was already "
          "standardised (centred), and PCA = SVD on centred data.")


def assignment3_q3():
    banner("A3-Q3 : House Sales regression with regularization")

    if not os.path.exists(HOUSES_CSV):
        print(f"   '{HOUSES_CSV}' not found. Download it from Kaggle "
              f"(harlfoxem/housesalesprediction) and place it here. Skipping.")
        return

    df = pd.read_csv(HOUSES_CSV)
    # Drop identifier / date columns if present; keep numeric predictors
    for col in ["id", "date", "zipcode"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna()
    target = "price"
    X = df.drop(columns=[target])
    y = df[target]
    X = X.select_dtypes(include=[np.number])     # keep numeric features only

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    # ---- Baseline Multiple Linear Regression ----
    lin = LinearRegression().fit(Xtr, y_train)
    mae, mse, rmse, r2 = _regression_metrics(y_test, lin.predict(Xte))
    print(f"   Linear Regression -> MAE={mae:,.0f}  MSE={mse:,.0f}  "
          f"RMSE={rmse:,.0f}  R2={r2:.4f}")

    # ---- Regularized models over a grid of lambda (alpha) ----
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    reg_builders = {
        "Ridge":      lambda a: Ridge(alpha=a),
        "Lasso":      lambda a: Lasso(alpha=a, max_iter=10000),
        "ElasticNet": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000),
    }

    records, curves = [], {}
    for name, builder in reg_builders.items():
        train_err, test_err = [], []
        for a in lambdas:
            m = builder(a).fit(Xtr, y_train)
            tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(Xtr)))
            te_rmse = np.sqrt(mean_squared_error(y_test, m.predict(Xte)))
            train_err.append(tr_rmse)
            test_err.append(te_rmse)
            _, _, te_rmse2, te_r2 = _regression_metrics(y_test, m.predict(Xte))
            records.append({"Model": name, "lambda": a,
                            "Test_RMSE": te_rmse2, "Test_R2": te_r2})
        curves[name] = (train_err, test_err)

        # train/test error vs lambda
        plt.figure(figsize=(7, 5))
        plt.plot(lambdas, train_err, marker="o", label="train RMSE")
        plt.plot(lambdas, test_err,  marker="s", label="test RMSE")
        plt.xscale("log"); plt.xlabel("lambda (alpha)"); plt.ylabel("RMSE")
        plt.title(f"{name}: error vs lambda"); plt.legend()
        _savefig(f"a3q3_{name.lower()}_error_vs_lambda.png")

    reg_table = pd.DataFrame(records)
    print("\n   Regularized model metrics across lambda:")
    print(reg_table.round(3).to_string(index=False))

    # ---- Compare coefficients at a fixed, moderate lambda ----
    a = 1.0
    coefs = pd.DataFrame({
        "Linear":     lin.coef_,
        "Ridge":      Ridge(alpha=a).fit(Xtr, y_train).coef_,
        "Lasso":      Lasso(alpha=a, max_iter=10000).fit(Xtr, y_train).coef_,
        "ElasticNet": ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000).fit(Xtr, y_train).coef_,
    }, index=X.columns)
    print(f"\n   Coefficients at lambda={a}:")
    print(coefs.round(1).to_string())

    coefs.plot(kind="bar", figsize=(13, 6))
    plt.ylabel("coefficient"); plt.title(f"Regression coefficients (lambda={a})")
    plt.xticks(rotation=60, ha="right")
    _savefig("a3q3_coefficient_comparison.png")
    big = 1000.0
    zeros = pd.DataFrame({
        f"lambda={a}":   [int((Lasso(alpha=a, max_iter=10000).fit(Xtr, y_train).coef_ == 0).sum()),
                          int((ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000).fit(Xtr, y_train).coef_ == 0).sum()),
                          int((Ridge(alpha=a).fit(Xtr, y_train).coef_ == 0).sum())]
        for a in (1.0, big)}, index=["Lasso", "ElasticNet", "Ridge"])
    print("\n   Number of coefficients that are EXACTLY zero:")
    print(zeros.to_string())
    print("\n   -> Ridge shrinks coefficients smoothly (never to zero). Lasso needs a")
    print("      large lambda before it zeroes features (feature selection). ElasticNet blends both.")
    print("      NOTE: sqft_living = sqft_above + sqft_basement exactly (multicollinearity),")
    print("      so Linear/Ridge/Lasso split weight between them differently.")
    print("      NOTE: sklearn's alpha is NOT on the same scale for Ridge and Lasso/ElasticNet,")
    print("      so the same lambda hurts ElasticNet far more than Ridge.")


# ===========================================================================
# PART 3 : ASSIGNMENT 4  (Hypothesis testing with t-tests)
# ===========================================================================
from scipy import stats


def _interpret(p, alpha=0.05):
    return ("Reject H0: statistically significant difference."
            if p < alpha else
            "Fail to reject H0: no statistically significant difference.")


def assignment4():
    banner("PART 3 : ASSIGNMENT 4  (t-tests)")
    alpha = 0.05

    # -------------------------------------------------------------------
    # Q1 : Independent two-sample t-test
    # -------------------------------------------------------------------
    banner("A4-Q1 : Independent two-sample t-test")
    group_A = [65, 70, 68, 72, 66, 75, 69, 71, 67, 70]
    group_B = [72, 75, 78, 74, 80, 77, 73, 79, 76, 81]
    t_stat, p_val = stats.ttest_ind(group_A, group_B)
    print(f"   group_A mean = {np.mean(group_A):.2f}, "
          f"group_B mean = {np.mean(group_B):.2f}")
    print(f"   t-statistic  = {t_stat:.4f}")
    print(f"   p-value      = {p_val:.6f}")
    print(f"   alpha        = {alpha}")
    print(f"   Conclusion   : {_interpret(p_val, alpha)}")

    # -------------------------------------------------------------------
    # Q2 : Paired t-test
    # -------------------------------------------------------------------
    banner("A4-Q2 : Paired t-test (before vs after training)")
    before = [52, 48, 65, 60, 55, 70, 58, 62, 50, 67]
    after  = [60, 55, 70, 66, 63, 75, 64, 68, 57, 72]
    t_stat2, p_val2 = stats.ttest_rel(before, after)
    diffs = np.array(after) - np.array(before)
    print(f"   mean before = {np.mean(before):.2f}, mean after = {np.mean(after):.2f}")
    print(f"   mean paired difference (after-before) = {diffs.mean():.2f}")
    print(f"   t-statistic  = {t_stat2:.4f}")
    print(f"   p-value      = {p_val2:.6f}")
    print(f"   alpha        = {alpha}")
    print(f"   Conclusion   : {_interpret(p_val2, alpha)}")

    # -------------------------------------------------------------------
    # Q3 : Demonstrate why the two tests differ
    # -------------------------------------------------------------------
    banner("A4-Q3 : Independent vs Paired t-test on the SAME data")
    # Same two columns treated two ways: as unrelated groups, and as pairs.
    t_ind, p_ind = stats.ttest_ind(before, after)
    t_par, p_par = stats.ttest_rel(before, after)
    comp = pd.DataFrame({
        "Test":        ["Independent t-test", "Paired t-test"],
        "t_statistic": [t_ind, t_par],
        "p_value":     [p_ind, p_par],
        "Decision":    [_interpret(p_ind, alpha), _interpret(p_par, alpha)],
    })
    print(comp.to_string(index=False))
    print("\n   Why they differ:")
    print("   - The independent test assumes the two samples are unrelated and")
    print("     uses the pooled between-group variability in its denominator.")
    print("   - The paired test works on each subject's before/after difference,")
    print("     removing person-to-person variation. That smaller, focused")
    print("     variance usually yields a larger |t| and a smaller p-value,")
    print("     which is why the paired test is more powerful when data is paired.")


# ===========================================================================
# MAIN
# ===========================================================================
def _safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"\n   !! {fn.__name__} failed: {type(e).__name__}: {e}")


def main():
    _safe(scrape_campus_events)     # PART 0 : web scraping
    _safe(assignment2)              # PART 1 : Assignment 2
    _safe(assignment3_q1_q2)        # PART 2 : Assignment 3 Q1 + Q2
    _safe(assignment3_q3)           # PART 2 : Assignment 3 Q3
    _safe(assignment4)              # PART 3 : Assignment 4
    banner("ALL DONE -- figures saved in the 'figures/' folder.")


if __name__ == "__main__":
    main()