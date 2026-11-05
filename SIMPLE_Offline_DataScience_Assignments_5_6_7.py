# DATA SCIENCE LAB - ASSIGNMENTS 5, 6 AND 7
# All plots are black/white and use simple Matplotlib defaults.
# OFFLINE VERSION: all required CSV files are read locally; no internet is used.

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Common helper functions
def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


# Read CSV files stored in the current working folder.
# If a file is missing, use a small generated fallback dataset so the code
# can still run offline. Real assignment CSVs are always preferred.
DATA_FOLDER = Path.cwd()

def fallback_data(filename):
    rng = np.random.default_rng(42)
    n = 300

    if filename == "student-mat.csv":
        studytime = rng.integers(1, 5, n)
        absences = rng.integers(0, 25, n)
        g1 = np.clip(rng.normal(11, 3, n), 0, 20)
        g2 = np.clip(g1 + rng.normal(0, 2, n), 0, 20)
        g3 = np.clip(0.35 * g1 + 0.55 * g2 + 1.5 * studytime
                     - 0.08 * absences + rng.normal(0, 2, n), 0, 20)
        return pd.DataFrame({
            "G1": g1, "G2": g2, "G3": g3,
            "studytime": studytime, "absences": absences,
            "school": rng.choice(["GP", "MS"], n),
            "sex": rng.choice(["F", "M"], n)
        })

    if filename == "bank-full.csv":
        age = rng.integers(18, 75, n)
        job = rng.choice(["admin.", "technician", "services", "blue-collar"], n)
        education = rng.choice(["primary", "secondary", "tertiary"], n)
        balance = rng.normal(1500, 2500, n).astype(int)
        duration = rng.integers(30, 900, n)
        campaign = rng.integers(1, 8, n)
        probability = 1 / (1 + np.exp(-(-2 + duration / 350
                                       + balance / 12000)))
        y = np.where(rng.random(n) < probability, "yes", "no")
        return pd.DataFrame({
            "age": age, "job": job, "marital": rng.choice(["married", "single"], n),
            "education": education, "default": rng.choice(["no", "yes"], n, p=[.95, .05]),
            "balance": balance, "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n), "contact": rng.choice(["cellular", "unknown"], n),
            "day": rng.integers(1, 31, n), "month": rng.choice(["may", "jun", "aug"], n),
            "duration": duration, "campaign": campaign,
            "pdays": rng.choice([-1, 10, 30, 90], n), "previous": rng.integers(0, 5, n),
            "poutcome": rng.choice(["unknown", "failure", "success"], n), "y": y
        })

    if filename == "tips.csv":
        total_bill = rng.uniform(5, 55, n)
        tip = np.maximum(0.5, total_bill * rng.normal(.17, .04, n))
        return pd.DataFrame({
            "total_bill": total_bill, "tip": tip,
            "sex": rng.choice(["Male", "Female"], n),
            "smoker": rng.choice(["Yes", "No"], n),
            "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n),
            "time": rng.choice(["Lunch", "Dinner"], n),
            "size": rng.integers(1, 7, n)
        })

    if filename == "lung.csv":
        sex = rng.choice([1, 2], n)
        time = rng.integers(20, 600, n)
        status = rng.choice([1, 2], n, p=[.35, .65])
        return pd.DataFrame({"time": time, "status": status, "sex": sex})

    if filename == "mpg.csv":
        horsepower = rng.uniform(45, 230, n)
        mpg = 48 - .13 * horsepower + .00025 * horsepower ** 2
        mpg += rng.normal(0, 3, n)
        return pd.DataFrame({
            "horsepower": horsepower, "mpg": mpg,
            "cylinders": rng.choice([4, 6, 8], n),
            "weight": rng.integers(1600, 5000, n)
        })

    raise FileNotFoundError("No fallback dataset is defined for " + filename)


def read_local_csv(filename, **kwargs):
    path = DATA_FOLDER / filename
    if path.exists():
        return pd.read_csv(path, **kwargs)

    print("\nWARNING: " + filename + " was not found.")
    print("A generated fallback dataset is being used for demonstration.")
    print("Results will NOT be the actual assignment dataset results.\n")
    return fallback_data(filename)


# ============================================================
# ASSIGNMENT 5
# ============================================================

# Q1(a): Binomial distribution - model 20 independent patients
n, p = 20, 0.70
binom = stats.binom(n, p)

# Q1(b): Calculate the three required probabilities
print("\nASSIGNMENT 5 - Q1: BINOMIAL")
print("P(X = 15):", binom.pmf(15))
print("P(X >= 16):", binom.sf(15))
print("P(X < 10):", binom.cdf(9))

# Q1(c): Calculate expected value and variance
print("Expected value:", n * p)
print("Variance:", n * p * (1 - p))

# Q1(d): Generate the probability distribution
x = np.arange(n + 1)
plt.figure()
plt.bar(x, binom.pmf(x), color="black")
plt.xlabel("Number of positive responses")
plt.ylabel("Probability")
plt.show()

# Q1(e): Explain why Binomial distribution is suitable
print("Binomial is suitable because there are 20 independent trials,")
print("two possible outcomes, and a constant response probability.")


# Q1(f): Load the given emergency arrival observations
arrivals = np.array([4, 7, 5, 6, 3, 8, 5, 4, 6, 7,
                     5, 3, 4, 6, 5, 8, 4, 5, 6, 7])

# Q1(g): Estimate the Poisson arrival rate
lam = arrivals.mean()
pois = stats.poisson(lam)
print("\nASSIGNMENT 5 - Q1: POISSON")
print("Average arrival rate:", lam)

# Q1(h): Calculate the required Poisson probabilities
print("P(X = 5):", pois.pmf(5))
print("P(X < 4):", pois.cdf(3))
print("P(X > 7):", pois.sf(7))

# Q1(i): Compare observed and theoretical frequencies
x = np.arange(0, arrivals.max() + 1)
observed = pd.Series(arrivals).value_counts().reindex(x, fill_value=0)
expected = pois.pmf(x) * len(arrivals)
print(pd.DataFrame({
    "Arrivals": x,
    "Observed frequency": observed.values,
    "Theoretical frequency": expected
}).to_string(index=False))

# Q1(j): Plot observed and theoretical frequencies
plt.figure()
plt.bar(x - 0.2, observed.values, width=0.4,
        color="black", label="Observed")
plt.bar(x + 0.2, expected, width=0.4,
        color="lightgray", edgecolor="black", label="Theoretical")
plt.xlabel("Number of arrivals")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Q1(k): Comment on whether Poisson is a reasonable model
print("Compare the observed and theoretical frequencies and the sample")
print("mean and variance. A Poisson model assumes independent arrivals")
print("at an approximately constant rate.")


# Q2(a): Load the 30 examination marks
marks = np.array([
    56, 72, 68, 81, 75, 63, 91, 48, 67, 73,
    85, 59, 77, 69, 54, 88, 71, 64, 79, 82,
    61, 74, 93, 57, 66, 76, 70, 52, 84, 62
])

# Q2(b): Calculate descriptive statistics
print("\nASSIGNMENT 5 - Q2: NORMAL DISTRIBUTION")
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
mode = stats.mode(marks, keepdims=True)
print("Mode:", mode.mode[0] if mode.count[0] > 1 else "No mode")
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Range:", np.ptp(marks))
print("Variance:", np.var(marks, ddof=1))
print("Standard deviation:", np.std(marks, ddof=1))
q1, q3 = np.percentile(marks, [25, 75])
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", q3 - q1)

# Q2(c): Calculate the z-score for 91
mean = marks.mean()
std = marks.std(ddof=1)
print("Z-score of 91:", (91 - mean) / std)

# Q2(d): Calculate the required Normal probabilities
normal = stats.norm(mean, std)
p_gt_80 = normal.sf(80)
print("P(X > 80):", p_gt_80)
print("P(X < 60):", normal.cdf(60))
print("P(60 < X < 80):", normal.cdf(80) - normal.cdf(60))

# Q2(e): Identify observations with absolute z-score greater than 2
z = (marks - mean) / std
print("Unusual marks:", marks[np.abs(z) > 2])

# Q2(f): Compare actual and theoretical proportions above 80
print("Actual proportion above 80:", np.mean(marks > 80))
print("Normal-model proportion above 80:", p_gt_80)

# Q2(g): Plot the distribution and interpret it
plt.figure()
plt.hist(marks, bins=8, color="black")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()
print("Compare the mean, median, histogram and actual/theoretical")
print("proportions to judge how closely the marks follow a Normal model.")


# Q3(a): Load the given study, attendance and marks data
data = pd.DataFrame({
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 7, 9],
    "Attendance": [62, 68, 71, 75, 78, 82, 85, 88, 91, 65, 79, 86],
    "Final_Marks": [48, 52, 58, 61, 67, 72, 76, 81, 85, 55, 70, 79]
})

# Q3(b): Calculate means and standard deviations
print("\nASSIGNMENT 5 - Q3: CORRELATION AND COVARIANCE")
print("Means:\n", data.mean())
print("Standard deviations:\n", data.std())

# Q3(c): Calculate covariance matrix
print("Covariance matrix:\n", data.cov())

# Q3(d): Calculate Pearson correlation coefficients
print("Correlation matrix:\n", data.corr())

# Q3(e): Identify and interpret the three pairwise relationships
pairs = [
    ("Study_Hours", "Final_Marks"),
    ("Attendance", "Final_Marks"),
    ("Study_Hours", "Attendance")
]
correlations = []
for a, b in pairs:
    r, p_value = stats.pearsonr(data[a], data[b])
    correlations.append((a, b, r))
    print(a, "and", b, ": r =", r, ", p-value =", p_value)

best_pair = max(correlations, key=lambda item: abs(item[2]))
print("Strongest relationship:", best_pair)

# Q3(f): Plot all three relationships
for a, b in pairs:
    plt.figure()
    plt.scatter(data[a], data[b], color="black")
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()

# Q3(g): Compare covariance and correlation
print("Covariance depends on measurement units and scale.")
print("Correlation is standardized and ranges from -1 to +1.")

# Q3(h): Explain correlation and causation
print("Correlation does not prove causation. Other variables may")
print("influence study hours, attendance and final marks.")


# ============================================================
# ASSIGNMENT 6
# ============================================================

# Q1(a): Load the local UCI Student Performance CSV
students = read_local_csv("student-mat.csv", sep=";")

# Q1(b): Display dimensions, data types, summary, missing values and duplicates
print("\nASSIGNMENT 6 - Q1: STUDENT PERFORMANCE")
print("Dimensions:", students.shape)
print("Data types:\n", students.dtypes)
print("Summary:\n", students.describe(include="all").T)
print("Missing values:\n", students.isnull().sum())
print("Duplicates:", students.duplicated().sum())

# Q1(c): Remove duplicates and handle missing values
students = students.drop_duplicates()
for col in students.select_dtypes(include=np.number):
    students[col] = students[col].fillna(students[col].median())
for col in students.select_dtypes(exclude=np.number):
    if students[col].isnull().any():
        students[col] = students[col].fillna(students[col].mode()[0])

# Q1(d): Detect outliers using IQR for important grade columns
grade_cols = ["G1", "G2", "G3"]
q1 = students[grade_cols].quantile(0.25)
q3 = students[grade_cols].quantile(0.75)
iqr = q3 - q1
outliers = ((students[grade_cols] < q1 - 1.5 * iqr) |
            (students[grade_cols] > q3 + 1.5 * iqr)).sum()
print("IQR outlier counts:\n", outliers)

# Q1(e): Calculate required descriptive statistics for three attributes
summary = pd.DataFrame({
    "Mean": students[grade_cols].mean(),
    "Median": students[grade_cols].median(),
    "Standard deviation": students[grade_cols].std(),
    "Minimum": students[grade_cols].min(),
    "Maximum": students[grade_cols].max(),
    "IQR": iqr
})
print("Grade statistics:\n", summary)

# Q1(f): Calculate Pearson correlations, including final grade G3
for col in ["G1", "G2", "studytime", "absences"]:
    r, p_value = stats.pearsonr(students[col], students["G3"])
    print(col, "vs G3: r =", r, ", p-value =", p_value)

# Q1(g): Create at least four visualizations
plt.figure()
plt.imshow(students[["G1", "G2", "G3", "studytime", "absences"]].corr(),
           cmap="gray", vmin=-1, vmax=1)
plt.xticks(range(5), ["G1", "G2", "G3", "Study time", "Absences"], rotation=45)
plt.yticks(range(5), ["G1", "G2", "G3", "Study time", "Absences"])
plt.show()

plt.figure()
plt.scatter(students["G1"], students["G3"], color="black")
plt.xlabel("G1")
plt.ylabel("Final grade G3")
plt.show()

plt.figure()
plt.scatter(students["G2"], students["G3"], color="black")
plt.xlabel("G2")
plt.ylabel("Final grade G3")
plt.show()

plt.figure()
plt.boxplot([students[c] for c in grade_cols],
            tick_labels=grade_cols)
plt.ylabel("Grade")
plt.show()

# Q1(h): Identify the strongest examined relationship with final grade
student_corr = students[["G1", "G2", "studytime", "absences"]].corrwith(students["G3"])
print("Correlations with G3:\n", student_corr)
print("Strongest relationship:", student_corr.abs().idxmax())
print("Correlation indicates association, not causation.")


# Q2(a): Load the local UCI Bank Marketing CSV
bank = read_local_csv("bank-full.csv", sep=";")

# Q2(b): Examine dimensions, types, statistics, missing values, duplicates and target
print("\nASSIGNMENT 6 - Q2: BANK MARKETING")
print("Dimensions:", bank.shape)
print("Data types:\n", bank.dtypes)
print("Summary:\n", bank.describe(include="all").T)
print("Missing values:\n", bank.isnull().sum())
print("Duplicates:", bank.duplicated().sum())
print("Target distribution:\n", bank["y"].value_counts())

# Q2(c): Remove duplicates and separate features and target
bank = bank.drop_duplicates()
X = bank.drop(columns="y")
y = bank["y"].map({"no": 0, "yes": 1})
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Q2(d): Create at least three EDA visualizations
plt.figure()
plt.bar(["No", "Yes"], bank["y"].value_counts().reindex(["no", "yes"]).values,
        color="black")
plt.xlabel("Subscribed")
plt.ylabel("Customers")
plt.show()

plt.figure()
bank.groupby("y")["age"].mean().plot(kind="bar", color="black")
plt.xlabel("Subscribed")
plt.ylabel("Average age")
plt.show()

plt.figure()
bank["job"].value_counts().plot(kind="bar", color="black")
plt.xlabel("Job")
plt.ylabel("Customers")
plt.xticks(rotation=45)
plt.show()

# Q2(e): Split before fitting preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Q2(f): Impute, scale numerical features and encode categorical features
numeric_process = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", numeric_process, numeric_cols),
    ("cat", categorical_process, categorical_cols)
])

# Q2(g): Train Logistic Regression, Decision Tree and KNN
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}
results = []
matrices = {}

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # Q2(h): Calculate accuracy, precision, recall and F1-score
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0)
    })

    # Q2(i): Generate confusion matrix for each model
    matrices[name] = confusion_matrix(y_test, pred)

# Q2(j): Compare models and identify the best by F1-score
results = pd.DataFrame(results).sort_values("F1-score", ascending=False)
print("Model comparison:\n", results.to_string(index=False))
for name, matrix in matrices.items():
    print("Confusion matrix -", name, ":\n", matrix)
print("Best model by F1-score:", results.iloc[0]["Model"])
print("Use precision and recall as well as accuracy because the target")
print("classes are imbalanced.")


# Q3(a): Load the before and after training scores
before = np.array([54, 61, 58, 67, 63, 55, 72, 69, 60, 64, 57, 71])
after = np.array([59, 65, 63, 72, 68, 61, 76, 74, 65, 69, 62, 77])

# Q3(b): Calculate means and standard deviations
print("\nASSIGNMENT 6 - Q3: PAIRED T-TEST")
print("Before mean and SD:", before.mean(), before.std(ddof=1))
print("After mean and SD:", after.mean(), after.std(ddof=1))

# Q3(c): Perform paired t-test at alpha = 0.05
t_stat, p_value = stats.ttest_rel(after, before)
print("t-statistic:", t_stat)
print("p-value:", p_value)
print("Significant improvement:", p_value < 0.05)

# Q3(d): Calculate differences, mean, SD and probability improvement > 5
difference = after - before
mean_diff = difference.mean()
sd_diff = difference.std(ddof=1)
print("Individual differences:", difference)
print("Mean difference:", mean_diff)
print("SD of differences:", sd_diff)
print("P(improvement > 5):",
      stats.norm.sf(5, loc=mean_diff, scale=sd_diff))

# Q3(e): Compare before and after scores visually
plt.figure()
plt.boxplot([before, after], tick_labels=["Before", "After"])
plt.ylabel("Score")
plt.show()

# Q3(f): Interpret statistical and practical improvement
print("Statistical significance is determined by the p-value.")
print("Practical improvement depends on whether the mean score increase")
print("is meaningful in the context of the training program.")


# ============================================================
# ASSIGNMENT 7
# ============================================================

# Q1(a): Load and inspect the local Tips dataset
tips = read_local_csv("tips.csv")
print("\nASSIGNMENT 7 - Q1: POISSON REGRESSION")
print("Dimensions:", tips.shape)
print("Data types:\n", tips.dtypes)
print("Summary:\n", tips.describe(include="all"))
print("Missing values:\n", tips.isnull().sum())

# Q1(b): Split into 80% training and 20% testing data
train, test = train_test_split(tips, test_size=0.2, random_state=42)

# Q1(c): Fit Poisson regression with categorical variables encoded by C()
model = smf.poisson(
    "size ~ total_bill + tip + C(sex) + C(smoker) + C(day) + C(time)",
    data=train
).fit()

# Q1(d): Display model summary, coefficients and significance
print(model.summary())
print("Coefficients:\n", model.params)
print("P-values:\n", model.pvalues)

# Q1(e): Predict test observations and calculate MAE
pred = model.predict(test)
print("Test MAE:", mean_absolute_error(test["size"], pred))

# Q1(f): Explain why Poisson regression is suitable
print("Poisson regression models count responses such as party size.")


# Q2(a): Load and inspect the local Lung dataset
# The CSV must contain the columns: time, status and sex.
lung = read_local_csv("lung.csv")
print("\nASSIGNMENT 7 - Q2: SURVIVAL ANALYSIS")
print("Dimensions:", lung.shape)
print("Data types:\n", lung.dtypes)
print("Missing values:\n", lung.isnull().sum())
print("Summary:\n", lung.describe())

# Q2(b): Remove missing survival fields and correctly code the event
# In this dataset status=1 means censored and status=2 means death.
lung = lung.dropna(subset=["time", "status", "sex"]).copy()
lung["event"] = (lung["status"] == 2).astype(int)

# Q2(c): Fit and plot overall Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
kmf.fit(lung["time"], event_observed=lung["event"], label="Overall")
plt.figure()
kmf.plot_survival_function(color="black")
plt.xlabel("Time (days)")
plt.ylabel("Survival probability")
plt.show()

# Q2(d): Calculate survival probabilities and median survival time
for day in [100, 200, 365]:
    print("Survival probability at", day, "days:", kmf.predict(day))
print("Overall median survival time:", kmf.median_survival_time_)

# Q2(e): Separate patients by sex and fit two Kaplan-Meier curves
male = lung[lung["sex"] == 1]
female = lung[lung["sex"] == 2]

km_m = KaplanMeierFitter()
km_f = KaplanMeierFitter()
km_m.fit(male["time"], male["event"], label="Male")
km_f.fit(female["time"], female["event"], label="Female")

plt.figure()
km_m.plot_survival_function(color="black", linestyle="-")
km_f.plot_survival_function(color="black", linestyle="--")
plt.xlabel("Time (days)")
plt.ylabel("Survival probability")
plt.show()

# Q2(f): Calculate median survival times for both groups
print("Male median survival:", km_m.median_survival_time_)
print("Female median survival:", km_f.median_survival_time_)

# Q2(g): Perform log-rank test and interpret
test_result = logrank_test(
    male["time"], female["time"],
    event_observed_A=male["event"],
    event_observed_B=female["event"]
)
print("Log-rank statistic:", test_result.test_statistic)
print("Log-rank p-value:", test_result.p_value)
print("Significant difference at 0.05:", test_result.p_value < 0.05)


# Q3(a): Load local Auto MPG data, inspect it and remove missing values
mpg = read_local_csv("mpg.csv")
print("\nASSIGNMENT 7 - Q3: POLYNOMIAL REGRESSION")
print("Dimensions:", mpg.shape)
print("Missing values:\n", mpg.isnull().sum())
mpg = mpg.dropna(subset=["horsepower", "mpg"])

# Q3(b): Plot horsepower against MPG
X = mpg[["horsepower"]]
y = mpg["mpg"]
plt.figure()
plt.scatter(X["horsepower"], y, color="black")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()

# Q3(c): Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q3(d): Fit and evaluate simple linear regression
mpg_models = {"Linear": LinearRegression()}
mpg_models["Linear"].fit(X_train, y_train)
pred = mpg_models["Linear"].predict(X_test)
mpg_results = [{
    "Model": "Linear",
    "RMSE": rmse(y_test, pred),
    "R2": r2_score(y_test, pred)
}]

# Q3(e): Fit polynomial models of degrees 2, 3 and 4
for degree in [2, 3, 4]:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("regression", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    mpg_models["Degree " + str(degree)] = pipe
    pred = pipe.predict(X_test)
    mpg_results.append({
        "Model": "Degree " + str(degree),
        "RMSE": rmse(y_test, pred),
        "R2": r2_score(y_test, pred)
    })

# Q3(f): Compare models and identify the best test RMSE
mpg_results = pd.DataFrame(mpg_results).sort_values("RMSE")
print("Model comparison:\n", mpg_results.to_string(index=False))
print("Best model:", mpg_results.iloc[0]["Model"])

# Q3(g): Plot observed data and all fitted curves
grid = pd.DataFrame({
    "horsepower": np.linspace(X.horsepower.min(), X.horsepower.max(), 300)
})
plt.figure()
plt.scatter(X["horsepower"], y, color="black", label="Observed")
for name, fitted_model in mpg_models.items():
    plt.plot(grid["horsepower"], fitted_model.predict(grid),
             color="black", linestyle="-" if name == "Linear" else "--",
             label=name)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()

# Q3(h): Explain polynomial flexibility and overfitting
print("Higher polynomial degree increases flexibility but may overfit.")
print("The best model should be selected using test-set performance.")


# Q4(a): Load Diabetes data and create a DataFrame
# If diabetes.csv is present, use it; otherwise use sklearn's bundled dataset.
diabetes_path = DATA_FOLDER / "diabetes.csv"
if diabetes_path.exists():
    df = pd.read_csv(diabetes_path)
else:
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
print("\nASSIGNMENT 7 - Q4: SPLINE REGRESSION")

# Q4(b): Select BMI and target, then plot their relationship
X = df[["bmi"]]
y = df["target"]
plt.figure()
plt.scatter(X["bmi"], y, color="black")
plt.xlabel("BMI")
plt.ylabel("Diabetes progression")
plt.show()

# Q4(c): Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q4(d): Fit and evaluate the linear regression baseline
spline_models = {"Linear": LinearRegression()}
spline_models["Linear"].fit(X_train, y_train)
pred = spline_models["Linear"].predict(X_test)
spline_results = [{
    "Model": "Linear",
    "RMSE": rmse(y_test, pred),
    "R2": r2_score(y_test, pred)
}]

# Q4(e): Fit spline regression with 3, 5 and 7 knots
for knots in [3, 5, 7]:
    pipe = Pipeline([
        ("spline", SplineTransformer(
            n_knots=knots, degree=3, include_bias=False
        )),
        ("regression", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    spline_models[str(knots) + " knots"] = pipe
    pred = pipe.predict(X_test)
    spline_results.append({
        "Model": str(knots) + " knots",
        "RMSE": rmse(y_test, pred),
        "R2": r2_score(y_test, pred)
    })

# Q4(f): Compare all models and identify the best test RMSE
spline_results = pd.DataFrame(spline_results).sort_values("RMSE")
print("Model comparison:\n", spline_results.to_string(index=False))
print("Best model:", spline_results.iloc[0]["Model"])

# Q4(g): Plot observed data and all fitted spline curves
grid = pd.DataFrame({
    "bmi": np.linspace(X.bmi.min(), X.bmi.max(), 300)
})
plt.figure()
plt.scatter(X["bmi"], y, color="black", label="Observed")
for name, fitted_model in spline_models.items():
    plt.plot(grid["bmi"], fitted_model.predict(grid),
             color="black", linestyle="-" if name == "Linear" else "--",
             label=name)
plt.xlabel("BMI")
plt.ylabel("Diabetes progression")
plt.legend()
plt.show()

# Q4(h): Explain the effect of the number of knots
print("More knots increase spline flexibility.")
print("Too many knots may fit noise and cause overfitting.")
print("Use test RMSE and R2 to compare the spline models with linear regression.")

print("\nALL ASSIGNMENTS FINISHED")
