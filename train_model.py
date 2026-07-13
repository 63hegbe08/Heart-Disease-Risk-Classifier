

import pandas as pd
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

TARGET = "TenYearCHD"
BINARY_FIELDS = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]


def train_and_save(csv_path: str = "framingham.csv", model_path: str = "model.pkl"):
    df = pd.read_csv(csv_path)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    over_sampler = RandomOverSampler(random_state=42)
    x_train_over, y_train_over = over_sampler.fit_resample(x_train, y_train)
    base_model = make_pipeline(SimpleImputer(), RandomForestClassifier(random_state=42))
    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "randomforestclassifier__n_estimators": [10, 100],
        "randomforestclassifier__max_depth": [2, 10, 30],
    }

    grid_search = GridSearchCV(
        base_model,
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring="roc_auc",
    )
    grid_search.fit(x_train_over, y_train_over)

    best_model = grid_search.best_estimator_
    test_score = best_model.score(x_test, y_test)
    print("Best params:", grid_search.best_params_)
    print("Test accuracy:", test_score)

    feature_columns = x.columns.tolist()

    numeric_stats = {
        col: {
            "min": float(df[col].min(skipna=True)),
            "max": float(df[col].max(skipna=True)),
            "mean": float(df[col].mean(skipna=True)),
        }
        for col in feature_columns
    }

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "binary_fields": [c for c in BINARY_FIELDS if c in feature_columns],
            "numeric_stats": numeric_stats,
            "test_accuracy": test_score,
            "best_params": grid_search.best_params_,
        },
        model_path,
    )
    print(f"Model and metadata saved to '{model_path}'")


if __name__ == "__main__":
    train_and_save()
