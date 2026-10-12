
import optuna
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Define Optuna objective
def objective(trial):

    # Optuna chooses model parameters
    tree_depth = trial.suggest_int(
        "tree_depth", 2, 12
    )

    forest_estimators = trial.suggest_int(
        "forest_estimators", 50, 200, step=50
    )

    forest_depth = trial.suggest_int(
        "forest_depth", 3, 20
    )

    logistic_C = trial.suggest_float(
        "logistic_C", 0.01, 10, log=True
    )

    # Create individual models
    tree = DecisionTreeClassifier(
        max_depth=tree_depth,
        random_state=42
    )

    forest = RandomForestClassifier(
        n_estimators=forest_estimators,
        max_depth=forest_depth,
        random_state=42,
        n_jobs=-1
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=logistic_C,
            max_iter=2000
        )
    )

    # Combine the models
    ensemble = VotingClassifier(
        estimators=[
            ("tree", tree),
            ("forest", forest),
            ("logistic", logistic)
        ],
        voting="soft"
    )

    # Evaluate using cross-validation
    scores = cross_val_score(
        ensemble,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    return scores.mean()


# 4. Optimize the ensemble with Optuna
study = optuna.create_study(
    direction="maximize"
)

study.optimize(objective, n_trials=25)


# 5. Show best parameters
print("\nBest parameters:")
print(study.best_params)

print("\nBest cross-validation accuracy:")
print(round(study.best_value, 4))


# 6. Build models using the best parameters
best = study.best_params

tree = DecisionTreeClassifier(
    max_depth=best["tree_depth"],
    random_state=42
)

forest = RandomForestClassifier(
    n_estimators=best["forest_estimators"],
    max_depth=best["forest_depth"],
    random_state=42,
    n_jobs=-1
)

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=best["logistic_C"],
        max_iter=2000
    )
)

ensemble = VotingClassifier(
    estimators=[
        ("tree", tree),
        ("forest", forest),
        ("logistic", logistic)
    ],
    voting="soft"
)


# 7. Compare individual models and ensemble
models = {
    "Decision Tree": tree,
    "Random Forest": forest,
    "Logistic Regression": logistic,
    "Optimized Ensemble": ensemble
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    results[name] = accuracy

    print(f"{name}: {accuracy:.4f}")


# 8. Plot Optuna optimization history
trial_scores = [
    trial.value for trial in study.trials
]

best_so_far = np.maximum.accumulate(trial_scores)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(trial_scores) + 1),
    trial_scores,
    marker="o",
    label="Trial accuracy"
)

plt.plot(
    range(1, len(best_so_far) + 1),
    best_so_far,
    label="Best so far"
)

plt.title("Optuna Optimization History")
plt.xlabel("Trial")
plt.ylabel("Cross-validation accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 9. Plot model comparison
plt.figure(figsize=(10, 5))

bars = plt.bar(
    results.keys(),
    results.values()
)

plt.title("Test Accuracy: Individual Models vs Ensemble")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    value = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()