import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=500,
    random_state=42
)

# Train model
mlp.fit(X_train_scaled, y_train)
print("MLPRegressor model trained successfully")

os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_actual_vs_pred(y_true, y_pred, split_name, filename):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3, label="Predictions")

    ax_max = 5.5
    ax.plot([0, ax_max], [0, ax_max], "r--", linewidth=1, label="Perfect fit")
    ax.set_xlim(0, ax_max)
    ax.set_ylim(0, ax_max)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{split_name}: Actual vs Predicted\nR² = {r2:.3f}  |  RMSE = {rmse:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    print(f"{split_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    print(f"{split_name} plot saved to {filename}")


# Train predictions + plot
y_train_pred = mlp.predict(X_train_scaled)
plot_actual_vs_pred(y_train, y_train_pred, "Train",
                    os.path.join(FIGURES_DIR, "train_actual_vs_pred.png"))

# Test predictions + plot
y_test_pred = mlp.predict(X_test_scaled)
plot_actual_vs_pred(y_test, y_test_pred, "Test",
                    os.path.join(FIGURES_DIR, "test_actual_vs_pred.png"))
