r"""
RFECV + XGBoost + modelo neuronal seleccionable.

Uso:
  .venv\Scripts\python.exe RFECV.py mlp
  .venv\Scripts\python.exe RFECV.py cnn

Modo por defecto: mlp
"""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TARGET_NAMES = ["Healthy", "Generalized", "Focal", "Seizure"]
FEATURE_NAMES = [f"X{i + 1}" for i in range(16)]


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("BEED_Data.csv")
    print("=" * 70)
    print("PASO 1: CARGANDO DATOS DE EEG")
    print("=" * 70)
    print(f"Forma del dataset: {df.shape}")
    print(f"Clases: {df['y'].unique()}")
    print(f"Distribución de clases:\n{df['y'].value_counts().sort_index()}\n")
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y


def split_and_scale(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("=" * 70)
    print("PASO 2: DIVIDIENDO DATOS (80% train, 20% test)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def plot_rfecv(rfecv: RFECV, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mean_scores = rfecv.cv_results_["mean_test_score"]
    axes[0].plot(range(1, len(mean_scores) + 1), mean_scores, linewidth=2, marker="o")
    axes[0].set_xlabel("Número de Características")
    axes[0].set_ylabel("CV Accuracy")
    axes[0].set_title("RFECV: Selección Óptima de Características")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=rfecv.n_features_, color="r", linestyle="--", label="Óptimo", linewidth=2)
    axes[0].legend()

    ranking_df = pd.DataFrame({"Característica": FEATURE_NAMES, "Ranking": rfecv.ranking_}).sort_values(
        "Ranking"
    )
    colors = ["green" if ranking == 1 else "lightcoral" for ranking in ranking_df["Ranking"]]
    axes[1].barh(ranking_df["Característica"], ranking_df["Ranking"], color=colors)
    axes[1].set_xlabel("Ranking RFECV")
    axes[1].set_title("Ranking de Características (1=Seleccionada)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Gráfico guardado: {output_path}")


def run_rfecv_xgb(X_train_scaled: np.ndarray, y_train: np.ndarray, X_test_scaled: np.ndarray, y_test: np.ndarray, output_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    print("=" * 70)
    print("PASO 3: RFECV - SELECCIONAR CARACTERÍSTICAS CON XGBOOST")
    print("=" * 70)
    print("Ejecutando RFECV (puede tomar 2-3 minutos)...\n")

    estimator = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    rfecv.fit(X_train_scaled, y_train)

    selected_mask = rfecv.support_
    selected_features = [FEATURE_NAMES[i] for i, selected in enumerate(selected_mask) if selected]

    print(f"\n✓ Características seleccionadas: {len(selected_features)}/{len(FEATURE_NAMES)}")
    print(f"  Características: {selected_features}")
    print(f"  Ranking RFECV: {rfecv.ranking_}\n")

    plot_rfecv(rfecv, f"rfecv_selection_{output_prefix}.png")

    X_train_selected = X_train_scaled[:, selected_mask]
    X_test_selected = X_test_scaled[:, selected_mask]

    print("=" * 70)
    print("PASO 4: ENTRENAR XGBOOST")
    print("=" * 70)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train_selected, y_train)
    y_pred_xgb = xgb_model.predict(X_test_selected)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

    print(f"\n✓ XGBoost Accuracy: {accuracy_xgb:.4f}\n")
    print("Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb, target_names=TARGET_NAMES, zero_division=0))

    return selected_mask, X_train_selected, X_test_selected, accuracy_xgb, y_pred_xgb


def train_mlp(X_train_selected: np.ndarray, y_train: np.ndarray, X_test_selected: np.ndarray, y_test: np.ndarray, xgb_accuracy: float, y_pred_xgb: np.ndarray) -> None:
    print("=" * 70)
    print("PASO 5: ENTRENAR RED NEURONAL (MLP)")
    print("=" * 70)
    print("Entrenando MLP (puede tomar 1-2 minutos)...\n")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        verbose=True,
    )
    mlp.fit(X_train_selected, y_train)
    y_pred_mlp = mlp.predict(X_test_selected)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

    print(f"\n✓ Red Neuronal (MLP) Accuracy: {accuracy_mlp:.4f}\n")
    print("Classification Report (Red Neuronal MLP):")
    print(classification_report(y_test, y_pred_mlp, target_names=TARGET_NAMES, zero_division=0))

    print("=" * 70)
    print("PASO 6: COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(f"\n{'Modelo':<25} {'Accuracy':<15} {'Diferencia':<20}")
    print("-" * 60)
    for model_name, accuracy in sorted({"XGBoost": xgb_accuracy, "Red Neuronal MLP": accuracy_mlp}.items(), key=lambda item: item[1], reverse=True):
        diff = f"{((accuracy / xgb_accuracy) - 1) * 100:+.2f}%" if model_name != "XGBoost" else "+0.00%"
        print(f"{model_name:<25} {accuracy:<15.4f} {diff:<20}")

    print("=" * 70)
    print("PASO 7: GENERANDO VISUALIZACIONES")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["XGBoost", "Red Neuronal MLP"]
    accuracies = [xgb_accuracy, accuracy_mlp]
    colors = ["#1f77b4", "#ff7f0e"]
    axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Comparación: XGBoost vs MLP")
    axes[0].set_ylim([0.7, 1.0])
    axes[0].grid(axis="y", alpha=0.3)
    for index, accuracy in enumerate(accuracies):
        axes[0].text(index, accuracy + 0.01, f"{accuracy:.4f}", ha="center", va="bottom", fontweight="bold")

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False, square=True)
    axes[1].set_title(f"Matriz de Confusión - XGBoost\nAccuracy: {xgb_accuracy:.4f}")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predicción")

    sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Oranges", ax=axes[2], cbar=False, square=True)
    axes[2].set_title(f"Matriz de Confusión - MLP\nAccuracy: {accuracy_mlp:.4f}")
    axes[2].set_ylabel("Real")
    axes[2].set_xlabel("Predicción")

    plt.tight_layout()
    plt.savefig("comparison_mlp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Gráfico guardado: comparison_mlp.png")

    print("\n" + "=" * 70)
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    winner_name, winner_score = max({"XGBoost": xgb_accuracy, "Red Neuronal MLP": accuracy_mlp}.items(), key=lambda item: item[1])
    print(f"\n📊 GANADOR: {winner_name} con {winner_score:.4f} accuracy")
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  • rfecv_selection_mlp.png")
    print("  • comparison_mlp.png")


def train_cnn(X_train_selected: np.ndarray, y_train: np.ndarray, X_test_selected: np.ndarray, y_test: np.ndarray, xgb_accuracy: float, y_pred_xgb: np.ndarray) -> None:
    print("=" * 70)
    print("PASO 5: PREPARAR DATOS PARA CNN (Formato 3D)")
    print("=" * 70)
    print("Preparando datos para CNN 1D:\n")
    print(f"  Original: {X_train_selected.shape}")

    X_train_cnn = X_train_selected.reshape(X_train_selected.shape[0], 1, X_train_selected.shape[1])
    X_test_cnn = X_test_selected.reshape(X_test_selected.shape[0], 1, X_test_selected.shape[1])
    y_train_cnn = y_train
    y_test_cnn = y_test

    print(f"  CNN (train): {X_train_cnn.shape} (muestras, canales, features)")
    print(f"  CNN (test):  {X_test_cnn.shape}\n")
    print(f"Train CNN: {X_train_cnn.shape}")
    print(f"Val CNN:   {X_test_cnn.shape}\n")

    print("=" * 70)
    print("PASO 6: CONSTRUIR Y ENTRENAR CNN 1D")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch no está instalado. Ejecuta: .venv\\Scripts\\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu") from exc

    class EEG_CNN(nn.Module):
        def __init__(self, n_features: int, n_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_features)
                flattened = self.features(dummy).view(1, -1).shape[1]
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            return self.classifier(x)

    device = torch.device("cpu")
    n_classes = len(np.unique(y_train_cnn))
    n_features = X_train_cnn.shape[2]
    model = EEG_CNN(n_features=n_features, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nArquitectura CNN:")
    print(f"  Entrada real: (1, {n_features}) - canales × features")
    print("  Conv1D (1→32) → MaxPool")
    print("  Conv1D (32→64) → MaxPool")
    print("  Conv1D (64→128) → MaxPool")
    print(f"  FC: 256 → 128 → {n_classes} clases")
    print(f"\nDispositivo: {device}")
    print(f"Datos de entrenamiento: {X_train_cnn.shape}")
    print(f"Datos de validación:    {X_test_cnn.shape}")
    print("\nEntrenando CNN (aprox. 2-3 minutos)...\n")

    X_train_tensor = torch.FloatTensor(X_train_cnn)
    y_train_tensor = torch.LongTensor(y_train_cnn)
    X_test_tensor = torch.FloatTensor(X_test_cnn)
    y_test_tensor = torch.LongTensor(y_test_cnn)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    train_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = 100 * correct / total if total else 0.0
        train_losses.append(running_loss / len(train_loader))
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d}: Loss={running_loss / len(train_loader):.4f}, Val Acc={val_acc:.2f}%")

    model.eval()
    y_pred_cnn: list[int] = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_cnn.extend(predicted.cpu().numpy().tolist())

    y_pred_cnn_array = np.array(y_pred_cnn)
    accuracy_cnn = accuracy_score(y_test_cnn, y_pred_cnn_array)

    print(f"\n✓ CNN 1D Accuracy: {accuracy_cnn:.4f}\n")
    print("Classification Report (CNN 1D):")
    print(classification_report(y_test_cnn, y_pred_cnn_array, target_names=TARGET_NAMES, zero_division=0))

    print("=" * 70)
    print("PASO 7: COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(f"\n{'Modelo':<25} {'Accuracy':<15} {'Diferencia':<20}")
    print("-" * 60)
    for model_name, accuracy in sorted({"XGBoost": xgb_accuracy, "CNN 1D": accuracy_cnn}.items(), key=lambda item: item[1], reverse=True):
        diff = f"{((accuracy / xgb_accuracy) - 1) * 100:+.2f}%" if model_name != "XGBoost" else "+0.00%"
        print(f"{model_name:<25} {accuracy:<15.4f} {diff:<20}")

    print("=" * 70)
    print("PASO 8: GENERANDO VISUALIZACIONES")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["XGBoost", "CNN 1D"]
    accuracies = [xgb_accuracy, accuracy_cnn]
    colors = ["#1f77b4", "#ff7f0e"]
    axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Comparación: XGBoost vs CNN 1D")
    axes[0].set_ylim([0.7, 1.0])
    axes[0].grid(axis="y", alpha=0.3)
    for index, accuracy in enumerate(accuracies):
        axes[0].text(index, accuracy + 0.01, f"{accuracy:.4f}", ha="center", va="bottom", fontweight="bold")

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn_array)
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False, square=True)
    axes[1].set_title(f"Matriz de Confusión - XGBoost\nAccuracy: {xgb_accuracy:.4f}")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predicción")

    sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens", ax=axes[2], cbar=False, square=True)
    axes[2].set_title(f"Matriz de Confusión - CNN 1D\nAccuracy: {accuracy_cnn:.4f}")
    axes[2].set_ylabel("Real")
    axes[2].set_xlabel("Predicción")

    plt.tight_layout()
    plt.savefig("comparison_cnn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Gráfico guardado: comparison_cnn.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, marker="o", label="Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN: Pérdida de Entrenamiento")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(val_accuracies, marker="o", label="Validation Accuracy", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("CNN: Accuracy de Validación")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_training_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Gráfico guardado: cnn_training_history.png")

    print("\n" + "=" * 70)
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    winner_name, winner_score = max({"XGBoost": xgb_accuracy, "CNN 1D": accuracy_cnn}.items(), key=lambda item: item[1])
    print(f"\n📊 GANADOR: {winner_name} con {winner_score:.4f} accuracy")
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  • rfecv_selection_cnn.png")
    print("  • comparison_cnn.png")
    print("  • cnn_training_history.png")


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "mlp"
    if mode not in {"mlp", "cnn"}:
        raise SystemExit("Modo inválido. Usa: mlp o cnn")

    X, y = load_dataset()
    _, _, y_train, y_test, X_train_scaled, X_test_scaled = split_and_scale(X, y)
    _, X_train_selected, X_test_selected, xgb_accuracy, y_pred_xgb = run_rfecv_xgb(
        X_train_scaled, y_train, X_test_scaled, y_test, mode
    )

    if mode == "mlp":
        train_mlp(X_train_selected, y_train, X_test_selected, y_test, xgb_accuracy, y_pred_xgb)
    else:
        train_cnn(X_train_selected, y_train, X_test_selected, y_test, xgb_accuracy, y_pred_xgb)


if __name__ == "__main__":
    main()