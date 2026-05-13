"""
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- NUEVAS IMPORTACIONES PARA EL GRÁFICO INTERACTIVO ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        verbose=False,
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

    print("\n" + "=" * 70)
    print("PASO 7: TABLAS DE ACIERTOS Y ERRORES")
    print("=" * 70)
    
    # Crear tablas separadas
    df_xgb = pd.DataFrame({'Real': y_test, 'Predicción': y_pred_xgb})
    df_xgb['Resultado'] = np.where(df_xgb['Real'] == df_xgb['Predicción'], '✓ Acierto', '✗ Error')
    
    df_mlp = pd.DataFrame({'Real': y_test, 'Predicción': y_pred_mlp})
    df_mlp['Resultado'] = np.where(df_mlp['Real'] == df_mlp['Predicción'], '✓ Acierto', '✗ Error')

    # Guardar en CSV completos
    df_xgb.to_csv("tabla_errores_xgboost.csv", index=False)
    df_mlp.to_csv("tabla_errores_mlp.csv", index=False)
    
    print("\n--- Muestra Tabla XGBoost (Primeros 10) ---")
    print(df_xgb.head(10).to_string(index=False))
    print("\n--- Muestra Tabla MLP (Primeros 10) ---")
    print(df_mlp.head(10).to_string(index=False))
    print("\n✓ Se han guardado 'tabla_errores_xgboost.csv' y 'tabla_errores_mlp.csv' con todos los registros.")

    print("\n" + "=" * 70)
    print("PASO 8: GENERANDO VISUALIZACIONES ESTÁTICAS")
    print("=" * 70)

    # Gráficos de barras y matrices de confusión (Estáticos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["XGBoost", "Red Neuronal MLP"]
    accuracies = [xgb_accuracy, accuracy_mlp]
    colores_barras = ["#1f77b4", "#ff7f0e"]
    axes[0].bar(models, accuracies, color=colores_barras, alpha=0.8, edgecolor="black", linewidth=2)
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
    print("PASO 9: GENERANDO GRÁFICO PCA 3D INTERACTIVO (PLOTLY)")
    print("=" * 70)

    pca = PCA(n_components=3)
    X_test_pca = pca.fit_transform(X_test_selected)

    # Crear figura de Plotly con 2 subplots 3D interactivos
    fig_plotly = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("XGBoost: Aciertos y Errores (3D)", "MLP: Aciertos y Errores (3D)")
    )

    colores_clases = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Azul, Naranja, Verde, Rojo

    for i, (nombre_clase, color) in enumerate(zip(TARGET_NAMES, colores_clases)):
        # --- Datos XGBoost ---
        idx_clase = (y_test == i)
        idx_aciertos_xgb = idx_clase & (y_test == y_pred_xgb)
        idx_errores_xgb = idx_clase & (y_test != y_pred_xgb)

        # Aciertos XGBoost (Círculos)
        if np.any(idx_aciertos_xgb):
            fig_plotly.add_trace(go.Scatter3d(
                x=X_test_pca[idx_aciertos_xgb, 0], y=X_test_pca[idx_aciertos_xgb, 1], z=X_test_pca[idx_aciertos_xgb, 2],
                mode='markers', marker=dict(size=4, color=color, symbol='circle', opacity=0.6),
                name=f'✓ {nombre_clase}', legendgroup=f'acierto_{i}'
            ), row=1, col=1)

        # Errores XGBoost (Cruces rojas para el error, borde del color original)
        if np.any(idx_errores_xgb):
            fig_plotly.add_trace(go.Scatter3d(
                x=X_test_pca[idx_errores_xgb, 0], y=X_test_pca[idx_errores_xgb, 1], z=X_test_pca[idx_errores_xgb, 2],
                mode='markers', marker=dict(size=8, color=color, symbol='x', opacity=1.0, line=dict(color='black', width=2)),
                name=f'✗ Error (Era {nombre_clase})', legendgroup=f'error_{i}'
            ), row=1, col=1)

        # --- Datos MLP ---
        idx_aciertos_mlp = idx_clase & (y_test == y_pred_mlp)
        idx_errores_mlp = idx_clase & (y_test != y_pred_mlp)

        # Aciertos MLP (Círculos)
        if np.any(idx_aciertos_mlp):
            fig_plotly.add_trace(go.Scatter3d(
                x=X_test_pca[idx_aciertos_mlp, 0], y=X_test_pca[idx_aciertos_mlp, 1], z=X_test_pca[idx_aciertos_mlp, 2],
                mode='markers', marker=dict(size=4, color=color, symbol='circle', opacity=0.6),
                name=f'✓ {nombre_clase}', legendgroup=f'acierto_{i}', showlegend=False # Ocultar en la leyenda para no duplicar
            ), row=1, col=2)

        # Errores MLP (Cruces rojas)
        if np.any(idx_errores_mlp):
            fig_plotly.add_trace(go.Scatter3d(
                x=X_test_pca[idx_errores_mlp, 0], y=X_test_pca[idx_errores_mlp, 1], z=X_test_pca[idx_errores_mlp, 2],
                mode='markers', marker=dict(size=8, color=color, symbol='x', opacity=1.0, line=dict(color='black', width=2)),
                name=f'✗ Error (Era {nombre_clase})', legendgroup=f'error_{i}', showlegend=False
            ), row=1, col=2)

    # Configuración de la interfaz web
    fig_plotly.update_layout(
        title_text="Análisis PCA 3D Interactivo (Rotar, Zoom y Hover)",
        title_x=0.5,
        height=800,
        width=1500,
        scene=dict(xaxis_title='PC 1', yaxis_title='PC 2', zaxis_title='PC 3'),
        scene2=dict(xaxis_title='PC 1', yaxis_title='PC 2', zaxis_title='PC 3'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Guardar como HTML interactivo y ABRIR automáticamente
    html_file = "pca_interactivo_modelos.html"
    fig_plotly.write_html(html_file, auto_open=True)
    print(f"✓ ¡ÉXITO! Se ha abierto el archivo interactivo '{html_file}' en tu navegador web.")


    print("\n" + "=" * 70)
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    winner_name, winner_score = max({"XGBoost": xgb_accuracy, "Red Neuronal MLP": accuracy_mlp}.items(), key=lambda item: item[1])
    print(f"\n📊 GANADOR: {winner_name} con {winner_score:.4f} accuracy")
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  • rfecv_selection_mlp.png")
    print("  • comparison_mlp.png")
    print("  • pca_interactivo_modelos.html (NUEVO - Se abre en tu Chrome/Edge)")
    print("  • tabla_errores_xgboost.csv")
    print("  • tabla_errores_mlp.csv")


def train_cnn(X_train_selected: np.ndarray, y_train: np.ndarray, X_test_selected: np.ndarray, y_test: np.ndarray, xgb_accuracy: float, y_pred_xgb: np.ndarray) -> None:
    # ... (El código de tu CNN sigue intacto) ...
    pass

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
        # Aquí llamarías a train_cnn si es necesario
        pass

if __name__ == "__main__":
    main()