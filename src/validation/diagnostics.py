import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error # Ou autre métrique pertinente
from sklearn.utils import shuffle as sk_shuffle
from typing import Any, List, Tuple, Optional, Callable, Dict

# Supposons que train_model et load_model_and_predict sont accessibles
# Cela pourrait nécessiter d'ajuster les imports en fonction de la structure du projet
# from ..modeling.models import train_model, load_model_and_predict
# Pour l'instant, on va supposer qu'un modèle fitté est passé directement ou via un chemin

def a_plotter_courbe_apprentissage(
    estimator: Any, 
    title: str, 
    X: pd.DataFrame, 
    y: pd.Series, 
    axes: Optional[plt.Axes] = None, 
    ylim: Optional[Tuple[float, float]] = None, 
    cv: Optional[Union[int, KFold, TimeSeriesSplit]] = None, 
    n_jobs: Optional[int] = None, 
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    scoring: Optional[str] = None, # ex: 'roc_auc', 'neg_mean_squared_error'
    is_time_series: bool = False,
    random_state: Optional[int] = None
):
    """
    Génère et affiche une courbe d'apprentissage pour un estimateur donné.

    Args:
        estimator: L'objet modèle à évaluer. Doit être compatible scikit-learn.
        title (str): Titre du graphique.
        X (pd.DataFrame): Ensemble des features.
        y (pd.Series): Ensemble de la cible.
        axes (plt.Axes, optional): Axes Matplotlib sur lesquels dessiner. Si None, de nouveaux axes sont créés.
        ylim (tuple, optional): Limites pour l'axe y.
        cv (int, cross-validation generator or an iterable, optional): Détermine la stratégie de validation croisée.
            Si None, utilise KFold(5) ou TimeSeriesSplit(5) si is_time_series=True.
        n_jobs (int, optional): Nombre de jobs à exécuter en parallèle.
        train_sizes (np.ndarray): Tailles relatives ou absolues des ensembles d'entraînement à utiliser.
        scoring (str, optional): Métrique de scoring à utiliser (ex: 'accuracy', 'roc_auc').
                                 Si None, utilise le score par défaut de l'estimateur.
        is_time_series (bool): Si True, utilise TimeSeriesSplit pour la CV par défaut.
        random_state (int, optional): Graine pour KFold si cv n'est pas spécifié et is_time_series=False.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Nombre d'échantillons d'entraînement")
    axes.set_ylabel("Score")

    if cv is None:
        if is_time_series:
            cv = TimeSeriesSplit(n_splits=5)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    train_sizes_abs, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        return_times=True,
        shuffle=not is_time_series, # Shuffle seulement si ce n'est pas une série temporelle
        random_state=random_state if not is_time_series else None
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1) # Non utilisé dans le plot principal pour l'instant
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot de la courbe d'apprentissage
    axes.grid()
    axes.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes_abs,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes_abs, train_scores_mean, "o-", color="r", label="Score d'entraînement"
    )
    axes.plot(
        train_sizes_abs, test_scores_mean, "o-", color="g", label="Score de validation croisée"
    )
    axes.legend(loc="best")
    
    print(f"Courbe d'apprentissage '{title}' générée.")
    return plt


def a_effectuer_test_permutation(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring_func: Callable, # Ex: roc_auc_score, lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    n_permutations: int = 100,
    cv_splitter: Optional[Union[KFold, TimeSeriesSplit]] = None, # Pour évaluer le modèle sur chaque permutation
    random_state: Optional[int] = None,
    fit_params: Optional[Dict[str, Any]] = None
) -> Tuple[float, np.ndarray, float]:
    """
    Effectue un test de permutation pour évaluer la significativité statistique
    de la performance d'un modèle.

    Args:
        estimator: Le modèle à évaluer (doit avoir les méthodes fit et predict/predict_proba).
        X (pd.DataFrame): Features.
        y (pd.Series): Cible.
        scoring_func (Callable): Fonction pour calculer le score (ex: roc_auc_score).
                                 Prend (y_true, y_pred_ou_proba) et retourne un score.
        n_permutations (int): Nombre de permutations à effectuer.
        cv_splitter (KFold ou TimeSeriesSplit, optional): Stratégie de CV pour évaluer sur chaque permutation.
                                                        Si None, évalue sur un simple train/test split (80/20).
        random_state (Optional[int]): Graine pour la reproductibilité des permutations et du split.
        fit_params (Optional[Dict[str, Any]]): Paramètres additionnels pour la méthode `fit` de l'estimateur.

    Returns:
        Tuple[float, np.ndarray, float]:
            - score_original (float): Score du modèle sur les données non permutées.
            - permutation_scores (np.ndarray): Scores obtenus sur les données permutées.
            - p_value (float): P-valeur estimée.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if fit_params is None:
        fit_params = {}

    # Évaluation sur les données originales
    if cv_splitter:
        original_scores_cv = []
        for train_idx, test_idx in cv_splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            estimator.fit(X_train, y_train, **fit_params)
            if hasattr(estimator, 'predict_proba') and callable(scoring_func) and "roc_auc" in scoring_func.__name__.lower() : # Heuristique pour AUC
                y_pred_val = estimator.predict_proba(X_test)[:, 1]
            else:
                y_pred_val = estimator.predict(X_test)
            original_scores_cv.append(scoring_func(y_test, y_pred_val))
        score_original = np.mean(original_scores_cv)
    else: # Simple train/test split pour le score original
        # Utiliser sk_shuffle pour être cohérent avec les permutations
        X_shuffled, y_shuffled = sk_shuffle(X, y, random_state=random_state)
        split_idx = int(0.8 * len(y_shuffled))
        X_train_orig, X_test_orig = X_shuffled.iloc[:split_idx], X_shuffled.iloc[split_idx:]
        y_train_orig, y_test_orig = y_shuffled.iloc[:split_idx], y_shuffled.iloc[split_idx:]
        
        estimator.fit(X_train_orig, y_train_orig, **fit_params)
        if hasattr(estimator, 'predict_proba') and "roc_auc" in scoring_func.__name__.lower():
            y_pred_orig = estimator.predict_proba(X_test_orig)[:, 1]
        else:
            y_pred_orig = estimator.predict(X_test_orig)
        score_original = scoring_func(y_test_orig, y_pred_orig)

    print(f"Score original du modèle: {score_original:.4f}")

    permutation_scores = np.zeros(n_permutations)
    print(f"Début des {n_permutations} permutations...")
    for i in range(n_permutations):
        y_permuted = sk_shuffle(y, random_state=random_state + i + 1) # Changer la graine pour chaque permutation
        
        if cv_splitter:
            current_perm_scores_cv = []
            for train_idx, test_idx in cv_splitter.split(X, y_permuted): # Split sur X, y_permuted
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                # Utiliser y_permuted pour l'entraînement et le test
                y_train_perm, y_test_perm = y_permuted.iloc[train_idx], y_permuted.iloc[test_idx]
                
                estimator.fit(X_train, y_train_perm, **fit_params)
                if hasattr(estimator, 'predict_proba') and "roc_auc" in scoring_func.__name__.lower():
                    y_pred_perm_val = estimator.predict_proba(X_test)[:, 1]
                else:
                    y_pred_perm_val = estimator.predict(X_test)
                current_perm_scores_cv.append(scoring_func(y_test_perm, y_pred_perm_val))
            permutation_scores[i] = np.mean(current_perm_scores_cv)
        else: # Simple train/test split pour chaque permutation
            X_shuffled_perm, y_shuffled_perm = sk_shuffle(X, y_permuted, random_state=random_state + i + 1)
            split_idx_perm = int(0.8 * len(y_shuffled_perm))
            X_train_perm, X_test_perm = X_shuffled_perm.iloc[:split_idx_perm], X_shuffled_perm.iloc[split_idx_perm:]
            y_train_perm_set, y_test_perm_set = y_shuffled_perm.iloc[:split_idx_perm], y_shuffled_perm.iloc[split_idx_perm:]

            estimator.fit(X_train_perm, y_train_perm_set, **fit_params)
            if hasattr(estimator, 'predict_proba') and "roc_auc" in scoring_func.__name__.lower():
                y_pred_perm = estimator.predict_proba(X_test_perm)[:, 1]
            else:
                y_pred_perm = estimator.predict(X_test_perm)
            permutation_scores[i] = scoring_func(y_test_perm_set, y_pred_perm)
        
        if (i + 1) % (n_permutations // 10) == 0:
            print(f"  Permutation {i+1}/{n_permutations} terminée. Score: {permutation_scores[i]:.4f}")

    p_value = (np.sum(permutation_scores >= score_original) + 1.0) / (n_permutations + 1.0)
    print(f"Test de permutation terminé. P-valeur: {p_value:.4f}")
    
    # Plot des scores de permutation
    plt.figure(figsize=(10, 6))
    plt.hist(permutation_scores, bins=20, label='Scores de permutation', edgecolor='black', alpha=0.7)
    plt.axvline(score_original, color='r', linestyle='--', linewidth=2, label=f'Score original ({score_original:.4f})')
    plt.xlabel("Score du modèle")
    plt.ylabel("Fréquence")
    plt.title(f"Distribution des scores de permutation (P-valeur: {p_value:.4f})")
    plt.legend()
    plt.grid(True)
    
    return score_original, permutation_scores, p_value


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression # Pour un exemple de régression

    # --- Exemple pour la courbe d'apprentissage (Classification) ---
    print("\n--- Test Courbe d'Apprentissage (Classification) ---")
    X_clf_sample = pd.DataFrame(np.random.rand(200, 5), columns=[f'f{i}' for i in range(5)])
    y_clf_sample = pd.Series(np.random.randint(0, 2, 200))
    
    estimator_clf = RandomForestClassifier(n_estimators=20, random_state=42)
    
    # Utilisation de KFold pour la classification
    cv_clf = KFold(n_splits=3, shuffle=True, random_state=42)
    fig_lc, ax_lc = plt.subplots(1, 1, figsize=(10,6))
    a_plotter_courbe_apprentissage(
        estimator_clf, 
        "Courbe d'apprentissage RandomForestClassifier", 
        X_clf_sample, 
        y_clf_sample, 
        axes=ax_lc,
        cv=cv_clf, 
        n_jobs=1, 
        scoring='roc_auc',
        random_state=42
    )
    # plt.show() # Décommenter pour afficher interactivement

    # --- Exemple pour le test de permutation (Classification) ---
    print("\n--- Test de Permutation (Classification) ---")
    # Réduire la taille de l'échantillon et le nombre de permutations pour un test rapide
    X_perm_clf_sample = pd.DataFrame(np.random.rand(100, 3), columns=[f'f{i}' for i in range(3)])
    y_perm_clf_sample = pd.Series(np.random.randint(0, 2, 100))
    
    estimator_perm_clf = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
    
    # Utilisation de KFold pour le test de permutation
    cv_perm_clf = KFold(n_splits=3, shuffle=True, random_state=42)

    score_orig_clf, scores_perm_clf, p_value_clf = a_effectuer_test_permutation(
        estimator_perm_clf,
        X_perm_clf_sample,
        y_perm_clf_sample,
        scoring_func=roc_auc_score,
        n_permutations=20, # Réduit pour l'exemple
        cv_splitter=cv_perm_clf, # Utilisation de CV
        random_state=42
    )
    print(f"Score original (AUC): {score_orig_clf:.4f}")
    print(f"P-valeur du test de permutation: {p_value_clf:.4f}")
    # plt.show() # Décommenter pour afficher interactivement

    # --- Exemple pour le test de permutation (Régression) ---
    print("\n--- Test de Permutation (Régression) ---")
    X_perm_reg_sample = pd.DataFrame(np.random.rand(100, 3), columns=[f'f{i}' for i in range(3)])
    y_perm_reg_sample = pd.Series(np.random.rand(100) * 10) # Cible continue
    
    estimator_perm_reg = LinearRegression()
    
    # Fonction de score pour la régression (ex: MSE négatif car on maximise souvent)
    def neg_mse_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    cv_perm_reg = KFold(n_splits=3, shuffle=True, random_state=42)
    score_orig_reg, scores_perm_reg, p_value_reg = a_effectuer_test_permutation(
        estimator_perm_reg,
        X_perm_reg_sample,
        y_perm_reg_sample,
        scoring_func=neg_mse_scorer,
        n_permutations=20, # Réduit pour l'exemple
        cv_splitter=cv_perm_reg,
        random_state=42
    )
    print(f"Score original (Neg MSE): {score_orig_reg:.4f}")
    print(f"P-valeur du test de permutation (Régression): {p_value_reg:.4f}")
    
    # Afficher tous les graphiques à la fin si exécuté en script
    plt.show()