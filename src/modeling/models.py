import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Tuple, List, Any

def prepare_data_for_model(df: pd.DataFrame, target_shift_days: int = 1, feature_columns: List[str] = None, target_column: str = None, price_change_threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les données pour l'entraînement du modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les données avec au moins une colonne 'close'
                           et les colonnes de features.
        target_shift_days (int): Nombre de jours dans le futur pour prédire le changement de prix.
        feature_columns (List[str], optional): Liste des colonnes à utiliser comme features.
                                                Si None, utilise ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14'].
        target_column (str, optional): Nom de la colonne cible si elle existe déjà. 
                                       Si None, crée une target basée sur close.
        price_change_threshold (float): Seuil de changement de prix pour créer la target (défaut: 2%).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Un tuple contenant X (features) et y (target).
    """
    df_copy = df.copy()
    
    # Création de la variable cible si elle n'existe pas déjà
    if target_column is None:
        if 'close' not in df_copy.columns:
            raise ValueError("La colonne 'close' est requise dans le DataFrame lorsque target_column n'est pas spécifié.")
        df_copy['future_close'] = df_copy['close'].shift(-target_shift_days)
        # Utiliser un seuil de changement de prix pour créer une target plus discriminante
        df_copy['price_change'] = (df_copy['future_close'] - df_copy['close']) / df_copy['close']
        df_copy['target'] = (df_copy['price_change'] > price_change_threshold).astype(int)
        target_column = 'target'
        
        # Afficher la distribution de la target pour diagnostiquer les déséquilibres
        target_dist = df_copy['target'].value_counts()
        print(f"Distribution de la target (seuil: {price_change_threshold*100:.1f}%):")
        print(f"  Classe 0 (pas de hausse significative): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_copy)*100:.1f}%)")
        print(f"  Classe 1 (hausse significative): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_copy)*100:.1f}%)")
        
    elif target_column not in df_copy.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")

    # Sélection des features
    if feature_columns is None:
        # S'assurer que ces colonnes existent, sinon les ignorer ou lever une erreur
        default_features = ['sma_10', 'sma_20', 'ema_10', 'ema_20', 'rsi_14']
        # Garder seulement les features présentes dans le dataframe (en minuscules/majuscules)
        lowercase_cols = {col.lower(): col for col in df_copy.columns}
        feature_columns = [lowercase_cols.get(f.lower()) for f in default_features if f.lower() in lowercase_cols]
        if not feature_columns:
            raise ValueError("Aucune des features par défaut n'a été trouvée dans le DataFrame. Veuillez spécifier feature_columns.")

    # Vérifier si toutes les feature_columns spécifiées existent
    missing_features = [col for col in feature_columns if col not in df_copy.columns]
    if missing_features:
        raise ValueError(f"Les colonnes de features suivantes sont manquantes : {missing_features}")

    df_copy = df_copy[feature_columns + [target_column]].copy() # Copie pour éviter SettingWithCopyWarning

    # Suppression des lignes avec NaN (introduits par shift ou calculs de features)
    df_copy.dropna(inplace=True)

    X = df_copy[feature_columns]
    y = df_copy[target_column]

    return X, y

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic_regression', model_path: str = 'models_store/model.joblib', test_size: float = 0.2, random_state: int = 42, scale_features: bool = True) -> dict:
    """
    Entraîne un modèle de machine learning et le sauvegarde.

    Args:
        X (pd.DataFrame): DataFrame des features.
        y (pd.Series): Series de la variable cible.
        model_type (str): Type de modèle à entraîner ('logistic_regression' ou 'random_forest').
        model_path (str): Chemin pour sauvegarder le modèle entraîné.
        test_size (float): Proportion de l'ensemble de données à allouer à l'ensemble de test.
        random_state (int): Graine pour la reproductibilité du train_test_split.
        scale_features (bool): Si True, standardise les features pour la régression logistique.

    Returns:
        dict: Un dictionnaire contenant les métriques du modèle (par exemple, accuracy).
    """
    print(f"Entraînement du modèle {model_type} avec {len(X)} échantillons et {len(X.columns)} features")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False) # shuffle=False pour les séries temporelles

    # Standardisation des features si demandé
    scaler = None
    if scale_features and model_type == 'logistic_regression':
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        print("Features standardisées pour la régression logistique")

    # Calcul des poids de classe pour gérer le déséquilibre
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Poids des classes calculés: {class_weight_dict}")

    if model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=random_state, 
            solver='liblinear', 
            class_weight=class_weight_dict,
            C=1.0,  # Moins de régularisation pour permettre plus de discrimination avec plus de features
            max_iter=1000
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            random_state=random_state, 
            n_estimators=100,
            class_weight=class_weight_dict,
            max_depth=10,
            min_samples_split=10
        )
    else:
        raise ValueError(f"Type de modèle non supporté : {model_type}. Choisissez 'logistic_regression' ou 'random_forest'.")

    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle et le scaler ensemble
    model_data = {'model': model, 'scaler': scaler, 'feature_columns': list(X.columns)}
    joblib.dump(model_data, model_path)

    # Évaluation sur l'ensemble de test
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, predictions)
    
    # Afficher des métriques détaillées
    print(f"Modèle '{model_type}' entraîné et sauvegardé dans '{model_path}'.")
    print(f"Accuracy sur l'ensemble de test : {accuracy:.4f}")
    
    if probabilities is not None:
        auc_score = roc_auc_score(y_test, probabilities)
        print(f"AUC Score : {auc_score:.4f}")
        print(f"Plage des probabilités prédites : [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        
        # Distribution des probabilités
        prob_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print("Distribution des probabilités prédites:")
        for low, high in prob_ranges:
            count = np.sum((probabilities >= low) & (probabilities < high))
            print(f"  [{low:.1f}-{high:.1f}): {count} ({count/len(probabilities)*100:.1f}%)")
    
    print("\nRapport de classification:")
    print(classification_report(y_test, predictions))

    return {'accuracy': accuracy, 'auc': auc_score if probabilities is not None else None}

def load_model_and_predict(X_new: pd.DataFrame, model_path: str = 'models_store/model.joblib', return_probabilities: bool = True) -> np.ndarray:
    """
    Charge un modèle sauvegardé et fait des prédictions.

    Args:
        X_new (pd.DataFrame): Nouvelles données (features) pour lesquelles faire des prédictions.
        model_path (str): Chemin vers le modèle sauvegardé.
        return_probabilities (bool): Si True, retourne les probabilités pour la classe positive.
                                   Si False, retourne les classes prédites.

    Returns:
        np.ndarray: Tableau NumPy des prédictions (probabilités ou classes).
    """
    try:
        model_data = joblib.load(model_path)
        
        # Vérifier si c'est le nouveau format (avec scaler) ou l'ancien format
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            scaler = model_data.get('scaler')
            expected_features = model_data.get('feature_columns')
        else:
            # Ancien format - juste le modèle
            model = model_data
            scaler = None
            expected_features = None
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Aucun modèle trouvé à l'emplacement : {model_path}. Veuillez d'abord entraîner et sauvegarder un modèle.")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle depuis {model_path}: {e}")

    # Vérifier les features si disponibles
    if expected_features is not None:
        missing_features = [f for f in expected_features if f not in X_new.columns]
        if missing_features:
            print(f"Attention: Features manquantes lors de la prédiction: {missing_features}")

    # Appliquer la standardisation si un scaler est disponible
    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
    else:
        X_scaled = X_new

    if return_probabilities and hasattr(model, 'predict_proba'):
        # Retourner les probabilités pour la classe positive (classe 1)
        probabilities = model.predict_proba(X_scaled)
        if probabilities.shape[1] == 2:  # Classification binaire
            predictions = probabilities[:, 1]  # Probabilité de la classe 1
        else:
            predictions = probabilities
    else:
        # Retourner les classes prédites
        predictions = model.predict(X_scaled)
    
    return predictions

if __name__ == '__main__':
    # Exemple d'utilisation (à des fins de test)
    # Créer un DataFrame de démonstration
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                      '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                      '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15']),
        'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 108, 109, 111, 112],
        'SMA_10': [np.nan]*2 + list(np.random.rand(13)*10 + 100), # Simuler des features
        'SMA_30': [np.nan]*5 + list(np.random.rand(10)*10 + 100),
        'EMA_10': [np.nan]*2 + list(np.random.rand(13)*10 + 100),
        'EMA_30': [np.nan]*5 + list(np.random.rand(10)*10 + 100),
        'RSI_14': [np.nan]*3 + list(np.random.rand(12)*50 + 25),
        'other_feature': np.random.rand(15) * 5
    }
    sample_df = pd.DataFrame(data)
    sample_df.set_index('timestamp', inplace=True)

    print("DataFrame initial:")
    print(sample_df.head())

    # 1. Préparer les données
    try:
        # Utiliser seulement les features qui existent et qui sont pertinentes
        # Pour le test, on s'assure que les features par défaut sont présentes ou on les spécifie
        available_features = ['SMA_10', 'EMA_10', 'RSI_14', 'other_feature'] # Supposons que ce sont nos features calculées
        # Filtrer pour ne garder que celles présentes dans sample_df
        actual_features_for_model = [f for f in available_features if f in sample_df.columns]

        X_prepared, y_prepared = prepare_data_for_model(sample_df.copy(), target_shift_days=1, feature_columns=actual_features_for_model)
        print("\nDonnées préparées (X):")
        print(X_prepared.head())
        print("\nDonnées préparées (y):")
        print(y_prepared.head())

        if not X_prepared.empty:
            # 2. Entraîner le modèle
            model_save_path = 'models_store/test_logistic_model.joblib'
            # Créer le dossier models_store s'il n'existe pas (joblib le fait aussi mais pour être sûr)
            import os
            os.makedirs('models_store', exist_ok=True)

            metrics = train_model(X_prepared, y_prepared, model_type='logistic_regression', model_path=model_save_path)
            print(f"\nMétriques d'entraînement : {metrics}")

            # 3. Charger le modèle et prédire
            # Simuler de nouvelles données (on pourrait prendre une partie de X_prepared pour le test)
            if len(X_prepared) > 5:
                X_new_sample = X_prepared.tail(5).copy()
                predictions_new = load_model_and_predict(X_new_sample, model_path=model_save_path)
                print(f"\nPrédictions sur de nouvelles données (dernières 5 lignes de X_prepared) :\n{predictions_new}")
            else:
                print("\nPas assez de données après préparation pour tester la prédiction.")

        else:
            print("\nPas assez de données après préparation pour entraîner un modèle.")

    except ValueError as ve:
        print(f"\nErreur de valeur lors de la préparation des données : {ve}")
    except FileNotFoundError as fnfe:
        print(f"\nErreur de fichier non trouvé : {fnfe}")
    except Exception as e:
        print(f"\nUne erreur est survenue : {e}")