import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression # Exemple de méta-modèle
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import tempfile
import shutil

# Importation depuis models.py pour instancier/entraîner les modèles de base
from .models import train_model, load_model_and_predict

# Note: prepare_data_for_model est supposé être appelé en amont.

class BaseModelsManager:
    """
    Niveau 1: Gère l'instanciation, l'entraînement et la prédiction
    des modèles de base spécialisés.
    Cette classe est conçue pour générer des prédictions out-of-fold (OOF)
    pour l'entraînement du méta-modèle.
    """
    def __init__(self, base_models_configs: List[Dict[str, Any]]):
        """
        Initialise le gestionnaire de modèles de base.

        Args:
            base_models_configs (List[Dict[str, Any]]): Liste de configurations pour chaque modèle de base.
                Chaque dictionnaire doit contenir au moins 'model_type' et 'model_params'.
                Il peut aussi contenir 'model_path_prefix' pour nommer les modèles sauvegardés.
                Exemple: [{'model_type': 'random_forest', 'model_params': {'n_estimators': 100}, 'model_path_prefix': 'rf_model'}]
        """
        self.base_models_configs = base_models_configs
        # self.fitted_base_models = [] # Remplacé par la sauvegarde sur disque
        self.trained_full_data_model_paths: List[str] = [] # Stockera les chemins des modèles entraînés sur toutes les données
        self.temp_dir_for_oof = tempfile.mkdtemp(prefix="oof_models_")
        self.base_model_feature_importances_oof: List[Optional[Dict[str, float]]] = [] # Pour stocker les importances des features OOF
        self.base_model_feature_importances_full: List[Optional[Dict[str, float]]] = [] # Pour stocker les importances des features des modèles complets


    def _cleanup_temp_dir(self):
        """Nettoie le répertoire temporaire utilisé pour les modèles OOF."""
        if hasattr(self, 'temp_dir_for_oof') and os.path.exists(self.temp_dir_for_oof):
            shutil.rmtree(self.temp_dir_for_oof)
            print(f"Répertoire temporaire {self.temp_dir_for_oof} supprimé.")

    def __del__(self):
        """S'assure que le répertoire temporaire est nettoyé lors de la suppression de l'objet."""
        self._cleanup_temp_dir()

    def fit_predict_oof(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: Optional[int] = None, is_time_series: bool = False, scale_features_for_oof: bool = False) -> pd.DataFrame:
        """
        Entraîne les modèles de base en utilisant une validation croisée et retourne les prédictions out-of-fold.
        Les modèles de chaque fold sont sauvegardés temporairement.

        Args:
            X (pd.DataFrame): Features d'entraînement.
            y (pd.Series): Cible d'entraînement.
            n_splits (int): Nombre de folds pour la validation croisée.
            random_state (Optional[int]): Graine aléatoire pour la reproductibilité des folds (si KFold).
            is_time_series (bool): Si True, utilise TimeSeriesSplit, sinon KFold.

        Returns:
            pd.DataFrame: Prédictions out-of-fold des modèles de base. Chaque colonne correspond à un modèle.
            scale_features_for_oof (bool): Si True, active la standardisation des features pour l'entraînement des modèles OOF.
                                           Cela sera passé à `train_model`.
        """
        if is_time_series:
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        oof_predictions = np.zeros((len(X), len(self.base_models_configs)))
        # S'assurer que le répertoire temporaire existe
        os.makedirs(self.temp_dir_for_oof, exist_ok=True)

        for model_idx, model_config in enumerate(self.base_models_configs):
            model_type = model_config.get('model_type', 'unknown_type')
            model_params = model_config.get('model_params', {})
            print(f"Génération des prédictions OOF pour le modèle de base {model_idx + 1}/{len(self.base_models_configs)}: {model_type}")
            
            # Pour OOF, l'importance des features est moins directe à agréger car elle varie par fold.
            # On pourrait stocker l'importance du premier fold à titre indicatif, ou une moyenne si pertinent.
            # Pour l'instant, on ne stocke pas l'importance des features des modèles OOF.
            # self.base_model_feature_importances_oof.append(None) # Initialiser pour ce modèle

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, _ = y.iloc[train_idx], y.iloc[val_idx] # y_val n'est pas utilisé pour l'entraînement du modèle de base

                temp_model_filename = f"model_{model_idx}_fold_{fold_idx}.joblib"
                temp_model_path = os.path.join(self.temp_dir_for_oof, temp_model_filename)

                print(f"  Fold {fold_idx + 1}/{n_splits} - Entraînement du modèle {model_type}...")
                # train_model retourne maintenant un dict incluant 'feature_importances'
                training_results_fold = train_model(
                    X_train, y_train,
                    model_type=model_type,
                    model_params=model_params.copy(), # Copie pour éviter modifications inattendues
                    model_path=temp_model_path,
                    test_size=0.01, # Non utilisé car nous fournissons X_train, y_train directement
                    random_state=random_state, # Pour la reproductibilité interne de train_model si applicable
                    scale_features=scale_features_for_oof
                )
                
                # Optionnel: stocker l'importance des features du premier fold pour ce modèle OOF
                # if fold_idx == 0 and training_results_fold.get('feature_importances'):
                #     self.base_model_feature_importances_oof[model_idx] = training_results_fold['feature_importances']

                # Prédire sur l'ensemble de validation du fold
                # load_model_and_predict retourne des probabilités si applicable et return_probabilities=True (par défaut)
                try:
                    fold_preds = load_model_and_predict(X_val, model_path=temp_model_path, return_probabilities=True)
                    # S'assurer que fold_preds est 1D (probabilité de la classe positive)
                    if fold_preds.ndim > 1 and fold_preds.shape[1] >= 2: # Cas (N, 2) ou plus pour multiclasse
                        fold_preds = fold_preds[:, 1] # Supposons la proba de la classe 1
                    elif fold_preds.ndim > 1 and fold_preds.shape[1] == 1: # Cas (N,1)
                         fold_preds = fold_preds.ravel()

                    oof_predictions[val_idx, model_idx] = fold_preds

                except Exception as e:
                    print(f"  Erreur lors de la prédiction pour le fold {fold_idx+1} du modèle {model_type}: {e}")
                    oof_predictions[val_idx, model_idx] = np.nan # Marquer comme NaN en cas d'erreur
                finally:
                    # Optionnel: supprimer le modèle temporaire du fold immédiatement si l'espace est critique
                    # if os.path.exists(temp_model_path):
                    #     os.remove(temp_model_path)
                    pass # Le nettoyage se fera via _cleanup_temp_dir

        oof_df = pd.DataFrame(oof_predictions, index=X.index, columns=[f"base_model_{i}_{self.base_models_configs[i].get('model_type', 'unknown')}" for i in range(len(self.base_models_configs))])
        return oof_df

    def fit_base_models_on_full_data(self, X: pd.DataFrame, y: pd.Series, models_dir: str = "trained_ensemble_base_models", scale_features_full: bool = False):
        """
        Entraîne chaque modèle de base sur l'ensemble des données X et y et sauvegarde les modèles.
        Ces modèles seront utilisés pour les prédictions sur de nouvelles données (non OOF).

        Args:
            X (pd.DataFrame): Features d'entraînement.
            y (pd.Series): Cible d'entraînement.
            models_dir (str): Répertoire où sauvegarder les modèles de base entraînés sur toutes les données.
            scale_features_full (bool): Si True, active la standardisation pour l'entraînement final.
        """
        self.trained_full_data_model_paths = []
        self.base_model_feature_importances_full = [] # Réinitialiser pour ce fit
        os.makedirs(models_dir, exist_ok=True)
        print(f"Entraînement des modèles de base sur l'ensemble des données (sauvegardés dans '{models_dir}')...")

        for model_idx, model_config in enumerate(self.base_models_configs):
            model_type = model_config.get('model_type', 'unknown_type')
            model_params = model_config.get('model_params', {})
            model_path_prefix = model_config.get('model_path_prefix', f"base_model_{model_idx}_{model_type}")
            full_model_path = os.path.join(models_dir, f"{model_path_prefix}_full.joblib")

            print(f"  Entraînement du modèle {model_type} sur toutes les données...")
            training_results_full = train_model(
                X, y,
                model_type=model_type,
                model_params=model_params.copy(),
                model_path=full_model_path,
                test_size=0.01, # Non pertinent ici car on entraîne sur tout X, y
                scale_features=scale_features_full
            )
            self.trained_full_data_model_paths.append(full_model_path)
            self.base_model_feature_importances_full.append(training_results_full.get('feature_importances'))
            if training_results_full.get('feature_importances'):
                print(f"    Importance des features pour {model_type} (full data): {training_results_full.get('feature_importances')}")

        print("Tous les modèles de base ont été entraînés sur l'ensemble des données et sauvegardés.")

    def predict_with_full_data_models(self, X_new: pd.DataFrame, return_uncertainty_if_available: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Génère des prédictions à partir des modèles de base entraînés sur l'ensemble des données.

        Args:
            X_new (pd.DataFrame): Nouvelles données pour la prédiction.

        Returns:
            pd.DataFrame: Prédictions des modèles de base (probabilités si applicable).
        """
        if not self.trained_full_data_model_paths:
            raise RuntimeError("Les modèles de base entraînés sur toutes les données ne sont pas disponibles. Appelez d'abord fit_base_models_on_full_data.")

        predictions_array = np.zeros((len(X_new), len(self.trained_full_data_model_paths)))
        uncertainties_list = [] # Pour stocker les DataFrames d'incertitude
        column_names = []
        uncertainty_column_names = []

        for model_idx, model_path in enumerate(self.trained_full_data_model_paths):
            model_config = self.base_models_configs[model_idx]
            model_type_name = model_config.get('model_type', 'unknown')
            base_col_name = f"base_model_{model_idx}_{model_type_name}"
            column_names.append(base_col_name)

            try:
                # Demander l'incertitude si return_uncertainty_if_available est True
                pred_output = load_model_and_predict(
                    X_new,
                    model_path=model_path,
                    return_probabilities=True, # Toujours demander les probas/valeurs brutes ici
                    return_uncertainty=return_uncertainty_if_available
                )

                current_preds: np.ndarray
                current_uncertainty: Optional[np.ndarray] = None

                if isinstance(pred_output, tuple):
                    current_preds, current_uncertainty = pred_output
                else:
                    current_preds = pred_output
                
                # Assurer que current_preds est 1D pour la colonne de prédictions
                if current_preds.ndim > 1 and current_preds.shape[1] >= 2: # Ex: (N, 2) pour probas binaires
                    current_preds = current_preds[:, 1] # Proba classe 1
                elif current_preds.ndim > 1 and current_preds.shape[1] == 1: # Ex: (N, 1)
                     current_preds = current_preds.ravel()
                
                predictions_array[:, model_idx] = current_preds

                if return_uncertainty_if_available and current_uncertainty is not None:
                    uncertainties_list.append(pd.Series(current_uncertainty.ravel(), index=X_new.index, name=f"{base_col_name}_uncertainty"))
                    if f"{base_col_name}_uncertainty" not in uncertainty_column_names: # Évite doublons si plusieurs modèles retournent incertitude
                        uncertainty_column_names.append(f"{base_col_name}_uncertainty")
                elif return_uncertainty_if_available: # Si l'incertitude était demandée mais non fournie
                    # Ajouter une colonne de NaN pour maintenir la structure du DataFrame d'incertitude
                    uncertainties_list.append(pd.Series(np.nan, index=X_new.index, name=f"{base_col_name}_uncertainty"))
                    if f"{base_col_name}_uncertainty" not in uncertainty_column_names:
                         uncertainty_column_names.append(f"{base_col_name}_uncertainty")


            except Exception as e:
                print(f"Erreur lors de la prédiction avec le modèle {model_path}: {e}")
                predictions_array[:, model_idx] = np.nan
                if return_uncertainty_if_available:
                    uncertainties_list.append(pd.Series(np.nan, index=X_new.index, name=f"{base_col_name}_uncertainty"))
                    if f"{base_col_name}_uncertainty" not in uncertainty_column_names:
                         uncertainty_column_names.append(f"{base_col_name}_uncertainty")

        
        predictions_df = pd.DataFrame(predictions_array, index=X_new.index, columns=column_names)

        if return_uncertainty_if_available and uncertainties_list:
            uncertainties_df = pd.concat(uncertainties_list, axis=1)
            # S'assurer que les colonnes correspondent à celles attendues
            if uncertainty_column_names:
                 uncertainties_df = uncertainties_df.reindex(columns=uncertainty_column_names)
            else: # Si aucun modèle n'a retourné d'incertitude mais qu'elle était demandée
                 uncertainties_df = pd.DataFrame(index=X_new.index) # DataFrame vide avec le bon index
            return predictions_df, uncertainties_df
        
        return predictions_df


class MetaModel:
    """
    Niveau 2: Méta-modèle d'agrégation adaptative.
    Prend en entrée les prédictions des modèles de base et potentiellement des features contextuelles.
    """
    def __init__(self, model_type: str = 'logistic_regression', model_params: Optional[Dict[str, Any]] = None):
        """
        Initialise le méta-modèle.

        Args:
            model_type (str): Type de modèle à utiliser comme méta-modèle (ex: 'logistic_regression', 'lightgbm').
            model_params (Optional[Dict[str, Any]]): Paramètres pour le méta-modèle.
        """
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}
        self.model: Any = None # Le méta-modèle réel sera stocké ici

        # Initialisation du modèle placeholder
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**self.model_params)
        # Ajouter d'autres types de méta-modèles ici (ex: LightGBM, XGBoost)
        # elif self.model_type == 'lightgbm':
        #     import lightgbm as lgb
        #     self.model = lgb.LGBMClassifier(**self.model_params)
        else:
            raise ValueError(f"Type de méta-modèle non supporté: {self.model_type}")

    def fit(self, X_meta: pd.DataFrame, y_meta: pd.Series, X_contextual: Optional[pd.DataFrame] = None):
        """
        Entraîne le méta-modèle.

        Args:
            X_meta (pd.DataFrame): Prédictions out-of-fold des modèles de base.
            y_meta (pd.Series): Cible réelle pour l'entraînement du méta-modèle.
            X_contextual (Optional[pd.DataFrame]): Features contextuelles supplémentaires.
        """
        if X_contextual is not None:
            # S'assurer que les index correspondent pour la jointure
            if not X_meta.index.equals(X_contextual.index):
                 raise ValueError("Les index de X_meta et X_contextual doivent correspondre.")
            X_train_meta = pd.concat([X_meta, X_contextual], axis=1)
        else:
            X_train_meta = X_meta
        
        print(f"Entraînement du méta-modèle ({self.model_type}) sur {X_train_meta.shape[0]} échantillons et {X_train_meta.shape[1]} features.")
        self.model.fit(X_train_meta, y_meta)
        print("Méta-modèle entraîné.")

    def predict(self, X_meta: pd.DataFrame, X_contextual: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit les sorties brutes du méta-modèle.

        Args:
            X_meta (pd.DataFrame): Prédictions des modèles de base pour les nouvelles données.
            X_contextual (Optional[pd.DataFrame]): Features contextuelles supplémentaires.

        Returns:
            np.ndarray: Prédictions brutes du méta-modèle.
        """
        if self.model is None:
            raise RuntimeError("Le méta-modèle n'est pas entraîné. Appelez d'abord la méthode fit.")

        if X_contextual is not None:
            if not X_meta.index.equals(X_contextual.index):
                 raise ValueError("Les index de X_meta et X_contextual doivent correspondre pour la prédiction.")
            X_predict_meta = pd.concat([X_meta, X_contextual], axis=1)
        else:
            X_predict_meta = X_meta
        
        return self.model.predict(X_predict_meta)

    def predict_proba(self, X_meta: pd.DataFrame, X_contextual: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit les probabilités du méta-modèle (pour les classifieurs).

        Args:
            X_meta (pd.DataFrame): Prédictions des modèles de base pour les nouvelles données.
            X_contextual (Optional[pd.DataFrame]): Features contextuelles supplémentaires.

        Returns:
            np.ndarray: Prédictions de probabilités du méta-modèle (souvent la probabilité de la classe positive).
        """
        if self.model is None:
            raise RuntimeError("Le méta-modèle n'est pas entraîné. Appelez d'abord la méthode fit.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Le méta-modèle de type {self.model_type} n'a pas de méthode 'predict_proba'.")

        if X_contextual is not None:
            if not X_meta.index.equals(X_contextual.index):
                 raise ValueError("Les index de X_meta et X_contextual doivent correspondre pour la prédiction.")
            X_predict_meta = pd.concat([X_meta, X_contextual], axis=1)
        else:
            X_predict_meta = X_meta
            
        # Retourne la probabilité de la classe 1 (supposant une classification binaire)
        return self.model.predict_proba(X_predict_meta)[:, 1]


class SignalCalibrator:
    """
    Niveau 3: Calibration des probabilités et transformation des signaux.
    Applique des techniques de calibration sur les sorties du méta-modèle.
    """
    def __init__(self, method: str = 'isotonic', cv_method_params: Optional[Dict[str, Any]] = None):
        """
        Initialise le calibrateur de signaux.

        Args:
            method (str): Méthode de calibration ('isotonic', 'platt').
            cv_method_params (Optional[Dict[str, Any]]): Paramètres pour CalibratedClassifierCV si method='platt'.
                                                        Ex: {'method': 'sigmoid', 'cv': 3}
        """
        self.method = method
        self.calibrator: Any = None
        self.cv_method_params = cv_method_params if cv_method_params is not None else {}

        if self.method not in ['isotonic', 'platt']:
            raise ValueError(f"Méthode de calibration non supportée: {self.method}. Choisissez 'isotonic' ou 'platt'.")

    def fit(self, y_true: pd.Series, y_pred_proba: pd.Series):
        """
        Entraîne le calibrateur.

        Args:
            y_true (pd.Series): Valeurs cibles réelles (0 ou 1).
            y_pred_proba (pd.Series): Probabilités prédites par le méta-modèle (avant calibration).
        """
        print(f"Entraînement du calibrateur de signaux avec la méthode: {self.method}")
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        elif self.method == 'platt':
            # CalibratedClassifierCV nécessite un estimateur de base.
            # Ici, on ne ré-entraîne pas le méta-modèle, mais on calibre ses sorties.
            # Une astuce est d'utiliser un estimateur "dummy" qui retourne les prédictions fournies.
            # Cependant, CalibratedClassifierCV est conçu pour calibrer un classifieur *pendant* son entraînement
            # ou en post-hoc sur un classifieur déjà entraîné.
            # Pour une calibration post-hoc simple des probabilités, la régression logistique sur les probas
            # est une forme de calibration de Platt. IsotonicRegression est souvent plus flexible.
            # Si on veut utiliser CalibratedClassifierCV directement sur des probas, c'est moins direct.
            # Pour l'instant, on va se concentrer sur IsotonicRegression qui est plus simple à appliquer post-hoc.
            # Une implémentation plus complète de Platt pourrait utiliser LogisticRegression directement.
            print("  La calibration de Platt avec CalibratedClassifierCV est plus complexe à appliquer directement sur des probabilités pré-calculées.")
            print("  Envisagez d'utiliser IsotonicRegression ou d'intégrer la calibration dans l'entraînement du méta-modèle si Platt est requis via CalibratedClassifierCV.")
            # Pour une version simple de Platt (régression logistique sur les probas):
            # self.calibrator = LogisticRegression()
            # self.calibrator.fit(y_pred_proba.values.reshape(-1, 1), y_true)
            # Pour l'instant, on lève une exception si Platt est demandé de cette manière simpliste.
            # raise NotImplementedError("La calibration de Platt via CalibratedClassifierCV sur des probabilités pré-calculées n'est pas directement implémentée de manière simple. Utilisez 'isotonic' ou adaptez.")
            # Alternative: utiliser LogisticRegression comme calibrateur de Platt
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_pred_proba.values.reshape(-1, 1), y_true)
            print("  Utilisation de LogisticRegression comme proxy pour la calibration de Platt sur les probabilités.")

        print("Calibrateur entraîné.")

    def transform(self, y_pred_proba: pd.Series) -> np.ndarray:
        """
        Transforme les probabilités prédites en utilisant le calibrateur entraîné.

        Args:
            y_pred_proba (pd.Series): Probabilités prédites par le méta-modèle (avant calibration).

        Returns:
            np.ndarray: Probabilités calibrées.
        """
        if self.calibrator is None:
            raise RuntimeError("Le calibrateur n'est pas entraîné. Appelez d'abord la méthode fit.")

        if self.method == 'isotonic':
            return self.calibrator.predict(y_pred_proba)
        elif self.method == 'platt': # En supposant que self.calibrator est une LogisticRegression
            if hasattr(self.calibrator, 'predict_proba'):
                 return self.calibrator.predict_proba(y_pred_proba.values.reshape(-1, 1))[:, 1]
            else: # Fallback si ce n'est pas un classifieur avec predict_proba (ne devrait pas arriver avec LR)
                 return self.calibrator.predict(y_pred_proba.values.reshape(-1, 1))


class HierarchicalEnsemble:
    """
    Orchestre l'ensemble hiérarchique à trois niveaux.
    1. Modèles de base spécialisés (prédictions OOF)
    2. Méta-modèle d'agrégation
    3. Calibration des signaux
    """
    def __init__(self,
                 base_models_configs: List[Dict[str, Any]],
                 meta_model_config: Dict[str, Any],
                 calibrator_config: Dict[str, Any],
                 cv_n_splits: int = 5,
                 cv_random_state: Optional[int] = None,
                 cv_is_time_series: bool = False):
        """
        Initialise l'ensemble hiérarchique.

        Args:
            base_models_configs: Configurations pour BaseModelsManager.
            meta_model_config: Configuration pour MetaModel (ex: {'model_type': 'logistic_regression'}).
            calibrator_config: Configuration pour SignalCalibrator (ex: {'method': 'isotonic'}).
            cv_n_splits: Nombre de folds pour la génération des OOF.
            cv_random_state: Graine aléatoire pour KFold.
            cv_is_time_series: Utiliser TimeSeriesSplit pour OOF.
        """
        self.base_models_manager = BaseModelsManager(base_models_configs)
        self.meta_model = MetaModel(**meta_model_config)
        self.signal_calibrator = SignalCalibrator(**calibrator_config)
        
        self.cv_n_splits = cv_n_splits
        self.cv_random_state = cv_random_state
        self.cv_is_time_series = cv_is_time_series
        
        self.is_fitted = False
        self.aggregated_feature_importances: Dict[str, Any] = {} # Pour stocker les importances agrégées

    def fit(self, X: pd.DataFrame, y: pd.Series, X_contextual_meta: Optional[pd.DataFrame] = None):
        """
        Entraîne l'ensemble du pipeline hiérarchique.

        Args:
            X (pd.DataFrame): Features d'entraînement.
            y (pd.Series): Cible d'entraînement.
            X_contextual_meta (Optional[pd.DataFrame]): Features contextuelles pour le méta-modèle.
                                                       Doit avoir le même index que X et y.
        """
        print("Début de l'entraînement de l'ensemble hiérarchique...")

        # Niveau 1: Entraîner les modèles de base et obtenir les prédictions OOF
        print("\n--- Niveau 1: Entraînement des modèles de base et génération des OOF ---")
        oof_base_predictions = self.base_models_manager.fit_predict_oof(
            X, y,
            n_splits=self.cv_n_splits,
            random_state=self.cv_random_state,
            is_time_series=self.cv_is_time_series
        )
        # Entraîner également les modèles de base sur l'ensemble des données pour les prédictions futures
        self.base_models_manager.fit_base_models_on_full_data(X, y)
        
        # Stocker les importances des features des modèles de base (entraînés sur full data)
        if self.base_models_manager.base_model_feature_importances_full:
            for i, config in enumerate(self.base_models_manager.base_models_configs):
                model_name = f"base_model_{i}_{config.get('model_type', 'unknown')}"
                importances = self.base_models_manager.base_model_feature_importances_full[i]
                if importances:
                    self.aggregated_feature_importances[model_name] = importances
            print(f"Importances des features des modèles de base (full data) stockées dans l'ensemble.")


        # Niveau 2: Entraîner le méta-modèle sur les prédictions OOF
        print("\n--- Niveau 2: Entraînement du méta-modèle ---")
        # S'assurer que y_meta correspond aux prédictions OOF (même index)
        y_meta_train = y.loc[oof_base_predictions.index]
        X_contextual_meta_train = None
        if X_contextual_meta is not None:
            X_contextual_meta_train = X_contextual_meta.loc[oof_base_predictions.index]

        self.meta_model.fit(oof_base_predictions, y_meta_train, X_contextual=X_contextual_meta_train)
        
        # Obtenir les prédictions du méta-modèle sur les données OOF pour entraîner le calibrateur
        meta_model_oof_probas = self.meta_model.predict_proba(oof_base_predictions, X_contextual=X_contextual_meta_train)
        meta_model_oof_probas_series = pd.Series(meta_model_oof_probas, index=oof_base_predictions.index)

        # Niveau 3: Entraîner le calibrateur de signaux
        print("\n--- Niveau 3: Entraînement du calibrateur de signaux ---")
        self.signal_calibrator.fit(y_meta_train, meta_model_oof_probas_series)
        
        self.is_fitted = True
        print("\nEnsemble hiérarchique entraîné avec succès.")

    def predict_proba(self, X_new: pd.DataFrame, X_contextual_new: Optional[pd.DataFrame] = None, return_full_output: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Génère des prédictions de probabilités finales calibrées pour de nouvelles données.
        Peut optionnellement retourner une sortie plus riche incluant l'incertitude.

        Args:
            X_new (pd.DataFrame): Nouvelles features pour la prédiction.
            X_contextual_new (Optional[pd.DataFrame]): Nouvelles features contextuelles.
                                                       Doit avoir le même index que X_new.
            return_full_output (bool): Si True, retourne un dictionnaire avec probabilités calibrées,
                                       prédictions brutes du méta-modèle, et incertitudes (si disponibles).

        Returns:
            Union[np.ndarray, Dict[str, Any]]:
                - np.ndarray des probabilités finales calibrées (si return_full_output=False).
                - Dict avec 'calibrated_probabilities', 'meta_model_raw_probabilities',
                  'base_model_predictions', et optionnellement 'base_model_uncertainties'
                  (si return_full_output=True).
        """
        if not self.is_fitted:
            raise RuntimeError("L'ensemble hiérarchique n'est pas entraîné. Appelez d'abord la méthode fit.")

        print("Génération des prédictions avec l'ensemble hiérarchique...")

        # Niveau 1: Prédictions des modèles de base (entraînés sur toutes les données)
        print("  Niveau 1: Prédictions des modèles de base...")
        # Demander l'incertitude si return_full_output est True
        base_model_output = self.base_models_manager.predict_with_full_data_models(X_new, return_uncertainty_if_available=return_full_output)

        base_model_predictions_new: pd.DataFrame
        base_model_uncertainties_new: Optional[pd.DataFrame] = None

        if isinstance(base_model_output, tuple):
            base_model_predictions_new, base_model_uncertainties_new = base_model_output
        else:
            base_model_predictions_new = base_model_output
            
        # Niveau 2: Prédictions du méta-modèle
        print("  Niveau 2: Prédictions du méta-modèle...")
        meta_model_probas_raw_new = self.meta_model.predict_proba(base_model_predictions_new, X_contextual=X_contextual_new)
        meta_model_probas_raw_new_series = pd.Series(meta_model_probas_raw_new, index=X_new.index)
        
        # Niveau 3: Calibration des signaux
        print("  Niveau 3: Calibration des signaux...")
        calibrated_probas_final = self.signal_calibrator.transform(meta_model_probas_raw_new_series)
        
        print("Prédictions finales calibrées générées.")

        if return_full_output:
            output_dict: Dict[str, Any] = {
                "calibrated_probabilities": calibrated_probas_final,
                "meta_model_raw_probabilities": meta_model_probas_raw_new_series.values, # Retourner comme array numpy
                "base_model_predictions": base_model_predictions_new
            }
            if base_model_uncertainties_new is not None:
                output_dict["base_model_uncertainties"] = base_model_uncertainties_new
            return output_dict
        else:
            return calibrated_probas_final

# Exemple d'utilisation (à mettre dans un bloc if __name__ == "__main__" ou un script de test)
# def example_usage():
#     # Simuler des données
#     n_samples = 200
#     n_features = 10
#     X_train_data = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
#     y_train_data = pd.Series(np.random.randint(0, 2, n_samples))
#     X_context_data = pd.DataFrame(np.random.rand(n_samples, 2), columns=['ctx_feat_1', 'ctx_feat_2'], index=X_train_data.index)

#     # Configurations
#     base_configs = [
#         {'model_type': 'logistic_regression', 'model_params': {'C': 1.0}},
#         {'model_type': 'random_forest', 'model_params': {'n_estimators': 50, 'max_depth': 5}} # Nécessiterait une vraie implémentation
#     ]
#     # Pour BaseModelsManager, il faudrait que les model_type correspondent à des modèles qu'il peut réellement instancier et entraîner.
#     # Pour l'instant, il utilise LogisticRegression en interne comme placeholder.

#     meta_config = {'model_type': 'logistic_regression', 'model_params': {'solver': 'liblinear'}}
#     calib_config = {'method': 'isotonic'}

#     # Initialiser l'ensemble
#     ensemble_model = HierarchicalEnsemble(
#         base_models_configs=base_configs, # Utiliser des configs simples pour les placeholders
#         meta_model_config=meta_config,
#         calibrator_config=calib_config,
#         cv_n_splits=3,
#         cv_is_time_series=False
#     )

#     # Entraîner
#     ensemble_model.fit(X_train_data, y_train_data, X_contextual_meta=X_context_data)

#     # Prédire sur de nouvelles données
#     X_test_data = pd.DataFrame(np.random.rand(50, n_features), columns=[f'feature_{i}' for i in range(n_features)])
#     X_test_context_data = pd.DataFrame(np.random.rand(50, 2), columns=['ctx_feat_1', 'ctx_feat_2'], index=X_test_data.index)
#     final_predictions = ensemble_model.predict_proba(X_test_data, X_contextual_new=X_test_context_data)
    
#     print("\nExemple de prédictions finales calibrées:")
#     print(final_predictions[:5])

# if __name__ == '__main__':
#     # Pour que l'exemple fonctionne, BaseModelsManager appelle maintenant `train_model` et `load_model_and_predict`.
#     # Assurez-vous que les `model_type` dans `base_configs` sont gérés par `train_model` dans `models.py`.
#     # Par exemple, 'logistic_regression' et 'random_forest' sont gérés.
#     # Les modèles de base réels seront sauvegardés dans 'trained_ensemble_base_models/' et les OOF temporairement.
    
#     # Décommenter pour tester si `train_model` et `load_model_and_predict` sont fonctionnels
#     # et si les types de modèles de base sont correctement configurés.
#     # try:
#     #     example_usage()
#     # finally:
#     #     # Nettoyage du répertoire des modèles de base si créé par l'exemple
#     #     if os.path.exists("trained_ensemble_base_models_example"):
#     #         shutil.rmtree("trained_ensemble_base_models_example")
#     #     # Le nettoyage du répertoire temp OOF est géré par le destructeur de BaseModelsManager
#     #     # ou explicitement si l'objet ensemble_model est supprimé ou sort du scope.
    pass