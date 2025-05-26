#!/usr/bin/env python3
"""
Walk-Forward Validation Module
Implements rolling window model retraining for adaptive trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from pathlib import Path
import joblib
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppression des warnings sklearn
warnings.filterwarnings("ignore", category=UserWarning)


class WalkForwardResults:
    """
    Classe pour stocker les résultats de la validation walk-forward
    """
    
    def __init__(self, 
                 scores: List[float],
                 drift_scores: List[float],
                 windows: List[Dict],
                 models: List[BaseEstimator]):
        """
        Initialise les résultats de la validation walk-forward
        
        Args:
            scores: Liste des scores de performance pour chaque fenêtre
            drift_scores: Liste des scores de dérive conceptuelle
            windows: Liste des informations détaillées sur chaque fenêtre
            models: Liste des modèles entraînés pour chaque fenêtre
        """
        self.scores = scores
        self.drift_scores = drift_scores
        self.windows = windows
        self.models = models
        
        # Statistiques agrégées
        if scores:
            self.mean_score = np.mean(scores)
            self.std_score = np.std(scores)
            self.min_score = np.min(scores)
            self.max_score = np.max(scores)
        else:
            self.mean_score = self.std_score = self.min_score = self.max_score = 0.0
            
        if drift_scores:
            self.mean_drift = np.mean(drift_scores)
            self.max_drift = np.max(drift_scores)
        else:
            self.mean_drift = self.max_drift = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des résultats
        """
        return {
            'num_windows': len(self.windows),
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'mean_drift': self.mean_drift,
            'max_drift': self.max_drift,
            'scores': self.scores,
            'drift_scores': self.drift_scores
        }
    
    def __str__(self) -> str:
        """
        Représentation textuelle des résultats
        """
        return (f"WalkForwardResults("
                f"windows={len(self.windows)}, "
                f"mean_score={self.mean_score:.4f}, "
                f"std_score={self.std_score:.4f}, "
                f"mean_drift={self.mean_drift:.4f})")


class WalkForwardValidator:
    """
    Implémente la validation walk-forward avec réentraînement automatique des modèles
    """
    
    def __init__(self, 
                 training_window_days: int = 252,  # 1 an de données d'entraînement
                 validation_window_days: int = 63,   # 3 mois de validation
                 retrain_frequency_days: int = 21,   # Réentraîner toutes les 3 semaines
                 min_training_samples: int = 100,
                 performance_threshold: float = 0.55,  # Seuil de performance minimum
                 drift_threshold: float = 0.10,       # Seuil de dérive pour forcer un réentraînement
                 model_save_path: str = "models_store/walk_forward",
                 expanding_training_window: bool = False):
       """
       Initialise le validateur walk-forward
       
       Args:
           training_window_days: Nombre de jours pour la fenêtre d'entraînement.
                                 Si expanding_training_window est False, c'est la taille de la fenêtre glissante.
                                 Si expanding_training_window est True, c'est la taille initiale de la fenêtre croissante.
           validation_window_days: Nombre de jours pour la fenêtre de validation/test.
           retrain_frequency_days: Fréquence de réentraînement en jours (pas glissant de la fenêtre).
           min_training_samples: Nombre minimum d'échantillons pour l'entraînement.
           performance_threshold: Seuil de performance en dessous duquel on réentraîne (utilisé par run_walk_forward_validation).
           drift_threshold: Seuil de dérive conceptuelle (utilisé par run_walk_forward_validation).
           model_save_path: Chemin pour sauvegarder les modèles.
           expanding_training_window: Si True, la fenêtre d'entraînement est croissante.
                                      Si False (par défaut), la fenêtre d'entraînement est glissante.
       """
       self.training_window_days = training_window_days
       self.validation_window_days = validation_window_days
       self.retrain_frequency_days = retrain_frequency_days
       self.min_training_samples = min_training_samples
       self.performance_threshold = performance_threshold
       self.drift_threshold = drift_threshold
       self.model_save_path = Path(model_save_path)
       self.expanding_training_window = expanding_training_window
       self.model_save_path.mkdir(parents=True, exist_ok=True)
       
       # Historique des performances et modèles
       self.performance_history = []
       self.model_history = {}
       self.drift_scores = []
       self.retrain_dates = []
       
       # Configuration du logging
       logging.basicConfig(level=logging.INFO)
       self.logger = logging.getLogger(__name__)
        
    def should_retrain(self, current_date: datetime, last_retrain_date: Optional[datetime],
                       recent_performance: Optional[float], drift_score: Optional[float]) -> bool:
        """
        Détermine si un réentraînement est nécessaire
        
        Args:
            current_date: Date actuelle
            last_retrain_date: Date du dernier réentraînement
            recent_performance: Performance récente du modèle
            drift_score: Score de dérive conceptuelle
            
        Returns:
            True si un réentraînement est nécessaire
        """
        # Vérification de la fréquence temporelle
        if last_retrain_date is None:
            return True
            
        days_since_retrain = (current_date - last_retrain_date).days
        if days_since_retrain >= self.retrain_frequency_days:
            self.logger.info(f"Réentraînement nécessaire: {days_since_retrain} jours depuis le dernier")
            return True
            
        # Vérification de la performance
        if recent_performance is not None and recent_performance < self.performance_threshold:
            self.logger.info(f"Réentraînement nécessaire: performance {recent_performance:.3f} < seuil {self.performance_threshold}")
            return True
            
        # Vérification de la dérive conceptuelle
        if drift_score is not None and drift_score > self.drift_threshold:
            self.logger.info(f"Réentraînement nécessaire: dérive {drift_score:.3f} > seuil {self.drift_threshold}")
            return True
            
        return False
    
    def calculate_concept_drift(self,
                                historical_features: pd.DataFrame,
                                recent_features: pd.DataFrame,
                                method: str = 'kolmogorov_smirnov') -> float:
        """
        Calcule un score de dérive conceptuelle entre les données historiques et récentes
        
        Args:
            historical_features: Features des données historiques
            recent_features: Features des données récentes
            method: Méthode de calcul ('kolmogorov_smirnov', 'population_stability')
            
        Returns:
            Score de dérive (0 = pas de dérive, 1 = dérive maximale)
        """
        from scipy.stats import ks_2samp
        
        if method == 'kolmogorov_smirnov':
            drift_scores_list = [] # Renommé pour éviter confusion avec self.drift_scores
            
            for column in historical_features.columns:
                if pd.api.types.is_numeric_dtype(historical_features[column]):
                    # Test de Kolmogorov-Smirnov pour chaque feature
                    ks_stat, p_value = ks_2samp(
                        historical_features[column].dropna(),
                        recent_features[column].dropna()
                    )
                    drift_scores_list.append(ks_stat)
                    
            return np.mean(drift_scores_list) if drift_scores_list else 0.0
            
        elif method == 'population_stability':
            return self._calculate_population_stability_index(historical_features, recent_features)
        
        return 0.0
    
    def _calculate_population_stability_index(self,
                                              historical_data: pd.DataFrame,
                                              recent_data: pd.DataFrame,
                                              num_bins: int = 10) -> float:
        """
        Calcule l'Index de Stabilité de Population (PSI)
        """
        psi_values = []
        
        for column in historical_data.columns:
            if pd.api.types.is_numeric_dtype(historical_data[column]):
                # Création des bins basés sur les données historiques
                # S'assurer que les bins sont valides même si les données sont constantes
                try:
                    # Tentative avec qcut pour des bins basés sur les quantiles
                    # Si qcut échoue (par exemple, données non uniques), utiliser cut standard
                    try:
                        historical_binned, bins = pd.qcut(historical_data[column], q=num_bins, retbins=True, duplicates='drop')
                    except ValueError: # Peut arriver si pas assez de valeurs uniques pour les quantiles
                        _, bins = pd.cut(historical_data[column], bins=num_bins, retbins=True, duplicates='drop')

                    if len(bins) < 2: # Pas assez de bins créés
                        self.logger.debug(f"PSI: Pas assez de bins pour la colonne {column}, saut.")
                        continue

                    # Distribution historique
                    hist_counts = pd.cut(historical_data[column], bins=bins, include_lowest=True, right=True).value_counts(sort=False)
                    hist_dist = hist_counts / hist_counts.sum()
                    
                    # Distribution récente
                    recent_counts = pd.cut(recent_data[column], bins=bins, include_lowest=True, right=True).value_counts(sort=False)
                    recent_dist = recent_counts / recent_counts.sum()

                    # S'assurer que les index correspondent pour l'alignement
                    hist_dist = hist_dist.reindex(recent_dist.index).fillna(0)
                    recent_dist = recent_dist.reindex(hist_dist.index).fillna(0)

                except Exception as e_binning:
                    self.logger.warning(f"PSI: Erreur lors du binning pour la colonne {column}: {e_binning}. Saut.")
                    continue

                # Calcul PSI
                psi = 0
                for i in range(len(hist_dist)):
                    # Remplacer les zéros par une petite valeur pour éviter log(0) ou division par zéro
                    p_hist = hist_dist.iloc[i] if hist_dist.iloc[i] > 0 else 1e-6
                    p_recent = recent_dist.iloc[i] if recent_dist.iloc[i] > 0 else 1e-6
                    
                    psi += (p_recent - p_hist) * np.log(p_recent / p_hist)
                
                psi_values.append(abs(psi))
                
        return np.mean(psi_values) if psi_values else 0.0
    
    def evaluate_model_performance(self,
                                   model: BaseEstimator,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   metric: str = 'accuracy') -> float:
        """
        Évalue la performance d'un modèle sur un ensemble de test
        
        Args:
            model: Modèle à évaluer
            X_test: Features de test
            y_test: Labels de test
            metric: Métrique d'évaluation ('accuracy', 'precision', 'recall', 'f1')
            
        Returns:
            Score de performance
        """
        if X_test.empty or y_test.empty:
            self.logger.warning("Ensemble de test vide pour l'évaluation.")
            return 0.0
        try:
            predictions = model.predict(X_test)
            
            if metric == 'accuracy':
                return accuracy_score(y_test, predictions)
            elif metric == 'precision':
                return precision_score(y_test, predictions, average='weighted', zero_division=0)
            elif metric == 'recall':
                return recall_score(y_test, predictions, average='weighted', zero_division=0)
            elif metric == 'f1':
                return f1_score(y_test, predictions, average='weighted', zero_division=0)
            else:
                self.logger.warning(f"Métrique d'évaluation inconnue '{metric}'. Utilisation de 'accuracy'.")
                return accuracy_score(y_test, predictions)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du modèle: {e}", exc_info=True)
            return 0.0 # Retourner une valeur neutre en cas d'erreur
    
    def save_model(self, model: BaseEstimator, model_name: str, timestamp: datetime) -> str:
        """
        Sauvegarde un modèle avec un timestamp
        
        Args:
            model: Modèle à sauvegarder
            model_name: Nom du modèle
            timestamp: Timestamp pour le versioning
            
        Returns:
            Chemin du fichier sauvegardé
        """
        filename = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.joblib"
        filepath = self.model_save_path / filename
        
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"Modèle sauvegardé: {filepath}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle {filepath}: {e}", exc_info=True)
            return "" # Retourner un chemin vide en cas d'erreur
            
        return str(filepath)
    
    def load_latest_model(self, model_name: str) -> Optional[BaseEstimator]:
        """
        Charge le modèle le plus récent pour un nom donné
        
        Args:
            model_name: Nom du modèle à charger
            
        Returns:
            Modèle chargé ou None si aucun modèle trouvé ou en cas d'erreur.
        """
        pattern = f"{model_name}_*.joblib"
        try:
            model_files = list(self.model_save_path.glob(pattern))
            
            if not model_files:
                self.logger.info(f"Aucun modèle trouvé pour le pattern '{pattern}' dans {self.model_save_path}")
                return None
                
            # Trier par date de modification et prendre le plus récent
            latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
            
            model = joblib.load(latest_file)
            self.logger.info(f"Modèle chargé: {latest_file}")
            return model
        except FileNotFoundError:
            self.logger.info(f"Aucun modèle trouvé (FileNotFound) pour le pattern '{pattern}' dans {self.model_save_path}")
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle le plus récent pour '{model_name}': {e}", exc_info=True)
            return None
    
    def run_walk_forward_validation(self,
                                    data: pd.DataFrame,
                                    target_column: str,
                                    feature_columns: List[str],
                                    model_factory: Callable[[], BaseEstimator],
                                    date_column: str = 'timestamp',
                                    model_name: str = 'walk_forward_model') -> Dict[str, Any]:
        """
        Lance la validation walk-forward complète.
        Cette méthode est plus orientée vers une simulation de trading adaptatif
        où le modèle est réentraîné basé sur des conditions (temps, performance, dérive).
        
        Args:
            data: DataFrame avec toutes les données, doit contenir `date_column`.
            target_column: Nom de la colonne cible.
            feature_columns: Liste des colonnes de features.
            model_factory: Fonction qui retourne une nouvelle instance de modèle non entraîné.
            date_column: Nom de la colonne de date/timestamp.
            model_name: Nom de base pour sauvegarder les modèles versionnés.
            
        Returns:
            Dictionnaire avec les résultats détaillés de la validation.
        """
        if not isinstance(data.index, pd.DatetimeIndex) and date_column not in data.columns:
             data = data.set_index(pd.to_datetime(data[date_column])) # Assurer un DatetimeIndex
        elif date_column in data.columns:
             data[date_column] = pd.to_datetime(data[date_column])
             data = data.sort_values(by=date_column) # S'assurer que les données sont triées par date

        if data.empty:
            self.logger.warning("Le DataFrame d'entrée est vide. Arrêt de la validation walk-forward.")
            return {}

        # Initialiser les variables de tracking
        results_agg = {
            'performance_history': [], # Liste de dicts: {'date', 'performance', 'training_samples', 'drift_score'}
            'retrain_dates': [],       # Liste des dates de réentraînement
            'drift_scores_log': [],    # Liste de dicts: {'date', 'drift_score'}
            'predictions': [],         # Liste de dicts: {'date', 'actual', 'predicted', 'probabilities'(opt)}
            'model_versions': []       # Liste de dicts: {'date', 'model_path', 'training_samples'}
        }
        
        current_model: Optional[BaseEstimator] = None
        last_retrain_time: Optional[datetime] = None # Utiliser datetime pour la comparaison
        
        # Déterminer les points de début pour chaque fenêtre de validation (pas de `retrain_frequency_days`)
        # La boucle avance par `retrain_frequency_days` (interprété comme nombre d'échantillons ici)
        # ou par la taille de la fenêtre de validation si plus petit.
        step = self.retrain_frequency_days
        
        # Le premier point de départ de la validation doit permettre une fenêtre d'entraînement complète avant.
        # Si training_window_days est le nombre d'échantillons pour l'entraînement.
        first_validation_start_sample_idx = self.training_window_days

        self.logger.info(f"Démarrage de la validation walk-forward sur {len(data)} échantillons.")
        self.logger.info(f"Fenêtre d'entraînement initiale: {self.training_window_days} jours/échantillons, "
                         f"Fenêtre de validation: {self.validation_window_days} jours/échantillons, "
                         f"Fréquence de réévaluation/pas: {step} jours/échantillons.")

        current_validation_start_sample_idx = first_validation_start_sample_idx

        while current_validation_start_sample_idx + self.validation_window_days <= len(data):
            
            # Définir la fenêtre de validation actuelle
            current_validation_end_sample_idx = current_validation_start_sample_idx + self.validation_window_days
            validation_data_window = data.iloc[current_validation_start_sample_idx:current_validation_end_sample_idx]
            
            if validation_data_window.empty:
                self.logger.warning(f"Fenêtre de validation vide à l'indice {current_validation_start_sample_idx}. Arrêt.")
                break

            current_eval_date = validation_data_window[date_column].iloc[0] if date_column in validation_data_window else validation_data_window.index[0]
            
            # Définir la fenêtre d'entraînement
            # L'entraînement se termine juste avant le début de la fenêtre de validation
            training_data_end_sample_idx = current_validation_start_sample_idx
            
            if self.expanding_training_window:
                training_data_start_sample_idx = 0
            else: # Fenêtre glissante
                training_data_start_sample_idx = max(0, training_data_end_sample_idx - self.training_window_days)
            
            training_data_window = data.iloc[training_data_start_sample_idx:training_data_end_sample_idx]

            if len(training_data_window) < self.min_training_samples:
                self.logger.info(f"Pas assez de données d'entraînement ({len(training_data_window)} < {self.min_training_samples}) "
                                 f"à {current_eval_date}. Avance de la fenêtre.")
                current_validation_start_sample_idx += step
                continue
                
            X_train = training_data_window[feature_columns].fillna(0)
            y_train = training_data_window[target_column].fillna(0) # Ou une autre stratégie de remplissage
            X_val = validation_data_window[feature_columns].fillna(0)
            y_val = validation_data_window[target_column].fillna(0)
            
            # Calculer la dérive conceptuelle (optionnel, avant de décider de réentraîner)
            drift_score_value = None
            if current_model is not None and not X_train.empty: # Nécessite des données de référence
                 # Pour la dérive, comparer les données d'entraînement actuelles avec une période précédente
                 # ou les données de validation avec les données d'entraînement.
                 # Ici, on compare X_train (référence) avec X_val (nouveau)
                 drift_score_value = self.calculate_concept_drift(X_train.tail(self.training_window_days // 2), # Partie récente de l'entraînement
                                                                  X_val.head(self.validation_window_days // 2)) # Partie initiale de la validation
                 results_agg['drift_scores_log'].append({'date': current_eval_date, 'drift_score': drift_score_value})
            
            # Calculer la performance récente (si le modèle existe et a été évalué)
            recent_perf_value = None
            if results_agg['performance_history']: # Utiliser l'historique agrégé
                # Prendre la moyenne des N dernières performances stockées
                last_n_performances = [p['performance'] for p in results_agg['performance_history'][-3:]] # Ex: 3 dernières
                if last_n_performances:
                    recent_perf_value = np.nanmean(last_n_performances) # Ignorer les NaN
            
            # Décider si on doit réentraîner
            needs_retrain = self.should_retrain(current_eval_date, last_retrain_time, recent_perf_value, drift_score_value)
            
            if needs_retrain or current_model is None:
                self.logger.info(f"Réentraînement du modèle à la date {current_eval_date} (ou initialisation). "
                                 f"Raison: Initial={current_model is None}, Freq={needs_retrain and last_retrain_time is not None}, Perf={recent_perf_value}, Drift={drift_score_value}")
                
                current_model = model_factory()
                try:
                    current_model.fit(X_train, y_train)
                    model_file_path = self.save_model(current_model, model_name, current_eval_date)
                    last_retrain_time = current_eval_date
                    results_agg['retrain_dates'].append(current_eval_date)
                    results_agg['model_versions'].append({
                        'date': current_eval_date,
                        'model_path': model_file_path,
                        'training_samples': len(training_data_window)
                    })
                except Exception as e_fit:
                    self.logger.error(f"Erreur lors de l'entraînement du modèle à {current_eval_date}: {e_fit}", exc_info=True)
                    current_model = None # Ne pas utiliser un modèle mal entraîné
                    # On pourrait choisir de sauter cette fenêtre ou d'utiliser un modèle précédent si disponible
            
            if current_model is None:
                self.logger.warning(f"Aucun modèle disponible pour la prédiction à {current_eval_date}. Saut de la fenêtre de validation.")
                current_validation_start_sample_idx += step
                continue

            # Faire des prédictions et évaluer
            try:
                predictions_val = current_model.predict(X_val)
                proba_val = current_model.predict_proba(X_val) if hasattr(current_model, 'predict_proba') else None
                
                perf_score = self.evaluate_model_performance(current_model, X_val, y_val)
                
                results_agg['performance_history'].append({
                    'date': current_eval_date, # Date de début de la fenêtre de validation
                    'performance': perf_score,
                    'training_samples': len(training_data_window), # Taille de l'entraînement pour ce modèle
                    'validation_samples': len(validation_data_window),
                    'drift_score_at_eval': drift_score_value # Dérive calculée avant cette évaluation
                })
                
                # Stocker les prédictions détaillées
                for idx_val, original_data_idx in enumerate(validation_data_window.index):
                    pred_entry = {
                        'date': original_data_idx, # Index original de la donnée (date)
                        'actual': y_val.iloc[idx_val],
                        'predicted': predictions_val[idx_val],
                    }
                    if proba_val is not None:
                        pred_entry['probabilities'] = proba_val[idx_val]
                    results_agg['predictions'].append(pred_entry)
                
                self.logger.info(f"Fenêtre de validation débutant à {current_eval_date}: Performance={perf_score:.4f}")
                
            except Exception as e_pred:
                self.logger.error(f"Erreur lors de la prédiction/évaluation à {current_eval_date}: {e_pred}", exc_info=True)
                results_agg['performance_history'].append({ # Enregistrer l'échec
                    'date': current_eval_date, 'performance': np.nan,
                    'training_samples': len(training_data_window), 'validation_samples': len(validation_data_window),
                    'drift_score_at_eval': drift_score_value, 'error': str(e_pred)
                })
            
            # Avancer la fenêtre de validation
            current_validation_start_sample_idx += step
        
        # Résumé final
        if results_agg['performance_history']:
            valid_performances = [p['performance'] for p in results_agg['performance_history'] if not np.isnan(p['performance'])]
            avg_perf = np.mean(valid_performances) if valid_performances else np.nan
            std_perf = np.std(valid_performances) if valid_performances else np.nan
            
            self.logger.info(f"Validation walk-forward (simulation adaptative) terminée:")
            self.logger.info(f"  - Performance moyenne (valide): {avg_perf:.4f} ± {std_perf:.4f} sur {len(valid_performances)} évaluations valides.")
            self.logger.info(f"  - Nombre de réentraînements effectués: {len(results_agg['retrain_dates'])}")
            self.logger.info(f"  - Nombre total de prédictions générées: {len(results_agg['predictions'])}")
            
            results_agg['summary'] = {
                'avg_performance': avg_perf,
                'std_performance': std_perf,
                'num_valid_evaluations': len(valid_performances),
                'num_retrains': len(results_agg['retrain_dates']),
                'num_predictions': len(results_agg['predictions'])
            }
        else:
            self.logger.warning("Aucune performance n'a été enregistrée pendant la validation walk-forward.")
            results_agg['summary'] = {'error': "Aucune évaluation de performance complétée."}
            
        return results_agg
    
    def create_training_windows(self, data_index: pd.Index) -> List[Tuple[List[int], List[int]]]:
        """
        Crée les fenêtres d'entraînement et de test pour la validation walk-forward.
        Utilisé par `run_validation` pour une validation walk-forward plus "classique"
        où le modèle est réentraîné à chaque fenêtre.
        Supporte les fenêtres d'entraînement glissantes ou croissantes.

        Args:
            data_index: Index temporel du dataset.

        Returns:
            Liste de tuples (indices_train, indices_test) pour chaque fenêtre.
        """
        windows = []
        total_samples = len(data_index)
        
        initial_training_samples = self.training_window_days
        validation_samples = self.validation_window_days
        step_size = self.retrain_frequency_days # Interprété comme le pas de la fenêtre de validation

        if initial_training_samples <= 0 or validation_samples <= 0 or step_size <= 0:
            self.logger.warning("Les tailles de fenêtre (training, validation, step) doivent être positives pour create_training_windows.")
            return []

        # `current_validation_start_idx` est l'indice de début de la fenêtre de validation actuelle.
        # La première fenêtre de validation commence après la première fenêtre d'entraînement.
        current_validation_start_idx = initial_training_samples

        while current_validation_start_idx + validation_samples <= total_samples:
            test_start_idx = current_validation_start_idx
            test_end_idx = current_validation_start_idx + validation_samples
            
            train_end_idx = current_validation_start_idx # L'entraînement se termine juste avant le test
            
            if self.expanding_training_window:
                train_start_idx = 0
            else: # Fenêtre glissante
                train_start_idx = max(0, train_end_idx - initial_training_samples)
            
            if (train_end_idx - train_start_idx) >= self.min_training_samples:
                train_indices = list(range(train_start_idx, train_end_idx))
                test_indices = list(range(test_start_idx, test_end_idx))
                windows.append((train_indices, test_indices))
            else:
                self.logger.debug(f"create_training_windows: Fenêtre d'entraînement sautée car trop petite: "
                                 f"{train_end_idx - train_start_idx} < {self.min_training_samples}")

            current_validation_start_idx += step_size # Avancer pour la prochaine fenêtre de validation
            
        if not windows:
            self.logger.warning(f"create_training_windows: Aucune fenêtre n'a pu être créée. "
                                f"Total: {total_samples}, TrainInit: {initial_training_samples}, "
                                f"Valid: {validation_samples}, Step: {step_size}, MinTrain: {self.min_training_samples}")
        return windows

    def detect_concept_drift(self, X_historical: pd.DataFrame, X_recent: pd.DataFrame) -> float:
        """
        Détecte la dérive conceptuelle entre données historiques et récentes.
        Wrapper pour `calculate_concept_drift`.
        
        Args:
            X_historical: Features des données historiques.
            X_recent: Features des données récentes.
            
        Returns:
            Score de dérive conceptuelle (0 = pas de dérive, 1 = dérive maximale).
        """
        if X_historical.empty or X_recent.empty:
            self.logger.warning("Données historiques ou récentes vides pour la détection de dérive.")
            return 0.0 # Ou np.nan si préféré pour indiquer l'impossibilité de calculer
        return self.calculate_concept_drift(X_historical, X_recent)
    
    def run_validation(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     model_class: Callable[..., BaseEstimator], # Type hint pour la classe du modèle
                     model_params: Optional[Dict] = None) -> 'WalkForwardResults':
       """
       Effectue une validation walk-forward complète avec réentraînement périodique du modèle
       à chaque nouvelle fenêtre d'entraînement. Cette méthode est une forme "classique" de
       validation walk-forward, par opposition à `run_walk_forward_validation` qui simule
       un réentraînement plus adaptatif.

       Structure de base :
       1. Initialisation : Crée des fenêtres d'entraînement et de test successives via `create_training_windows`.
       2. Boucle sur chaque fenêtre :
          a. Réentraînement : Un nouveau modèle est instancié et entraîné sur la fenêtre d'entraînement actuelle.
          b. Évaluation : Le modèle entraîné est évalué sur la fenêtre de test suivante.
       3. Agrégation des résultats.

       Args:
           X: DataFrame des features. L'index doit être temporel et trié.
           y: Series de la variable cible. L'index doit correspondre à X.
           model_class: Classe du modèle à instancier et entraîner (ex: RandomForestClassifier).
           model_params: Dictionnaire des hyperparamètres à passer au constructeur du modèle.

       Returns:
           WalkForwardResults: Objet contenant les scores, les modèles et les détails des fenêtres.
       """
       if model_params is None:
           model_params = {}
           
       all_results_details = [] # Stocke des dictionnaires de détails pour chaque fenêtre
       trained_models_list = []    # Stocke les instances de modèles entraînés
       performance_scores_list = [] # Stocke les scores de performance
       drift_scores_list = []     # Stocke les scores de dérive

       # 1. Initialisation : Création des fenêtres
       # S'assurer que X est trié par index si c'est un DatetimeIndex
       if isinstance(X.index, pd.DatetimeIndex) and not X.index.is_monotonic_increasing:
           self.logger.warning("L'index de X n'est pas trié par ordre chronologique. Tri en cours.")
           X = X.sort_index()
           y = y.loc[X.index] # S'assurer que y est aligné après le tri de X

       windows_indices = self.create_training_windows(X.index) # Utilise les paramètres de la classe
       
       if not windows_indices:
           self.logger.warning("run_validation: Aucune fenêtre de validation générée. "
                               "Vérifiez les paramètres et la taille des données.")
           return WalkForwardResults(scores=[], drift_scores=[], windows=[], models=[])

       self.logger.info(f"run_validation: Démarrage avec {len(windows_indices)} fenêtres.")
       self.logger.info(f"  Fenêtre d'entraînement {'croissante' if self.expanding_training_window else 'glissante'}.")

       for i, (train_idx, test_idx) in enumerate(windows_indices):
           try:
               # Préparation des données pour la fenêtre actuelle
               X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
               y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
               
               if X_train_fold.empty or y_train_fold.empty:
                   self.logger.warning(f"Fenêtre {i+1}: Données d'entraînement vides. Saut.")
                   performance_scores_list.append(np.nan)
                   drift_scores_list.append(np.nan)
                   # Pas de modèle à ajouter, détails de fenêtre avec erreur ?
                   all_results_details.append({'window_num': i + 1, 'error': 'Empty training data',
                                               'performance_score': np.nan, 'concept_drift_score': np.nan})
                   continue

               self.logger.debug(f"Fenêtre {i+1}/{len(windows_indices)}: "
                                f"Train: {len(X_train_fold)} obs ({X_train_fold.index[0]} - {X_train_fold.index[-1]}), "
                                f"Test: {len(X_test_fold)} obs ({X_test_fold.index[0]} - {X_test_fold.index[-1]})")

               # 2a. Réentraînement du modèle
               model_instance = model_class(**model_params)
               model_instance.fit(X_train_fold, y_train_fold)
               trained_models_list.append(model_instance)

               # 2b. Évaluation du modèle
               score_fold = self.evaluate_model_performance(model_instance, X_test_fold, y_test_fold)
               performance_scores_list.append(score_fold)
               
               # Calcul de la dérive conceptuelle (optionnel)
               # Comparer la fin de l'entraînement avec le début du test, par exemple.
               drift_score_fold = self.detect_concept_drift(
                   X_train_fold.tail(len(X_test_fold)), # Comparer avec une portion de même taille
                   X_test_fold
               ) if not X_train_fold.empty and not X_test_fold.empty else np.nan
               drift_scores_list.append(drift_score_fold)
               
               # Stockage des détails de la fenêtre
               window_detail_dict = {
                   'window_num': i + 1,
                   'train_start_time': X_train_fold.index[0],
                   'train_end_time': X_train_fold.index[-1],
                   'test_start_time': X_test_fold.index[0],
                   'test_end_time': X_test_fold.index[-1],
                   'train_samples': len(X_train_fold),
                   'test_samples': len(X_test_fold),
                   'performance_score': score_fold,
                   'concept_drift_score': drift_score_fold,
               }
               all_results_details.append(window_detail_dict)
               
               self.logger.info(f"Fenêtre {i+1}/{len(windows_indices)}: Score={score_fold:.4f}, Dérive={drift_score_fold:.4f}")
               
           except Exception as e_fold:
               self.logger.error(f"Erreur dans la fenêtre {i+1} de run_validation: {e_fold}", exc_info=True)
               performance_scores_list.append(np.nan)
               drift_scores_list.append(np.nan)
               # Ne pas ajouter de modèle si l'entraînement a échoué.
               # Si l'erreur est après fit, model_instance pourrait exister.
               # Pour la cohérence, on n'ajoute pas de modèle en cas d'erreur dans la fenêtre.
               all_results_details.append({
                   'window_num': i + 1, 'error': str(e_fold),
                   'performance_score': np.nan, 'concept_drift_score': np.nan
               })
               continue
       
       # 3. Agrégation des résultats
       return WalkForwardResults(
           scores=performance_scores_list,
           drift_scores=drift_scores_list,
           windows=all_results_details,
           models=trained_models_list
       )


class AdaptiveModelManager:
    """
    Gestionnaire de modèles adaptatif pour le trading en temps réel
    """
    
    def __init__(self, walk_forward_validator: WalkForwardValidator):
        self.validator = walk_forward_validator
        self.active_models = {}
        self.model_performance_tracker = {}
        
    def get_or_create_model(self, 
                          symbol: str,
                          model_factory: Callable[[], BaseEstimator],
                          training_data: pd.DataFrame,
                          feature_columns: List[str],
                          target_column: str) -> BaseEstimator:
        """
        Récupère un modèle existant ou en crée un nouveau pour un symbole donné
        """
        model_key = f"{symbol}_model"
        
        # Essayer de charger un modèle existant
        if model_key not in self.active_models:
            loaded_model = self.validator.load_latest_model(model_key)
            if loaded_model is not None:
                self.active_models[model_key] = loaded_model
                self.validator.logger.info(f"Modèle chargé pour {symbol}")
            else:
                # Créer et entraîner un nouveau modèle
                new_model = model_factory()
                X_train = training_data[feature_columns].fillna(0)
                y_train = training_data[target_column].fillna(0)
                new_model.fit(X_train, y_train)
                
                # Sauvegarder le modèle
                self.validator.save_model(new_model, model_key, datetime.now())
                self.active_models[model_key] = new_model
                self.validator.logger.info(f"Nouveau modèle créé et entraîné pour {symbol}")
        
        return self.active_models[model_key]
    
    def update_model_performance(self, symbol: str, performance: float, timestamp: datetime):
        """
        Met à jour le tracking de performance pour un modèle
        """
        if symbol not in self.model_performance_tracker:
            self.model_performance_tracker[symbol] = []
        
        self.model_performance_tracker[symbol].append({
            'timestamp': timestamp,
            'performance': performance
        })
        
        # Garder seulement les 50 dernières performances
        if len(self.model_performance_tracker[symbol]) > 50:
            self.model_performance_tracker[symbol] = self.model_performance_tracker[symbol][-50:]
    
    def should_retrain_model(self, symbol: str) -> bool:
        """
        Détermine si un modèle doit être réentraîné basé sur sa performance récente
        """
        if symbol not in self.model_performance_tracker:
            return False
        
        performances = self.model_performance_tracker[symbol]
        if len(performances) < 10:  # Pas assez de données
            return False
        
        recent_performances = [p['performance'] for p in performances[-10:]]
        avg_recent = np.mean(recent_performances)
        
        return avg_recent < self.validator.performance_threshold
