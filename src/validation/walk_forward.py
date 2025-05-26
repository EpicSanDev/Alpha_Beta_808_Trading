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
                 model_save_path: str = "models_store/walk_forward"):
        """
        Initialise le validateur walk-forward
        
        Args:
            training_window_days: Nombre de jours pour la fenêtre d'entraînement
            validation_window_days: Nombre de jours pour la fenêtre de validation
            retrain_frequency_days: Fréquence de réentraînement en jours
            min_training_samples: Nombre minimum d'échantillons pour l'entraînement
            performance_threshold: Seuil de performance en dessous duquel on réentraîne
            drift_threshold: Seuil de dérive conceptuelle
            model_save_path: Chemin pour sauvegarder les modèles
        """
        self.training_window_days = training_window_days
        self.validation_window_days = validation_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.min_training_samples = min_training_samples
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.model_save_path = Path(model_save_path)
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
            drift_scores = []
            
            for column in historical_features.columns:
                if pd.api.types.is_numeric_dtype(historical_features[column]):
                    # Test de Kolmogorov-Smirnov pour chaque feature
                    ks_stat, p_value = ks_2samp(
                        historical_features[column].dropna(),
                        recent_features[column].dropna()
                    )
                    drift_scores.append(ks_stat)
                    
            return np.mean(drift_scores) if drift_scores else 0.0
            
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
                _, bins = pd.cut(historical_data[column], bins=num_bins, retbins=True, duplicates='drop')
                
                # Distribution historique
                hist_counts = pd.cut(historical_data[column], bins=bins, include_lowest=True).value_counts()
                hist_dist = hist_counts / hist_counts.sum()
                
                # Distribution récente
                recent_counts = pd.cut(recent_data[column], bins=bins, include_lowest=True).value_counts()
                recent_dist = recent_counts / recent_counts.sum()
                
                # Calcul PSI
                psi = 0
                for i in range(len(hist_dist)):
                    if hist_dist.iloc[i] > 0 and recent_dist.iloc[i] > 0:
                        psi += (recent_dist.iloc[i] - hist_dist.iloc[i]) * np.log(recent_dist.iloc[i] / hist_dist.iloc[i])
                
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
                return accuracy_score(y_test, predictions)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation: {e}")
            return 0.0
    
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
        
        joblib.dump(model, filepath)
        self.logger.info(f"Modèle sauvegardé: {filepath}")
        
        return str(filepath)
    
    def load_latest_model(self, model_name: str) -> Optional[BaseEstimator]:
        """
        Charge le modèle le plus récent pour un nom donné
        
        Args:
            model_name: Nom du modèle à charger
            
        Returns:
            Modèle chargé ou None si aucun modèle trouvé
        """
        pattern = f"{model_name}_*.joblib"
        model_files = list(self.model_save_path.glob(pattern))
        
        if not model_files:
            return None
            
        # Trier par date de modification et prendre le plus récent
        latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
        
        try:
            model = joblib.load(latest_file)
            self.logger.info(f"Modèle chargé: {latest_file}")
            return model
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle {latest_file}: {e}")
            return None
    
    def run_walk_forward_validation(self,
                                  data: pd.DataFrame,
                                  target_column: str,
                                  feature_columns: List[str],
                                  model_factory: Callable[[], BaseEstimator],
                                  date_column: str = 'timestamp',
                                  model_name: str = 'walk_forward_model') -> Dict[str, Any]:
        """
        Lance la validation walk-forward complète
        
        Args:
            data: DataFrame avec toutes les données
            target_column: Nom de la colonne cible
            feature_columns: Liste des colonnes de features
            model_factory: Fonction qui retourne un nouveau modèle
            date_column: Nom de la colonne de date
            model_name: Nom du modèle pour la sauvegarde
            
        Returns:
            Dictionnaire avec les résultats de la validation
        """
        # Trier les données par date
        data = data.sort_values(date_column).copy()
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Initialiser les variables de tracking
        results = {
            'performance_history': [],
            'retrain_dates': [],
            'drift_scores': [],
            'predictions': [],
            'model_versions': []
        }
        
        current_model = None
        last_retrain_date = None
        validation_start_idx = 0
        
        self.logger.info(f"Démarrage de la validation walk-forward sur {len(data)} échantillons")
        
        # Boucle principale de validation
        while validation_start_idx < len(data) - self.validation_window_days:
            
            # Calculer les indices pour les fenêtres
            validation_end_idx = min(validation_start_idx + self.validation_window_days, len(data))
            training_end_idx = validation_start_idx
            training_start_idx = max(0, training_end_idx - self.training_window_days)
            
            # Extraire les données
            training_data = data.iloc[training_start_idx:training_end_idx]
            validation_data = data.iloc[validation_start_idx:validation_end_idx]
            
            if len(training_data) < self.min_training_samples:
                validation_start_idx += self.retrain_frequency_days
                continue
                
            current_date = validation_data[date_column].iloc[0]
            
            # Préparer les features et targets
            X_train = training_data[feature_columns].fillna(0)
            y_train = training_data[target_column].fillna(0)
            X_val = validation_data[feature_columns].fillna(0)
            y_val = validation_data[target_column].fillna(0)
            
            # Calculer la dérive conceptuelle si on a un modèle
            drift_score = None
            if current_model is not None and len(self.performance_history) > 0:
                # Utiliser les dernières données d'entraînement comme référence historique
                historical_window = max(50, self.training_window_days // 4)
                if len(training_data) >= historical_window * 2:
                    historical_features = training_data.iloc[-historical_window*2:-historical_window][feature_columns]
                    recent_features = training_data.iloc[-historical_window:][feature_columns]
                    drift_score = self.calculate_concept_drift(historical_features, recent_features)
                    results['drift_scores'].append({
                        'date': current_date,
                        'drift_score': drift_score
                    })
            
            # Calculer la performance récente
            recent_performance = None
            if len(self.performance_history) > 0:
                recent_performances = [p['score'] for p in self.performance_history[-5:]]  # 5 dernières performances
                recent_performance = np.mean(recent_performances)
            
            # Décider si on doit réentraîner
            should_retrain = self.should_retrain(current_date, last_retrain_date, recent_performance, drift_score)
            
            if should_retrain or current_model is None:
                self.logger.info(f"Réentraînement du modèle à la date {current_date}")
                
                # Créer et entraîner un nouveau modèle
                current_model = model_factory()
                current_model.fit(X_train, y_train)
                
                # Sauvegarder le modèle
                model_path = self.save_model(current_model, model_name, current_date)
                
                # Mettre à jour les variables de tracking
                last_retrain_date = current_date
                results['retrain_dates'].append(current_date)
                results['model_versions'].append({
                    'date': current_date,
                    'model_path': model_path,
                    'training_samples': len(training_data)
                })
            
            # Faire des prédictions sur la fenêtre de validation
            try:
                predictions = current_model.predict(X_val)
                probabilities = None
                if hasattr(current_model, 'predict_proba'):
                    probabilities = current_model.predict_proba(X_val)
                
                # Évaluer la performance
                performance = self.evaluate_model_performance(current_model, X_val, y_val)
                
                # Stocker les résultats
                self.performance_history.append({
                    'date': current_date,
                    'score': performance,
                    'training_samples': len(training_data),
                    'validation_samples': len(validation_data)
                })
                
                results['performance_history'].append({
                    'date': current_date,
                    'performance': performance,
                    'training_samples': len(training_data),
                    'drift_score': drift_score
                })
                
                # Stocker les prédictions
                for i, (idx, row) in enumerate(validation_data.iterrows()):
                    pred_result = {
                        'date': row[date_column],
                        'actual': row[target_column],
                        'predicted': predictions[i],
                        'performance': performance
                    }
                    if probabilities is not None:
                        pred_result['probabilities'] = probabilities[i]
                    
                    results['predictions'].append(pred_result)
                
                self.logger.info(f"Performance à {current_date}: {performance:.4f}")
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la prédiction à {current_date}: {e}")
            
            # Avancer la fenêtre
            validation_start_idx += self.retrain_frequency_days
        
        # Résumé final
        if results['performance_history']:
            avg_performance = np.mean([p['performance'] for p in results['performance_history']])
            std_performance = np.std([p['performance'] for p in results['performance_history']])
            
            self.logger.info(f"Validation walk-forward terminée:")
            self.logger.info(f"  - Performance moyenne: {avg_performance:.4f} ± {std_performance:.4f}")
            self.logger.info(f"  - Nombre de réentraînements: {len(results['retrain_dates'])}")
            self.logger.info(f"  - Nombre de prédictions: {len(results['predictions'])}")
            
            results['summary'] = {
                'avg_performance': avg_performance,
                'std_performance': std_performance,
                'num_retrains': len(results['retrain_dates']),
                'num_predictions': len(results['predictions'])
            }
        
        return results
    
    def create_training_windows(self, data_index: pd.Index) -> List[Tuple[List[int], List[int]]]:
        """
        Crée les fenêtres d'entraînement et de test pour la validation walk-forward
        
        Args:
            data_index: Index temporel du dataset
            
        Returns:
            Liste de tuples (indices_train, indices_test) pour chaque fenêtre
        """
        windows = []
        
        # Convertir en nombre d'échantillons si nécessaire
        total_samples = len(data_index)
        training_samples = min(self.training_window_days, total_samples // 2)
        validation_samples = min(self.validation_window_days, total_samples // 10)
        step_size = max(1, self.retrain_frequency_days)
        
        start_idx = 0
        
        while start_idx + training_samples + validation_samples <= total_samples:
            # Indices d'entraînement
            train_start = start_idx
            train_end = start_idx + training_samples
            
            # Indices de test
            test_start = train_end
            test_end = test_start + validation_samples
            
            # Vérifier qu'on a assez de données
            if train_end - train_start >= self.min_training_samples:
                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))
                windows.append((train_indices, test_indices))
            
            # Avancer la fenêtre
            start_idx += step_size
        
        return windows
    
    def detect_concept_drift(self, X_historical: pd.DataFrame, X_recent: pd.DataFrame) -> float:
        """
        Détecte la dérive conceptuelle entre données historiques et récentes
        
        Args:
            X_historical: Features des données historiques
            X_recent: Features des données récentes
            
        Returns:
            Score de dérive conceptuelle (0 = pas de dérive, 1 = dérive maximale)
        """
        return self.calculate_concept_drift(X_historical, X_recent)
    
    def run_validation(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     model_class,
                     model_params: Dict = None) -> 'WalkForwardResults':
        """
        Effectue une validation walk-forward complète
        
        Args:
            X: Features du dataset
            y: Variable cible
            model_class: Classe du modèle à entraîner
            model_params: Paramètres du modèle
            
        Returns:
            Résultats de la validation walk-forward
        """
        if model_params is None:
            model_params = {}
            
        results = []
        windows = self.create_training_windows(X.index)
        
        self.logger.info(f"Démarrage de la validation walk-forward avec {len(windows)} fenêtres")
        
        for i, (train_idx, test_idx) in enumerate(windows):
            try:
                # Données d'entraînement et de test
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Entraînement du modèle
                model = model_class(**model_params)
                model.fit(X_train, y_train)
                
                # Évaluation
                score = self.evaluate_model_performance(model, X_test, y_test)
                
                # Détection de dérive conceptuelle
                drift_score = self.detect_concept_drift(X_train, X_test)
                
                # Stockage des résultats
                window_result = {
                    'window': i + 1,
                    'train_start': X_train.index[0],
                    'train_end': X_train.index[-1],
                    'test_start': X_test.index[0],
                    'test_end': X_test.index[-1],
                    'score': score,
                    'drift_score': drift_score,
                    'model': model
                }
                
                results.append(window_result)
                
                self.logger.info(f"Fenêtre {i+1}/{len(windows)}: Score={score:.4f}, Dérive={drift_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Erreur dans la fenêtre {i+1}: {e}")
                continue
        
        # Agrégation des résultats
        return WalkForwardResults(
            scores=[r['score'] for r in results],
            drift_scores=[r['drift_score'] for r in results],
            windows=results,
            models=[r['model'] for r in results]
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
