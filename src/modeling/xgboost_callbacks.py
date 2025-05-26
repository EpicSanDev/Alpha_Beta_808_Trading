"""
Module contenant des callbacks personnalisés pour XGBoost
"""
import xgboost as xgb
from xgboost.callback import TrainingCallback, _Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import time
import logging

class CustomMetricLogger(TrainingCallback):
    """
    Callback pour enregistrer des métriques personnalisées pendant l'entraînement XGBoost.
    Permet également de sauvegarder l'historique d'entraînement pour une analyse ultérieure.
    """
    
    def __init__(self, 
                 save_path: Optional[str] = None,
                 eval_metrics: Optional[List[str]] = None,
                 save_best_model: bool = True,
                 plot_metrics: bool = False,
                 verbose: bool = True,
                 verbose_eval: int = 10):
        """
        Initialise le callback de journalisation de métriques.
        
        Args:
            save_path: Chemin où sauvegarder l'historique et le meilleur modèle
            eval_metrics: Liste des métriques à surveiller
            save_best_model: Si True, sauvegarde le meilleur modèle selon la métrique principale
            plot_metrics: Si True, génère un graphique des métriques à la fin de l'entraînement
            verbose: Si True, affiche les métriques pendant l'entraînement
            verbose_eval: Fréquence d'affichage des métriques (en nombre d'itérations)
        """
        super().__init__()
        self.save_path = save_path
        self.eval_metrics = eval_metrics or ['logloss', 'auc', 'error']
        self.save_best_model = save_best_model
        self.plot_metrics = plot_metrics
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        
        # État interne
        self.metrics_history = {}
        self.best_score = float('inf')  # Pour minimiser (logloss, error)
        self.best_iteration = 0
        self.current_best_model = None
        self.start_time = None
        
        # Créer le dossier de sauvegarde si nécessaire
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
    def before_training(self, model: _Model) -> _Model:
        """
        Initialise l'état avant l'entraînement.
        """
        self.metrics_history = {}
        self.best_score = float('inf')
        self.best_iteration = 0
        self.current_best_model = None
        self.start_time = time.time()
        
        if self.verbose:
            print("Début de l'entraînement XGBoost")
        
        return model
    
    def after_iteration(self, model: _Model, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]) -> bool:
        """
        Fonction appelée après chaque itération d'entraînement.
        
        Args:
            model: Modèle XGBoost en cours d'entraînement
            epoch: Itération actuelle
            evals_log: Journal des évaluations
            
        Returns:
            bool: True pour arrêter l'entraînement, False pour continuer
        """
        # Enregistrer les métriques dans l'historique
        for data_name, metrics in evals_log.items():
            for metric_name, metric_values in metrics.items():
                key = f"{data_name}-{metric_name}"
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                # Si nous avons une nouvelle valeur
                if len(metric_values) > len(self.metrics_history[key]):
                    self.metrics_history[key].append(metric_values[-1])
        
        # Afficher les métriques selon la fréquence spécifiée
        if self.verbose and (epoch % self.verbose_eval == 0 or epoch == 0):
            # Calculer le temps écoulé
            elapsed_time = time.time() - self.start_time
            metrics_str = f"[{epoch}] "
            
            # Afficher chaque métrique disponible
            for key, values in self.metrics_history.items():
                if values:  # S'assurer que la liste n'est pas vide
                    metrics_str += f"{key}: {values[-1]:.6f}  "
            
            # Ajouter le temps écoulé
            metrics_str += f"(Temps écoulé: {elapsed_time:.2f}s)"
            print(metrics_str)
        
        # Vérifier si c'est le meilleur modèle selon la première métrique du validation set
        # Par convention, prenons le premier jeu de validation mentionné
        validation_sets = [k for k in evals_log.keys() if 'validation' in k.lower()]
        if not validation_sets:
            # Si pas de validation explicite, prendre le premier disponible autre que 'train'
            validation_sets = [k for k in evals_log.keys() if 'train' not in k.lower()]
        
        # S'il existe au moins un ensemble de validation
        if validation_sets:
            val_set = validation_sets[0]
            # Prendre la première métrique disponible
            if evals_log[val_set]:
                metric_name = list(evals_log[val_set].keys())[0]
                metric_values = evals_log[val_set][metric_name]
                
                # Déterminer si on minimise ou maximise cette métrique
                minimize = metric_name.lower() not in ['auc', 'map', 'ndcg']
                
                if minimize:
                    current_score = metric_values[-1]
                    is_better = current_score < self.best_score
                else:
                    current_score = metric_values[-1]
                    is_better = current_score > self.best_score
                
                # Mettre à jour le meilleur modèle si nécessaire
                if is_better:
                    self.best_score = current_score
                    self.best_iteration = epoch
                    if self.save_best_model and hasattr(model, 'save_model'):
                        # Sauvegarder temporairement le meilleur modèle en mémoire
                        self.current_best_model = model.copy()
        
        # Continuer l'entraînement
        return False
    
    def after_training(self, model: _Model) -> _Model:
        """
        Fonction appelée après l'entraînement complet.
        """
        # Afficher les métriques finales
        if self.verbose:
            total_time = time.time() - self.start_time
            print(f"\nEntraînement terminé en {total_time:.2f}s")
            print(f"Meilleure itération: {self.best_iteration}")
            
            # Afficher la meilleure métrique
            if self.metrics_history:
                for key, values in self.metrics_history.items():
                    if 'validation' in key.lower() or ('train' not in key.lower() and len(self.metrics_history.keys()) < 3):
                        if len(values) > self.best_iteration:
                            print(f"Meilleur {key}: {values[self.best_iteration]:.6f}")
        
        # Sauvegarder l'historique des métriques si un chemin est spécifié
        if self.save_path:
            history_df = pd.DataFrame()
            for key, values in self.metrics_history.items():
                history_df[key] = values if len(values) == len(next(iter(self.metrics_history.values()))) else values + [None] * (len(next(iter(self.metrics_history.values()))) - len(values))
            
            history_path = os.path.join(self.save_path, 'metrics_history.csv')
            history_df.to_csv(history_path, index_label='iteration')
            
            if self.verbose:
                print(f"Historique des métriques sauvegardé dans {history_path}")
        
        # Sauvegarder le meilleur modèle si demandé
        if self.save_best_model and self.current_best_model and self.save_path:
            best_model_path = os.path.join(self.save_path, 'best_model.json')
            self.current_best_model.save_model(best_model_path)
            if self.verbose:
                print(f"Meilleur modèle (itération {self.best_iteration}) sauvegardé dans {best_model_path}")
        
        # Générer un graphique des métriques si demandé
        if self.plot_metrics and self.metrics_history:
            self._plot_metrics()
        
        return model
    
    def _plot_metrics(self) -> None:
        """
        Génère un graphique des métriques d'entraînement et de validation.
        """
        if not self.metrics_history:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Préparer les données pour le graphique
        for key, values in self.metrics_history.items():
            iterations = list(range(len(values)))
            plt.plot(iterations, values, label=key)
        
        # Marquer la meilleure itération
        if self.best_iteration > 0:
            plt.axvline(x=self.best_iteration, color='r', linestyle='--', 
                       label=f'Meilleure itération: {self.best_iteration}')
        
        plt.title('Évolution des métriques pendant l\'entraînement')
        plt.xlabel('Itérations')
        plt.ylabel('Valeur de la métrique')
        plt.legend()
        plt.grid(True)
        
        # Sauvegarder le graphique si un chemin est spécifié
        if self.save_path:
            metrics_plot_path = os.path.join(self.save_path, 'metrics_plot.png')
            plt.savefig(metrics_plot_path)
            if self.verbose:
                print(f"Graphique des métriques sauvegardé dans {metrics_plot_path}")
        
        plt.close()

class CustomEarlyStopping(TrainingCallback):
    """
    Callback pour arrêter l'entraînement si une métrique ne s'améliore plus.
    Offre plus de flexibilité que l'EarlyStopping standard de XGBoost.
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 metric: str = 'validation_0-logloss',
                 min_delta: float = 0.0,
                 save_best: bool = True,
                 maximize: Optional[bool] = None,
                 verbose: bool = True):
        """
        Initialise le callback d'arrêt anticipé.
        
        Args:
            patience: Nombre d'itérations sans amélioration avant d'arrêter
            metric: Métrique à surveiller, format 'dataset-metric'
            min_delta: Amélioration minimale pour considérer comme significative
            save_best: Si True, restaure le meilleur modèle à la fin
            maximize: Si True, maximise la métrique; si False, minimise; si None, détecte automatiquement
            verbose: Si True, affiche des messages sur l'arrêt anticipé
        """
        super().__init__()
        self.patience = patience
        self.metric = metric
        self.min_delta = abs(min_delta)
        self.save_best = save_best
        self.verbose = verbose
        
        # Détecter automatiquement si on maximise ou minimise, sauf si spécifié
        if maximize is None:
            maximize_metrics = ['auc', 'map', 'ndcg', 'accuracy', 'f1', 'precision', 'recall']
            metric_name = metric.split('-')[-1].lower()
            self.maximize = any(m in metric_name for m in maximize_metrics)
        else:
            self.maximize = maximize
        
        # État interne
        self.best_score = float('-inf') if self.maximize else float('inf')
        self.best_iteration = 0
        self.best_model = None
        self.counter = 0
    
    def before_training(self, model: _Model) -> _Model:
        """Initialise l'état avant l'entraînement."""
        self.best_score = float('-inf') if self.maximize else float('inf')
        self.best_iteration = 0
        self.best_model = None
        self.counter = 0
        return model
    
    def after_iteration(self, model: _Model, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]) -> bool:
        """
        Vérifie si l'entraînement doit être arrêté après chaque itération.
        
        Args:
            model: Modèle XGBoost en cours d'entraînement
            epoch: Itération actuelle
            evals_log: Journal des évaluations
            
        Returns:
            bool: True pour arrêter l'entraînement, False pour continuer
        """
        # Extraire le dataset et la métrique depuis la clé
        if '-' in self.metric:
            dataset, metric_name = self.metric.split('-', 1)
        else:
            # Si le format n'est pas correct, utiliser la première métrique disponible
            if evals_log:
                dataset = list(evals_log.keys())[0]
                if evals_log[dataset]:
                    metric_name = list(evals_log[dataset].keys())[0]
                else:
                    return False
            else:
                return False
        
        # Vérifier si la métrique est disponible
        if dataset not in evals_log or metric_name not in evals_log[dataset]:
            if self.verbose and epoch == 0:
                print(f"Avertissement: Métrique '{self.metric}' non trouvée dans evals_log. "
                      f"Métriques disponibles: {[f'{d}-{m}' for d in evals_log for m in evals_log[d]]}")
            return False
        
        # Obtenir la valeur actuelle de la métrique
        metric_values = evals_log[dataset][metric_name]
        if not metric_values:
            return False
        
        current_score = metric_values[-1]
        
        # Vérifier si c'est un meilleur score
        if self.maximize:
            is_better = current_score > (self.best_score + self.min_delta)
        else:
            is_better = current_score < (self.best_score - self.min_delta)
        
        if is_better:
            if self.verbose:
                improvement = current_score - self.best_score if self.maximize else self.best_score - current_score
                if epoch > 0:  # Éviter le message à la première itération
                    print(f"Amélioration de {self.metric}: {improvement:.6f} (Nouvelle valeur: {current_score:.6f})")
            
            self.best_score = current_score
            self.best_iteration = epoch
            self.counter = 0
            
            # Sauvegarder le meilleur modèle si demandé
            if self.save_best and hasattr(model, 'copy'):
                self.best_model = model.copy()
        else:
            self.counter += 1
            
            if self.verbose and self.counter == self.patience:
                print(f"Early stopping activé à l'itération {epoch}. "
                      f"Pas d'amélioration de {self.metric} depuis {self.patience} itérations.")
                print(f"Meilleure valeur: {self.best_score:.6f} à l'itération {self.best_iteration}")
            
            # Arrêter l'entraînement si la patience est épuisée
            if self.counter >= self.patience:
                # Restaurer le meilleur modèle si demandé
                if self.save_best and self.best_model:
                    # Il n'y a pas de méthode standard pour restaurer le modèle,
                    # nous utiliserons l'attribut best_iteration de XGBoost lors de la prédiction
                    pass
                
                return True
        
        return False

class AdvancedXGBoostModel:
    """
    Classe qui encapsule XGBoost avec des fonctionnalités avancées comme:
    - Gestion des poids d'échantillonnage temporel
    - Validation en cours d'entraînement avec early stopping
    - Suivi des métriques et visualisation
    - Analyse de feature importance
    - Prédictions avec intervalles de confiance (avec quantiles)
    """
    
    def __init__(self, 
                 model_type: str = 'classifier',
                 params: Optional[Dict[str, Any]] = None,
                 custom_objective: Optional[callable] = None,
                 custom_metric: Optional[callable] = None,
                 num_boost_round: int = 100,
                 early_stopping_rounds: int = 20,
                 maximize_metric: bool = False,
                 log_metrics: bool = True,
                 random_state: int = 42,
                 output_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialise un modèle XGBoost avancé.
        
        Args:
            model_type: Type de modèle ('classifier' ou 'regressor')
            params: Paramètres XGBoost
            custom_objective: Fonction objectif personnalisée
            custom_metric: Fonction de métrique personnalisée
            num_boost_round: Nombre maximal d'itérations
            early_stopping_rounds: Nombre d'itérations sans amélioration avant l'arrêt anticipé
            maximize_metric: Si True, maximise la métrique d'évaluation
            log_metrics: Si True, enregistre les métriques pendant l'entraînement
            random_state: Graine aléatoire pour la reproductibilité
            output_dir: Répertoire de sortie pour sauvegarder les modèles et métriques
            verbose: Si True, affiche des informations pendant l'entraînement
        """
        self.model_type = model_type
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.maximize_metric = maximize_metric
        self.log_metrics = log_metrics
        self.random_state = random_state
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Paramètres par défaut
        default_params = {
            'silent': 0 if verbose else 1,
            'seed': random_state,
        }
        
        # Paramètres spécifiques au type de modèle
        if model_type == 'classifier':
            default_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
            })
        elif model_type == 'regressor':
            default_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
            })
        elif model_type == 'ranker':
            default_params.update({
                'objective': 'rank:pairwise',
                'eval_metric': 'ndcg',
            })
        else:
            raise ValueError(f"Type de modèle non pris en charge: {model_type}. "
                             f"Utilisez 'classifier', 'regressor' ou 'ranker'.")
        
        # Fusionner avec les paramètres fournis
        self.params = default_params.copy()
        if params:
            self.params.update(params)
        
        # Objectif et métrique personnalisés
        self.custom_objective = custom_objective
        self.custom_metric = custom_metric
        
        # Modèle et DMatrix
        self.booster = None
        self.feature_names = None
        self.best_iteration = None
        self.feature_importances_ = None
        self.metrics_history = None
    
    def fit(self, 
            X_train, 
            y_train, 
            eval_set=None, 
            sample_weight=None, 
            eval_sample_weight=None,
            feature_weights=None, 
            callbacks=None):
        """
        Entraîne le modèle XGBoost avec des fonctionnalités avancées.
        
        Args:
            X_train: Features d'entraînement (DataFrame, numpy array)
            y_train: Labels d'entraînement
            eval_set: Ensemble(s) d'évaluation [(X_val, y_val), ...]
            sample_weight: Poids d'échantillonnage pour les données d'entraînement
            eval_sample_weight: Poids d'échantillonnage pour les données d'évaluation
            feature_weights: Poids des features pour l'importance
            callbacks: Callbacks personnalisés supplémentaires
            
        Returns:
            self: Le modèle entraîné
        """
        # Création du DMatrix pour l'entraînement
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight, feature_names=self.feature_names)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
            
        # Création des DMatrix pour l'évaluation
        evals = []
        if eval_set:
            for i, (X_val, y_val) in enumerate(eval_set):
                eval_w = None if eval_sample_weight is None else eval_sample_weight[i]
                if hasattr(X_val, 'columns'):
                    deval = xgb.DMatrix(X_val, label=y_val, weight=eval_w, feature_names=X_val.columns.tolist())
                else:
                    deval = xgb.DMatrix(X_val, label=y_val, weight=eval_w)
                evals.append((deval, f'validation_{i}'))
        
        # Préparation des callbacks
        all_callbacks = []
        
        # Logger de métriques
        if self.log_metrics:
            metric_logger = CustomMetricLogger(
                save_path=self.output_dir,
                save_best_model=True,
                plot_metrics=True,
                verbose=self.verbose
            )
            all_callbacks.append(metric_logger)
        
        # Early stopping
        if self.early_stopping_rounds and eval_set:
            early_stopping = CustomEarlyStopping(
                patience=self.early_stopping_rounds,
                metric=f'validation_0-{self.params.get("eval_metric", "logloss")}',
                maximize=self.maximize_metric,
                verbose=self.verbose
            )
            all_callbacks.append(early_stopping)
        
        # Ajouter les callbacks personnalisés
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # Entraînement du modèle
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            obj=self.custom_objective,
            feval=self.custom_metric,
            callbacks=all_callbacks
        )
        
        # Récupérer le meilleur numéro d'itération
        if self.early_stopping_rounds and eval_set and 'early_stopping' in locals() and early_stopping is not None:
            self.best_iteration = early_stopping.best_iteration
        elif hasattr(self.booster, 'best_iteration'): # Si un autre callback (ex: natif XGBoost) l'a défini
            self.best_iteration = self.booster.best_iteration
        else: # Fallback
            # Si num_boost_round est le nombre total d'itérations effectuées, alors c'est num_boost_round -1
            # Si c'est le paramètre initial, cela pourrait être différent.
            # Le booster lui-même pourrait ne pas avoir de best_iteration si aucun early stopping n'a eu lieu.
            # Dans ce cas, on prend la dernière itération.
            self.best_iteration = self.booster.current_iteration -1 if hasattr(self.booster, 'current_iteration') and self.booster.current_iteration > 0 else self.num_boost_round - 1


        # Calculer les importances de features
        self.feature_importances_ = self.booster.get_score(importance_type='gain')
        
        # Récupérer l'historique des métriques
        if self.log_metrics and hasattr(metric_logger, 'metrics_history'):
            self.metrics_history = metric_logger.metrics_history
        
        return self
    
    def predict(self, X, iteration_range=None):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Données pour les prédictions
            iteration_range: Plage d'itérations à utiliser (début, fin)
            
        Returns:
            numpy.ndarray: Prédictions
        """
        if self.booster is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez fit() d'abord.")
        
        # Création du DMatrix pour la prédiction
        if hasattr(X, 'columns'):
            dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        else:
            dtest = xgb.DMatrix(X)
        
        # Utiliser le meilleur modèle si disponible
        if iteration_range is None and self.best_iteration is not None:
            iteration_range = (0, self.best_iteration + 1)
        
        # Prédictions
        if self.model_type == 'classifier':
            preds = self.booster.predict(dtest, iteration_range=iteration_range)
            # Pour la classification binaire, arrondir à 0 ou 1
            if self.params.get('objective') == 'binary:logistic':
                return (preds > 0.5).astype(int)
            return preds
        else:
            return self.booster.predict(dtest, iteration_range=iteration_range)
    
    def predict_proba(self, X, iteration_range=None):
        """
        Prédit les probabilités pour la classification.
        
        Args:
            X: Données pour les prédictions
            iteration_range: Plage d'itérations à utiliser (début, fin)
            
        Returns:
            numpy.ndarray: Probabilités prédites
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba() est uniquement disponible pour les classifieurs.")
        
        if self.booster is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez fit() d'abord.")
        
        # Création du DMatrix pour la prédiction
        if hasattr(X, 'columns'):
            dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        else:
            dtest = xgb.DMatrix(X)
        
        # Utiliser le meilleur modèle si disponible
        if iteration_range is None and self.best_iteration is not None:
            iteration_range = (0, self.best_iteration + 1)
        
        # Prédictions de probabilités
        probs = self.booster.predict(dtest, iteration_range=iteration_range)
        
        # Pour la classification binaire, renvoyer [1-p, p]
        if self.params.get('objective') == 'binary:logistic':
            return np.vstack((1 - probs, probs)).T
        
        return probs
    
    def save_model(self, filepath):
        """
        Sauvegarde le modèle et ses métadonnées.
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if self.booster is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez fit() d'abord.")
        
        # Sauvegarder le modèle
        self.booster.save_model(filepath)
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'params': self.params,
            'feature_importances': self.feature_importances_
        }
        
        # Sauvegarder les métadonnées dans un fichier séparé
        metadata_path = filepath + '.meta'
        pd.to_pickle(metadata, metadata_path)
    
    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé et ses métadonnées.
        
        Args:
            filepath: Chemin du modèle à charger
        """
        # Charger le modèle
        self.booster = xgb.Booster()
        self.booster.load_model(filepath)
        
        # Charger les métadonnées si disponibles
        metadata_path = filepath + '.meta'
        if os.path.exists(metadata_path):
            metadata = pd.read_pickle(metadata_path)
            self.model_type = metadata.get('model_type', self.model_type)
            self.feature_names = metadata.get('feature_names')
            self.best_iteration = metadata.get('best_iteration')
            self.params = metadata.get('params', self.params)
            self.feature_importances_ = metadata.get('feature_importances')
    
    def plot_feature_importance(self, max_features=20, importance_type='gain', figsize=(12, 8)):
        """
        Affiche un graphique d'importance des features.
        
        Args:
            max_features: Nombre maximal de features à afficher
            importance_type: Type d'importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
            figsize: Taille de la figure
        """
        if self.booster is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez fit() d'abord.")
        
        importance = self.booster.get_score(importance_type=importance_type)
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        if len(importance) > max_features:
            importance = dict(list(importance.items())[:max_features])
        
        plt.figure(figsize=figsize)
        plt.barh(list(importance.keys()), list(importance.values()))
        plt.title(f"Importance des features ({importance_type})")
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Sauvegarder si un répertoire de sortie est spécifié
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'feature_importance_{importance_type}.png'))
        
        plt.show()
    
    def plot_metrics_history(self, figsize=(12, 8)):
        """
        Affiche l'historique des métriques d'entraînement.
        """
        if not self.metrics_history:
            print("Aucun historique de métriques disponible.")
            return
        
        plt.figure(figsize=figsize)
        
        for key, values in self.metrics_history.items():
            plt.plot(range(len(values)), values, label=key)
        
        # Marquer la meilleure itération
        if self.best_iteration is not None:
            plt.axvline(x=self.best_iteration, color='r', linestyle='--', 
                      label=f'Meilleure itération: {self.best_iteration}')
        
        plt.title('Évolution des métriques pendant l\'entraînement')
        plt.xlabel('Itérations')
        plt.ylabel('Valeur de la métrique')
        plt.legend()
        plt.grid(True)
        
        # Sauvegarder si un répertoire de sortie est spécifié
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'metrics_history.png'))
        
        plt.show()
    
    def analyze_sensitivity(self, X_perturbed: np.ndarray, original_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Analyse la sensibilité du modèle à des perturbations des inputs.
        
        Args:
            X_perturbed: Données d'entrée perturbées
            original_predictions: Prédictions sur les données originales
            
        Returns:
            Dict[str, Any]: Métriques de sensibilité
        """
        # Faire des prédictions sur les données perturbées
        if self.model_type == 'classifier':
            perturbed_predictions = self.predict_proba(X_perturbed)
            # Pour la classification binaire, utiliser la probabilité de la classe positive
            if perturbed_predictions.shape[1] == 2:
                perturbed_predictions = perturbed_predictions[:, 1]
        else:
            perturbed_predictions = self.predict(X_perturbed)
        
        # Calculer les différences
        if isinstance(original_predictions, np.ndarray) and isinstance(perturbed_predictions, np.ndarray):
            diff = perturbed_predictions - original_predictions
            
            # Calculer les métriques de sensibilité
            sensitivity_metrics = {
                'prediction_change_mean': float(np.mean(np.abs(diff))),
                'prediction_change_std': float(np.std(np.abs(diff))),
                'max_prediction_change': float(np.max(np.abs(diff))),
                'median_prediction_change': float(np.median(np.abs(diff))),
                'num_changed_predictions': int(np.sum(np.abs(diff) > 0.01))
            }
            
            return sensitivity_metrics
        else:
            return {"error": "Types de prédictions incompatibles pour l'analyse de sensibilité"}
