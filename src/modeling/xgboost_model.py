import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
import logging
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('XGBoostModel')

class XGBoostCustomCallback:
    """
    Classe pour créer des callbacks personnalisés pour XGBoost.
    Permet d'ajouter des fonctionnalités comme:
    - Arrêt anticipé basé sur plusieurs métriques
    - Logging des performances
    - Sauvegarde des meilleurs modèles intermédiaires
    - Détection de stagnation
    """
    def __init__(self, early_stopping_rounds: int = 10,
                 metrics_to_monitor: List[str] = None,
                 save_snapshots: bool = False,
                 snapshot_dir: str = './model_snapshots',
                 verbose: bool = True,
                 verbose_eval: int = 10,
                 decay_rate_threshold: float = 0.001):
        """
        Initialise le callback personnalisé.
        
        Args:
            early_stopping_rounds: Nombre de rounds sans amélioration avant arrêt.
            metrics_to_monitor: Liste des métriques à surveiller (ex: ['auc', 'logloss']).
            save_snapshots: Si True, sauvegarde les modèles intermédiaires.
            snapshot_dir: Répertoire où sauvegarder les snapshots.
            verbose: Si True, affiche des logs.
            verbose_eval: Fréquence d'affichage des logs (tous les N rounds).
            decay_rate_threshold: Seuil pour détecter une stagnation de l'amélioration.
        """
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics_to_monitor = metrics_to_monitor if metrics_to_monitor else ['logloss']
        self.save_snapshots = save_snapshots
        self.snapshot_dir = snapshot_dir
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.decay_rate_threshold = decay_rate_threshold
        
        # Variables internes
        self.best_scores = {metric: float('-inf') if 'auc' in metric.lower() else float('inf') 
                            for metric in self.metrics_to_monitor}
        self.best_iterations = {metric: 0 for metric in self.metrics_to_monitor}
        self.rounds_without_improvement = {metric: 0 for metric in self.metrics_to_monitor}
        self.history = {metric: [] for metric in self.metrics_to_monitor}
        self.start_time = None
        self.best_models = {}
    
    def __call__(self, env):
        """
        Méthode appelée à chaque itération d'entraînement.
        
        Args:
            env: Environnement XGBoost contenant les informations d'entraînement.
        
        Returns:
            bool: True si l'entraînement doit continuer, False sinon.
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Récupérer l'évaluation actuelle
        current_iter = env.iteration
        evaluation_result = env.evaluation_result_list
        
        # Traiter chaque métrique
        should_stop = False
        for evaluation in evaluation_result:
            dataset_name, metric, value, is_higher_better = evaluation
            if metric not in self.metrics_to_monitor:
                continue
            
            # Ajouter à l'historique
            self.history[metric].append(value)
            
            # Vérifier si c'est le meilleur score
            is_best = False
            if is_higher_better and value > self.best_scores[metric]:
                is_best = True
                self.best_scores[metric] = value
                self.best_iterations[metric] = current_iter
                self.rounds_without_improvement[metric] = 0
                if self.save_snapshots:
                    self.best_models[metric] = env.model.copy()
            elif not is_higher_better and value < self.best_scores[metric]:
                is_best = True
                self.best_scores[metric] = value
                self.best_iterations[metric] = current_iter
                self.rounds_without_improvement[metric] = 0
                if self.save_snapshots:
                    self.best_models[metric] = env.model.copy()
            else:
                self.rounds_without_improvement[metric] += 1
            
            # Vérifier early stopping pour cette métrique
            if self.rounds_without_improvement[metric] >= self.early_stopping_rounds:
                should_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered by {metric} at iteration {current_iter}. "
                               f"Best score: {self.best_scores[metric]:.6f} at iteration {self.best_iterations[metric]}.")
            
            # Afficher les logs
            if self.verbose and (current_iter % self.verbose_eval == 0 or is_best):
                elapsed_time = time.time() - self.start_time
                improvement = ""
                if is_best:
                    improvement = " (New best)"
                logger.info(f"[{current_iter}] {dataset_name}-{metric}: {value:.6f}{improvement} "
                           f"[Best: {self.best_scores[metric]:.6f} @ {self.best_iterations[metric]}] "
                           f"Time: {elapsed_time:.2f}s")
        
        # Analyser la courbe d'apprentissage pour détecter une stagnation
        if current_iter > 10:  # Attendre quelques itérations pour avoir assez de données
            for metric in self.metrics_to_monitor:
                if len(self.history[metric]) > 10:
                    recent_values = self.history[metric][-10:]
                    if is_higher_better:
                        recent_improvements = [max(0, recent_values[i] - recent_values[i-1]) for i in range(1, len(recent_values))]
                    else:
                        recent_improvements = [max(0, recent_values[i-1] - recent_values[i]) for i in range(1, len(recent_values))]
                    
                    avg_improvement = sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0
                    if avg_improvement < self.decay_rate_threshold:
                        # La courbe d'apprentissage stagne
                        if self.verbose and current_iter % self.verbose_eval == 0:
                            logger.info(f"Learning curve for {metric} is flattening (avg improvement: {avg_improvement:.6f})")
        
        return not should_stop
    
    def get_best_iteration(self, metric=None):
        """
        Retourne la meilleure itération pour une métrique donnée.
        Si aucune métrique n'est spécifiée, retourne la meilleure itération pour la première métrique.
        
        Args:
            metric: Métrique pour laquelle récupérer la meilleure itération.
        
        Returns:
            int: Meilleure itération.
        """
        if metric is None:
            metric = self.metrics_to_monitor[0]
        return self.best_iterations.get(metric, 0)
    
    def get_best_score(self, metric=None):
        """
        Retourne le meilleur score pour une métrique donnée.
        Si aucune métrique n'est spécifiée, retourne le meilleur score pour la première métrique.
        
        Args:
            metric: Métrique pour laquelle récupérer le meilleur score.
        
        Returns:
            float: Meilleur score.
        """
        if metric is None:
            metric = self.metrics_to_monitor[0]
        return self.best_scores.get(metric, float('-inf'))
    
    def get_best_model(self, metric=None):
        """
        Retourne le meilleur modèle pour une métrique donnée.
        Si aucune métrique n'est spécifiée, retourne le meilleur modèle pour la première métrique.
        
        Args:
            metric: Métrique pour laquelle récupérer le meilleur modèle.
        
        Returns:
            Modèle XGBoost ou None si les snapshots ne sont pas activés.
        """
        if not self.save_snapshots:
            logger.warning("Snapshots are not enabled. Cannot retrieve best model.")
            return None
        
        if metric is None:
            metric = self.metrics_to_monitor[0]
        return self.best_models.get(metric)
    
    def get_learning_curve(self, metric=None):
        """
        Retourne la courbe d'apprentissage pour une métrique donnée.
        Si aucune métrique n'est spécifiée, retourne la courbe d'apprentissage pour la première métrique.
        
        Args:
            metric: Métrique pour laquelle récupérer la courbe d'apprentissage.
        
        Returns:
            list: Liste des valeurs de la métrique à chaque itération.
        """
        if metric is None:
            metric = self.metrics_to_monitor[0]
        return self.history.get(metric, [])


class XGBoostModel:
    """
    Wrapper autour de XGBoost pour simplifier l'entraînement et l'évaluation des modèles.
    Offre des fonctionnalités supplémentaires comme:
    - Gestion de données temporelles
    - Visualisation des performances
    - Feature importance et Shapley values
    - Export de modèles
    """
    def __init__(self, params=None, objective='binary:logistic', 
                 problem_type='classification', early_stopping_rounds=10,
                 random_state=42, verbose=True, callbacks=None):
        """
        Initialise le modèle XGBoost.
        
        Args:
            params: Paramètres du modèle XGBoost.
            objective: Objectif d'optimisation.
            problem_type: Type de problème ('classification' ou 'regression').
            early_stopping_rounds: Nombre de rounds sans amélioration avant arrêt.
            random_state: Graine aléatoire pour la reproductibilité.
            verbose: Si True, affiche des logs.
            callbacks: Liste des callbacks personnalisés supplémentaires.
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        
        # Paramètres par défaut
        default_params = {
            'objective': objective,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 1 if verbose else 0
        }
        
        # Fusions des paramètres par défaut et ceux fournis
        self.params = default_params.copy()
        if params:
            self.params.update(params)
        
        # Sélection des métriques d'évaluation par défaut selon le type de problème
        if self.problem_type == 'classification':
            if 'eval_metric' not in self.params:
                self.params['eval_metric'] = ['auc', 'logloss']
        elif self.problem_type == 'regression':
            if 'eval_metric' not in self.params:
                self.params['eval_metric'] = ['rmse', 'mae']
        
        # Initialisation des callbacks personnalisés
        self.custom_callback = XGBoostCustomCallback(
            early_stopping_rounds=early_stopping_rounds,
            metrics_to_monitor=self.params['eval_metric'] if isinstance(self.params['eval_metric'], list) else [self.params['eval_metric']],
            verbose=verbose
        )
        
        self.callbacks = [self.custom_callback]
        if callbacks:
            self.callbacks.extend(callbacks)
        
        # Le modèle sera initialisé lors de l'entraînement
        self.model = None
        self.feature_names = None
        self.classes_ = None  # Pour la compatibilité avec scikit-learn
        self.feature_importances_ = None
        self.best_iteration_ = None # Pour stocker la meilleure itération déterminée par le callback
    
    def fit(self, X, y, eval_set=None, sample_weight=None, 
            feature_weights=None, sample_groups=None, 
            categorical_features=None, **kwargs):
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X: Données d'entraînement.
            y: Labels d'entraînement.
            eval_set: Ensemble d'évaluation [(X_val, y_val)].
            sample_weight: Poids des échantillons.
            feature_weights: Poids des features.
            sample_groups: Groupes d'échantillons pour l'évaluation.
            categorical_features: Indices des features catégorielles.
            **kwargs: Arguments supplémentaires pour xgb.train().
        
        Returns:
            self: Modèle entraîné.
        """
        # Convertir en DMatrix
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        
        dtrain = xgb.DMatrix(data=X, label=y, weight=sample_weight, 
                            feature_names=self.feature_names, feature_weights=feature_weights,
                            group=sample_groups)
        
        # Préparer les ensembles d'évaluation
        evals = []
        if eval_set:
            for i, (X_val, y_val) in enumerate(eval_set):
                deval = xgb.DMatrix(data=X_val, label=y_val, 
                                    feature_names=self.feature_names)
                evals.append((deval, f"validation_{i}"))
        
        # Paramètres d'entraînement
        train_params = self.params.copy()
        if 'num_boost_round' not in kwargs:
            kwargs['num_boost_round'] = 1000
        
        # Entraînement du modèle
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            evals=evals,
            callbacks=self.callbacks,
            **kwargs
        )
        
        # Récupérer la meilleure itération depuis le callback pour la métrique principale
        # et la stocker pour une utilisation dans predict/predict_proba
        main_metric_key_for_callback = None
        if self.params.get('eval_metric'):
            current_eval_metrics = self.params['eval_metric']
            if isinstance(current_eval_metrics, list) and current_eval_metrics:
                main_metric_key_for_callback = current_eval_metrics[0]
            elif isinstance(current_eval_metrics, str):
                main_metric_key_for_callback = current_eval_metrics
        
        if main_metric_key_for_callback:
            self.best_iteration_ = self.custom_callback.get_best_iteration(main_metric_key_for_callback)
            if self.verbose:
                logger.info(f"Meilleure itération déterminée par callback pour '{main_metric_key_for_callback}': {self.best_iteration_}")

        # Si le booster a un attribut best_iteration (défini par XGBoost natif ou un autre callback),
        # on peut le considérer aussi, mais celui du custom_callback est prioritaire s'il est défini.
        if self.best_iteration_ is None and hasattr(self.model, 'best_iteration'):
            self.best_iteration_ = self.model.best_iteration
            if self.verbose:
                logger.info(f"Meilleure itération (fallback sur booster.best_iteration): {self.best_iteration_}")

        if self.model and self.best_iteration_ is not None and self.best_iteration_ >= 0:
             # best_iteration_ est 0-indexed. ntree_limit est le nombre d'arbres.
             self.model.best_ntree_limit = self.best_iteration_ + 1
        elif self.model:
             # Fallback si best_iteration_ n'est pas défini, utiliser toutes les itérations
             self.model.best_ntree_limit = kwargs.get('num_boost_round', 1000)
        
        # Extraction de l'importance des features
        if self.model:
            self.feature_importances_ = self.model.get_score(importance_type='gain')
        
        # Pour la compatibilité avec scikit-learn
        if self.problem_type == 'classification':
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X, ntree_limit=None):
        """
        Prédit les classes (classification) ou les valeurs (régression).
        
        Args:
            X: Données pour la prédiction.
            ntree_limit: Limite le nombre d'arbres utilisés pour la prédiction.
        
        Returns:
            np.ndarray: Prédictions.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        dtest = xgb.DMatrix(data=X, feature_names=self.feature_names)
        
        # Utiliser la meilleure limite d'arbres si ntree_limit n'est pas spécifié
        effective_ntree_limit = ntree_limit
        if effective_ntree_limit is None and hasattr(self.model, 'best_ntree_limit'):
            effective_ntree_limit = self.model.best_ntree_limit
            if self.verbose:
                logger.debug(f"Utilisation de best_ntree_limit ({effective_ntree_limit}) pour predict.")

        predictions = self.model.predict(dtest, iteration_range=(0, effective_ntree_limit) if effective_ntree_limit is not None else None)

        if self.problem_type == 'classification':
            if len(self.classes_) > 2:  # Multi-classe
                return np.argmax(predictions, axis=1) # Retourne les classes prédites
            else:  # Binaire
                return (predictions > 0.5).astype(int)
        else:  # Régression
            return predictions
    
    def predict_proba(self, X, ntree_limit=None):
        """
        Prédit les probabilités pour la classification.
        
        Args:
            X: Données pour la prédiction.
            ntree_limit: Limite le nombre d'arbres utilisés pour la prédiction.
        
        Returns:
            np.ndarray: Probabilités pour chaque classe.
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba n'est disponible que pour la classification.")
        
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        dtest = xgb.DMatrix(data=X, feature_names=self.feature_names)

        effective_ntree_limit = ntree_limit
        if effective_ntree_limit is None and hasattr(self.model, 'best_ntree_limit'):
            effective_ntree_limit = self.model.best_ntree_limit
            if self.verbose:
                logger.debug(f"Utilisation de best_ntree_limit ({effective_ntree_limit}) pour predict_proba.")

        probas = self.model.predict(dtest, iteration_range=(0, effective_ntree_limit) if effective_ntree_limit is not None else None)
        
        if len(self.classes_) > 2:  # Multi-classe
            # S'assurer que la sortie est bien (n_samples, n_classes)
            # Si l'objectif est 'multi:softprob', predict retourne déjà les probas par classe.
            return probas 
        else:  # Binaire
            # Si l'objectif est 'binary:logistic', predict retourne la proba de la classe positive.
            return np.vstack((1 - probas, probas)).T
    
    def score(self, X, y):
        """
        Calcule le score du modèle.
        Pour la classification, c'est l'accuracy.
        Pour la régression, c'est le MSE négatif.
        
        Args:
            X: Données pour l'évaluation.
            y: Labels pour l'évaluation.
        
        Returns:
            float: Score du modèle.
        """
        if self.problem_type == 'classification':
            return accuracy_score(y, self.predict(X))
        else:
            return -mean_squared_error(y, self.predict(X))
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Retourne l'importance des features.
        
        Args:
            importance_type: Type d'importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover').
        
        Returns:
            dict: Importance des features.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        return self.model.get_score(importance_type=importance_type)
    
    def plot_feature_importance(self, importance_type='gain', max_features=20, figsize=(10, 8)):
        """
        Affiche l'importance des features sous forme de graphique.
        
        Args:
            importance_type: Type d'importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover').
            max_features: Nombre maximum de features à afficher.
            figsize: Taille de la figure.
        """
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance(importance_type)
            importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            
            if max_features and len(importance) > max_features:
                importance = dict(list(importance.items())[:max_features])
            
            plt.figure(figsize=figsize)
            plt.barh(list(importance.keys()), list(importance.values()))
            plt.title(f"Feature Importance ({importance_type})")
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot plot feature importance.")
            return importance
    
    def plot_learning_curves(self, figsize=(12, 6)):
        """
        Affiche les courbes d'apprentissage du modèle.
        
        Args:
            figsize: Taille de la figure.
        """
        try:
            import matplotlib.pyplot as plt
            
            metrics = self.custom_callback.metrics_to_monitor
            fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
            
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                history = self.custom_callback.get_learning_curve(metric)
                best_iter = self.custom_callback.get_best_iteration(metric)
                best_score = self.custom_callback.get_best_score(metric)
                
                axes[i].plot(history, label=f'Training {metric}')
                axes[i].axvline(x=best_iter, color='r', linestyle='--', 
                              label=f'Best iteration: {best_iter}\nScore: {best_score:.4f}')
                axes[i].set_title(f'Learning Curve - {metric}')
                axes[i].set_xlabel('Iterations')
                axes[i].set_ylabel(metric)
                axes[i].legend()
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot plot learning curves.")
    
    def save_model(self, filename):
        """
        Sauvegarde le modèle XGBoost.
        
        Args:
            filename: Nom du fichier où sauvegarder le modèle.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        self.model.save_model(filename)
        logger.info(f"Modèle sauvegardé dans {filename}")
    
    def load_model(self, filename):
        """
        Charge un modèle XGBoost depuis un fichier.
        
        Args:
            filename: Nom du fichier contenant le modèle.
        
        Returns:
            self: Modèle chargé.
        """
        self.model = xgb.Booster()
        self.model.load_model(filename)
        logger.info(f"Modèle chargé depuis {filename}")
        return self
    
    def cross_validate(self, X, y, cv=5, stratify=None, 
                      early_stopping_rounds=None, random_state=None,
                      verbose=None):
        """
        Effectue une validation croisée.
        
        Args:
            X: Données d'entraînement.
            y: Labels d'entraînement.
            cv: Nombre de folds ou un itérable générant des indices de train/test.
            stratify: Si non None, effectue une stratification sur ces données.
            early_stopping_rounds: Nombre de rounds sans amélioration avant arrêt.
            random_state: Graine aléatoire pour la reproductibilité.
            verbose: Si True, affiche des logs.
        
        Returns:
            dict: Résultats de la validation croisée.
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if early_stopping_rounds is None:
            early_stopping_rounds = self.early_stopping_rounds
        
        if random_state is None:
            random_state = self.random_state
        
        if verbose is None:
            verbose = self.verbose
        
        # Définir le type de cross-validation
        if stratify is not None and self.problem_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_splits = cv_splitter.split(X, stratify)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_splits = cv_splitter.split(X)
        
        # Stocker les résultats
        cv_results = {
            'train_scores': [],
            'val_scores': [],
            'best_iterations': [],
            'feature_importances': []
        }
        
        # Effectuer la validation croisée
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"Fold {fold+1}/{cv}")
            
            # Séparer les données
            X_train, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_train, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            
            # Entraîner le modèle
            model_fold = XGBoostModel(
                params=self.params,
                problem_type=self.problem_type,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                verbose=verbose
            )
            
            model_fold.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            # Collecter les résultats
            train_score = model_fold.score(X_train, y_train)
            val_score = model_fold.score(X_val, y_val)
            best_iteration = model_fold.custom_callback.get_best_iteration()
            feature_importance = model_fold.get_feature_importance()
            
            cv_results['train_scores'].append(train_score)
            cv_results['val_scores'].append(val_score)
            cv_results['best_iterations'].append(best_iteration)
            cv_results['feature_importances'].append(feature_importance)
            
            logger.info(f"Fold {fold+1} - Train score: {train_score:.4f}, Val score: {val_score:.4f}")
        
        # Agréger les résultats
        cv_results['mean_train_score'] = np.mean(cv_results['train_scores'])
        cv_results['std_train_score'] = np.std(cv_results['train_scores'])
        cv_results['mean_val_score'] = np.mean(cv_results['val_scores'])
        cv_results['std_val_score'] = np.std(cv_results['val_scores'])
        cv_results['mean_best_iteration'] = np.mean(cv_results['best_iterations'])
        
        # Agréger les importances de features
        all_features = set()
        for imp in cv_results['feature_importances']:
            all_features.update(imp.keys())
        
        mean_importance = {feature: 0 for feature in all_features}
        for imp in cv_results['feature_importances']:
            for feature in all_features:
                mean_importance[feature] += imp.get(feature, 0) / cv
        
        cv_results['mean_feature_importance'] = mean_importance
        
        logger.info(f"Cross-validation results - Mean train score: {cv_results['mean_train_score']:.4f} ± {cv_results['std_train_score']:.4f}")
        logger.info(f"Cross-validation results - Mean val score: {cv_results['mean_val_score']:.4f} ± {cv_results['std_val_score']:.4f}")
        
        return cv_results
    
    def evaluate(self, X, y, metrics=None):
        """
        Évalue le modèle sur différentes métriques.
        
        Args:
            X: Données d'évaluation.
            y: Labels d'évaluation.
            metrics: Liste des métriques à calculer.
                    Pour la classification: 'accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss'
                    Pour la régression: 'mse', 'rmse', 'mae', 'r2', 'mape', 'median_ae'
        
        Returns:
            dict: Résultats d'évaluation pour chaque métrique.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                  roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, 
                                  r2_score, median_absolute_error)
        
        if metrics is None:
            if self.problem_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss']
            else:
                metrics = ['mse', 'rmse', 'mae', 'r2', 'median_ae']
        
        results = {}
        
        # Calcul des prédictions
        if self.problem_type == 'classification':
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            
            # Calcul des métriques de classification
            for metric in metrics:
                if metric == 'accuracy':
                    results[metric] = accuracy_score(y, y_pred)
                elif metric == 'precision':
                    results[metric] = precision_score(y, y_pred, average='weighted')
                elif metric == 'recall':
                    results[metric] = recall_score(y, y_pred, average='weighted')
                elif metric == 'f1':
                    results[metric] = f1_score(y, y_pred, average='weighted')
                elif metric == 'auc':
                    if len(self.classes_) == 2:  # Binaire
                        results[metric] = roc_auc_score(y, y_proba[:, 1])
                    else:  # Multi-classe
                        results[metric] = roc_auc_score(y, y_proba, multi_class='ovr')
                elif metric == 'logloss':
                    results[metric] = log_loss(y, y_proba)
        else:  # Régression
            y_pred = self.predict(X)
            
            # Calcul des métriques de régression
            for metric in metrics:
                if metric == 'mse':
                    results[metric] = mean_squared_error(y, y_pred)
                elif metric == 'rmse':
                    results[metric] = np.sqrt(mean_squared_error(y, y_pred))
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(y, y_pred)
                elif metric == 'r2':
                    results[metric] = r2_score(y, y_pred)
                elif metric == 'mape':
                    results[metric] = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-10))) * 100
                elif metric == 'median_ae':
                    results[metric] = median_absolute_error(y, y_pred)
        
        # Affichage des résultats
        if self.verbose:
            for metric, value in results.items():
                logger.info(f"{metric}: {value:.4f}")
        
        return results
    
    def tune_hyperparameters(self, X, y, param_space=None, cv=5, n_trials=50, 
                           timeout=None, stratify=None, pruner=None, direction='maximize'):
        """
        Optimise les hyperparamètres du modèle avec Optuna.
        
        Args:
            X: Données d'entraînement.
            y: Labels d'entraînement.
            param_space: Dictionnaire définissant l'espace de recherche pour les hyperparamètres.
                       Par exemple: {'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)}
            cv: Nombre de folds pour la validation croisée.
            n_trials: Nombre d'essais pour l'optimisation.
            timeout: Limite de temps en secondes pour l'optimisation.
            stratify: Si non None, effectue une stratification sur ces données.
            pruner: Pruner Optuna pour arrêter les essais inefficaces.
            direction: 'maximize' ou 'minimize' selon la métrique à optimiser.
        
        Returns:
            dict: Meilleurs hyperparamètres et résultats d'optimisation.
        """
        import optuna
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Définir l'espace de recherche par défaut si non fourni
        if param_space is None:
            if self.problem_type == 'classification':
                param_space = {
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 500),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'gamma': (0, 5),
                    'min_child_weight': (1, 10),
                    'lambda': (0.01, 10),
                    'alpha': (0.01, 10)
                }
            else:  # Régression
                param_space = {
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'n_estimators': (50, 500),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'gamma': (0, 5),
                    'min_child_weight': (1, 10),
                    'lambda': (0.01, 10),
                    'alpha': (0.01, 10)
                }
        
        # Définir le pruner si non fourni
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction=direction, pruner=pruner)
        
        # Définir la fonction objectif
        def objective(trial):
            # Générer les paramètres pour cet essai
            trial_params = self.params.copy()
            
            for param, space in param_space.items():
                if param == 'n_estimators':
                    trial_params[param] = trial.suggest_int(param, space[0], space[1])
                elif param in ['max_depth', 'min_child_weight']:
                    trial_params[param] = trial.suggest_int(param, space[0], space[1])
                elif param in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'lambda', 'alpha']:
                    trial_params[param] = trial.suggest_float(param, space[0], space[1], log=True)
            
            # Définir le type de cross-validation
            if stratify is not None and self.problem_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                cv_splits = cv_splitter.split(X, stratify)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                cv_splits = cv_splitter.split(X)
            
            # Effectuer la validation croisée
            scores = []
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
                y_train, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
                
                # Créer et entraîner le modèle
                model = XGBoostModel(
                    params=trial_params,
                    problem_type=self.problem_type,
                    early_stopping_rounds=self.early_stopping_rounds,
                    random_state=self.random_state,
                    verbose=False  # Désactiver les logs pour l'optimisation
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                
                # Calculer le score
                score = model.score(X_val, y_val)
                scores.append(score)
                
                # Signaler un score intermédiaire pour le pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return np.mean(scores)
        
        # Lancer l'optimisation
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Récupérer les meilleurs paramètres
        best_params = study.best_params
        
        # Mettre à jour les paramètres du modèle
        self.params.update(best_params)
        
        # Afficher les résultats
        if self.verbose:
            logger.info(f"Meilleurs hyperparamètres: {best_params}")
            logger.info(f"Meilleur score: {study.best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def time_series_cv(self, X, y, n_splits=5, train_window=None, 
                     test_window=None, gap=0, evaluation_window=None):
        """
        Effectue une validation croisée spécifique aux séries temporelles.
        
        Args:
            X: Données d'entraînement, avec un index de temps.
            y: Labels d'entraînement.
            n_splits: Nombre de divisions.
            train_window: Taille de la fenêtre d'entraînement en périodes.
            test_window: Taille de la fenêtre de test en périodes.
            gap: Nombre de périodes entre la fin de l'entraînement et le début du test.
            evaluation_window: Fenêtre d'évaluation pour chaque split (None = tout le test).
        
        Returns:
            dict: Résultats de la validation croisée.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X doit être un DataFrame avec un index de type DatetimeIndex.")
        
        # Définir les fenêtres automatiquement si non spécifiées
        if train_window is None:
            train_window = len(X) // (n_splits + 1)
        if test_window is None:
            test_window = len(X) // (n_splits * 2)
        
        # Trier les données par date
        dates = X.index.sort_values()
        
        # Stocker les résultats
        cv_results = {
            'train_scores': [],
            'test_scores': [],
            'train_periods': [],
            'test_periods': [],
            'feature_importances': []
        }
        
        # Effectuer la validation croisée
        for i in range(n_splits):
            # Définir les périodes d'entraînement et de test
            train_end = dates[train_window * (i + 1) - 1]
            test_start = dates[train_window * (i + 1) + gap]
            test_end = dates[min(train_window * (i + 1) + gap + test_window - 1, len(dates) - 1)]
            
            # Sélectionner les données
            X_train = X[X.index <= train_end]
            y_train = y[y.index <= train_end] if isinstance(y, pd.Series) else y[:len(X_train)]
            
            X_test = X[(X.index > test_start) & (X.index <= test_end)]
            y_test = y[(y.index > test_start) & (y.index <= test_end)] if isinstance(y, pd.Series) else y[len(X_train):len(X_train) + len(X_test)]
            
            # Vérifier que les données de test ne sont pas vides
            if len(X_test) == 0:
                logger.warning(f"Aucune donnée de test pour le split {i+1}/{n_splits}. Ignorer ce split.")
                continue
            
            # Entraîner le modèle
            model_fold = XGBoostModel(
                params=self.params,
                problem_type=self.problem_type,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            model_fold.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            
            # Collecter les résultats
            train_score = model_fold.score(X_train, y_train)
            test_score = model_fold.score(X_test, y_test)
            feature_importance = model_fold.get_feature_importance()
            
            cv_results['train_scores'].append(train_score)
            cv_results['test_scores'].append(test_score)
            cv_results['train_periods'].append((X_train.index.min(), X_train.index.max()))
            cv_results['test_periods'].append((X_test.index.min(), X_test.index.max()))
            cv_results['feature_importances'].append(feature_importance)
            
            logger.info(f"Split {i+1}/{n_splits} - Train: {X_train.index.min()} to {X_train.index.max()}")
            logger.info(f"Split {i+1}/{n_splits} - Test: {X_test.index.min()} to {X_test.index.max()}")
            logger.info(f"Split {i+1}/{n_splits} - Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        # Agréger les résultats
        cv_results['mean_train_score'] = np.mean(cv_results['train_scores'])
        cv_results['std_train_score'] = np.std(cv_results['train_scores'])
        cv_results['mean_test_score'] = np.mean(cv_results['test_scores'])
        cv_results['std_test_score'] = np.std(cv_results['test_scores'])
        
        # Agréger les importances de features
        all_features = set()
        for imp in cv_results['feature_importances']:
            all_features.update(imp.keys())
        
        mean_importance = {feature: 0 for feature in all_features}
        for imp in cv_results['feature_importances']:
            for feature in all_features:
                mean_importance[feature] += imp.get(feature, 0) / len(cv_results['feature_importances'])
        
        cv_results['mean_feature_importance'] = mean_importance
        
        logger.info(f"Time Series CV - Mean train score: {cv_results['mean_train_score']:.4f} ± {cv_results['std_train_score']:.4f}")
        logger.info(f"Time Series CV - Mean test score: {cv_results['mean_test_score']:.4f} ± {cv_results['std_test_score']:.4f}")
        
        return cv_results
    
    def shap_analysis(self, X, max_display=20):
        """
        Effectue une analyse SHAP pour l'interprétation du modèle.
        
        Args:
            X: Données pour l'analyse SHAP.
            max_display: Nombre maximum de features à afficher.
        
        Returns:
            tuple: (shap_values, explainer) pour une utilisation personnalisée.
        """
        try:
            import shap
            
            if self.model is None:
                raise ValueError("Le modèle n'a pas encore été entraîné.")
            
            # Créer l'explainer SHAP
            explainer = shap.TreeExplainer(self.model)
            
            # Calculer les valeurs SHAP
            if isinstance(X, pd.DataFrame):
                X_shap = X
            else:
                X_shap = pd.DataFrame(X, columns=self.feature_names)
            
            shap_values = explainer.shap_values(X_shap)
            
            # Afficher les résumés SHAP
            shap.summary_plot(shap_values, X_shap, max_display=max_display, show=False)
            
            # Retourner les valeurs SHAP pour une utilisation personnalisée
            return shap_values, explainer
        
        except ImportError:
            logger.warning("La librairie SHAP n'est pas installée. Impossible d'effectuer l'analyse SHAP.")
            return None, None
    
    def plot_shap_dependence(self, X, feature_idx, interaction_idx=None):
        """
        Affiche un graphique de dépendance SHAP pour une feature.
        
        Args:
            X: Données pour l'analyse SHAP.
            feature_idx: Indice ou nom de la feature à analyser.
            interaction_idx: Indice ou nom de la feature d'interaction (None = auto).
        """
        try:
            import shap
            
            shap_values, explainer = self.shap_analysis(X, max_display=0)
            
            if shap_values is None:
                return
            
            # Convertir les indices en noms de features si nécessaire
            if isinstance(feature_idx, int) and self.feature_names:
                feature_name = self.feature_names[feature_idx]
            else:
                feature_name = feature_idx
            
            if interaction_idx is not None:
                if isinstance(interaction_idx, int) and self.feature_names:
                    interaction_name = self.feature_names[interaction_idx]
                else:
                    interaction_name = interaction_idx
            else:
                interaction_name = None
            
            # Créer le graphique de dépendance
            if isinstance(X, pd.DataFrame):
                X_shap = X
            else:
                X_shap = pd.DataFrame(X, columns=self.feature_names)
            
            shap.dependence_plot(
                feature_name, 
                shap_values, 
                X_shap, 
                interaction_index=interaction_name
            )
        
        except ImportError:
            logger.warning("La librairie SHAP n'est pas installée. Impossible d'effectuer l'analyse SHAP.")
    
    def select_features(self, X, y, method='shap', threshold=0.01, k=None):
        """
        Sélectionne les features les plus importantes pour le modèle.
        
        Args:
            X: Données d'entraînement.
            y: Labels d'entraînement.
            method: Méthode de sélection ('shap', 'gain', 'weight', 'cover').
            threshold: Seuil d'importance (proportion du total) pour conserver une feature.
            k: Nombre de features à conserver (prioritaire sur threshold).
        
        Returns:
            list: Liste des features sélectionnées.
        """
        if self.model is None:
            logger.info("Le modèle n'est pas encore entraîné. Entraînement avec les données fournies.")
            self.fit(X, y)
        
        selected_features = []
        
        if method == 'shap':
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                
                if isinstance(X, pd.DataFrame):
                    X_shap = X
                else:
                    X_shap = pd.DataFrame(X, columns=self.feature_names)
                
                shap_values = explainer.shap_values(X_shap)
                
                # Pour la classification multi-classe, shap_values est une liste
                if isinstance(shap_values, list):
                    # Moyenne des valeurs absolues pour toutes les classes
                    importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    importance = np.abs(shap_values).mean(axis=0)
                
                # Normaliser les importances
                importance = importance / importance.sum()
                
                # Créer un dictionnaire d'importance
                feature_importance = {self.feature_names[i]: importance[i] for i in range(len(self.feature_names))}
                
            except ImportError:
                logger.warning("La librairie SHAP n'est pas installée. Utilisation de la méthode 'gain' à la place.")
                feature_importance = self.get_feature_importance('gain')
        else:
            feature_importance = self.get_feature_importance(method)
        
        # Normaliser les valeurs d'importance
        total_importance = sum(feature_importance.values())
        normalized_importance = {feature: importance / total_importance 
                              for feature, importance in feature_importance.items()}
        
        # Trier les features par importance
        sorted_features = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Sélectionner les features selon le critère
        if k is not None:
            selected_features = [feature for feature, _ in sorted_features[:k]]
        else:
            selected_features = [feature for feature, importance in sorted_features 
                               if importance >= threshold]
        
        if self.verbose:
            logger.info(f"Sélection de features: {len(selected_features)}/{len(feature_importance)} features sélectionnées.")
            logger.info(f"Features sélectionnées: {selected_features}")
        
        return selected_features
    
    def plot_permutation_importance(self, X, y, n_repeats=10, random_state=None, max_features=20, figsize=(10, 8)):
        """
        Calcule et affiche l'importance des features par permutation.
        
        Args:
            X: Données d'évaluation.
            y: Labels d'évaluation.
            n_repeats: Nombre de répétitions pour la permutation.
            random_state: Graine aléatoire pour la reproductibilité.
            max_features: Nombre maximum de features à afficher.
            figsize: Taille de la figure.
        
        Returns:
            pd.DataFrame: Résultats de l'importance par permutation.
        """
        try:
            from sklearn.inspection import permutation_importance
            import matplotlib.pyplot as plt
            
            if self.model is None:
                raise ValueError("Le modèle n'a pas encore été entraîné.")
            
            if random_state is None:
                random_state = self.random_state
            
            # Calculer l'importance par permutation
            result = permutation_importance(
                self, X, y, 
                n_repeats=n_repeats, 
                random_state=random_state
            )
            
            # Créer un DataFrame des résultats
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns
            else:
                feature_names = self.feature_names
            
            perm_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            })
            
            # Trier par importance
            perm_importance_df = perm_importance_df.sort_values('importance_mean', ascending=False)
            
            # Limiter le nombre de features affichées
            if max_features and len(perm_importance_df) > max_features:
                plot_df = perm_importance_df.head(max_features)
            else:
                plot_df = perm_importance_df
            
            # Créer le graphique
            plt.figure(figsize=figsize)
            plt.barh(plot_df['feature'], plot_df['importance_mean'])
            plt.errorbar(plot_df['importance_mean'], plot_df['feature'], 
                       xerr=plot_df['importance_std'], fmt='o', color='black')
            plt.title('Permutation Feature Importance')
            plt.xlabel('Mean Importance')
            plt.tight_layout()
            plt.show()
            
            return perm_importance_df
        
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot plot permutation importance.")
            return None

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données fictives
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1) > 1
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créer et entraîner le modèle
    model = XGBoostModel(
        params={
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': ['auc', 'logloss']
        },
        early_stopping_rounds=10,
        verbose=True
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Afficher l'importance des features
    print("\nFeature Importance:")
    for feature, importance in model.get_feature_importance().items():
        print(f"{feature}: {importance}")
    
    # Validation croisée
    cv_results = model.cross_validate(X, y, cv=3)
    print(f"\nCross-validation - Mean val score: {cv_results['mean_val_score']:.4f} ± {cv_results['std_val_score']:.4f}")
