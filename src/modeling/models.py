import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Tuple, List, Any, Dict, Optional, Union # Ajout de Union
import xgboost as xgb
import optuna
import hashlib # Pour data_hash
import subprocess # Pour git_revision_hash
import sklearn # Pour la version
import xgboost # Pour la version
# pandas et numpy sont déjà importés
# optuna est déjà importé
# Imports pour les nouveaux modèles
# Pour les réseaux de neurones (LSTM, CNN) - choisir l'un ou l'autre ou les deux
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Attention
# from tensorflow.keras.layers import Reshape, Add, Activation, BatchNormalization # Pour Squeeze-and-Excitation, connexions résiduelles

# import torch
# import torch.nn as nn
# import torch.optim as optim

# Pour les Processus Gaussiens
# import GPy
# import gpflow

# Pour les Modèles Bayésiens Hiérarchiques
# import pymc3 as pm
# import arviz as az
# import stan

# --- Définitions des Nouveaux Modèles ---

# Section 5.3.3: Réseaux de Neurones Spécialisés

class BidirectionalLSTMModel:
    """
    Modèle LSTM Bidirectionnel.
    Inspiré de la section 5.3.3 du document de référence.
    """
    def __init__(self, input_shape, num_layers=1, units=50, output_units=1, activation='sigmoid', dropout_rate=0.2, variational_dropout=False, l1_reg=0.0, l2_reg=0.0, use_temporal_attention=False, use_multi_resolution=False, temporal_regularization_factor=None, **kwargs):
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.units = units
        self.output_units = output_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout # Placeholder
        self.l1_reg = l1_reg # Placeholder pour régularisation L1 sur les poids
        self.l2_reg = l2_reg # Placeholder pour régularisation L2 sur les poids
        self.use_temporal_attention = use_temporal_attention # Placeholder
        self.use_multi_resolution = use_multi_resolution # Placeholder
        self.temporal_regularization_factor = temporal_regularization_factor # Placeholder pour régularisation temporelle
        self.model = self._build_model()
        # Potentielle dépendance: tensorflow.keras

    def _build_model(self):
        # Placeholder pour la construction du modèle avec Keras/TensorFlow ou PyTorch
        # Exemple avec Keras (à adapter)
        # from tensorflow.keras import regularizers
        # kernel_regularizer = None
        # if self.l1_reg > 0 and self.l2_reg > 0:
        #     kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        # elif self.l1_reg > 0:
        #     kernel_regularizer = regularizers.l1(self.l1_reg)
        # elif self.l2_reg > 0:
        #     kernel_regularizer = regularizers.l2(self.l2_reg)
        #
        # model = Sequential()
        # for i in range(self.num_layers):
        #     return_sequences = True if i < self.num_layers - 1 else False
        #     lstm_layer = LSTM(self.units, return_sequences=return_sequences, kernel_regularizer=kernel_regularizer)
        #     if i == 0:
        #         model.add(Bidirectional(lstm_layer, input_shape=self.input_shape))
        #     else:
        #         model.add(Bidirectional(lstm_layer))
        #     if self.dropout_rate > 0:
        #         # Placeholder pour dropout variationnel si self.variational_dropout est True
        #         model.add(Dropout(self.dropout_rate))
        #
        # if self.use_temporal_attention:
        #     # Placeholder pour couche d'attention temporelle
        #     pass # model.add(Attention(...))
        #
        # model.add(Dense(self.output_units, activation=self.activation))
        # model.compile(optimizer='adam', loss='binary_crossentropy' if self.activation == 'sigmoid' else 'mse')
        # if self.temporal_regularization_factor:
        #     # Ajouter une logique de perte personnalisée pour la régularisation temporelle
        #     print(f"    Régularisation temporelle (facteur {self.temporal_regularization_factor}) à implémenter via perte personnalisée.")
        print(f"Placeholder: Modèle LSTM Bidirectionnel construit avec {self.num_layers} couches, {self.units} unités.")
        print("  Fonctionnalités avancées (placeholders):")
        print(f"    Dropout: {self.dropout_rate}, Dropout variationnel: {self.variational_dropout}")
        print(f"    Régularisation L1: {self.l1_reg}, L2: {self.l2_reg}")
        print(f"    Attention temporelle: {self.use_temporal_attention}")
        print(f"    Architecture multi-résolution: {self.use_multi_resolution}")
        print(f"    Régularisation temporelle: {self.temporal_regularization_factor}")
        return "keras_model_placeholder" # Remplacer par le vrai modèle

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, early_stopping_patience=None, **kwargs):
        # Placeholder pour l'entraînement
        # callbacks = []
        # if validation_data and early_stopping_patience:
        #     from tensorflow.keras.callbacks import EarlyStopping
        #     early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        #     callbacks.append(early_stop_callback)
        #     print(f"  Early stopping activé avec patience={early_stopping_patience} sur val_loss.")
        #
        # self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, **kwargs)
        print(f"Placeholder: Entraînement du LSTM Bidirectionnel pour {epochs} époques.")
        if early_stopping_patience:
            print(f"  Early stopping (placeholder) avec patience {early_stopping_patience}.")
        pass

    def predict(self, X, **kwargs):
        # Placeholder pour la prédiction
        # return self.model.predict(X, **kwargs)
        print("Placeholder: Prédiction avec LSTM Bidirectionnel.")
        return np.random.rand(len(X), self.output_units) # Retourne des prédictions aléatoires

    def predict_proba(self, X, **kwargs):
        # Placeholder pour la prédiction de probabilités (si applicable)
        # return self.model.predict(X, **kwargs) # Keras predict donne des probas pour sortie sigmoïde/softmax
        print("Placeholder: Prédiction de probabilités avec LSTM Bidirectionnel.")
        # Simule une sortie de probabilité pour la classe positive
        raw_preds = np.random.rand(len(X), self.output_units)
        if self.output_units == 1 and self.activation == 'sigmoid': # Cas binaire
             return np.hstack((1-raw_preds, raw_preds)) # proba classe 0, proba classe 1
        return raw_preds # Cas multi-classe ou régression (à ajuster)

    def analyze_sensitivity(self, X_perturbed_sequence: np.ndarray, original_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Placeholder pour analyser la sensibilité du modèle à des perturbations des inputs.
        Cette méthode devrait évaluer comment les prédictions changent lorsque les données d'entrée sont perturbées.

        Args:
            X_perturbed_sequence (np.ndarray): Une séquence de données d'entrée perturbées.
                                              Doit avoir la même forme que les données d'entraînement/prédiction.
            original_predictions (np.ndarray): Les prédictions du modèle sur les données originales non perturbées.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant des métriques de sensibilité.
                            Par exemple: 'prediction_change_mean', 'prediction_change_std'.
        """
        print(f"Placeholder: Analyse de sensibilité pour BidirectionalLSTMModel.")
        # perturbed_predictions = self.predict(X_perturbed_sequence)
        # diff = perturbed_predictions - original_predictions
        # sensitivity_metrics = {
        #     'prediction_change_mean': np.mean(np.abs(diff)),
        #     'prediction_change_std': np.std(np.abs(diff)),
        #     'max_prediction_change': np.max(np.abs(diff))
        # }
        # return sensitivity_metrics
        return {"status": "sensitivity analysis placeholder for BidirectionalLSTMModel"}


class TemporalCNNModel:
    """
    Modèle CNN Temporel (par exemple, TCN).
    Inspiré de la section 5.3.3 du document de référence.
    """
    def __init__(self, input_shape, num_filters=64, kernel_size=3, num_conv_layers=2, output_units=1, activation='sigmoid', dropout_rate=0.2, l1_reg=0.0, l2_reg=0.0, use_residual_connections=False, use_squeeze_excitation=False, use_hourglass_architecture=False, temporal_regularization_factor=None, **kwargs):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.output_units = output_units
        self.activation = activation
        self.dropout_rate = dropout_rate # Ajout du paramètre dropout
        self.l1_reg = l1_reg # Placeholder pour régularisation L1 sur les poids Conv/Dense
        self.l2_reg = l2_reg # Placeholder pour régularisation L2 sur les poids Conv/Dense
        self.use_residual_connections = use_residual_connections # Placeholder
        self.use_squeeze_excitation = use_squeeze_excitation # Placeholder
        self.use_hourglass_architecture = use_hourglass_architecture # Placeholder
        self.temporal_regularization_factor = temporal_regularization_factor # Placeholder pour régularisation temporelle
        self.model = self._build_model()
        # Potentielle dépendance: tensorflow.keras ou torch

    def _build_model(self):
        # Placeholder pour la construction du modèle
        # Exemple avec Keras (à adapter)
        # from tensorflow.keras import regularizers
        # from tensorflow.keras.layers import Dropout # Ajout import Dropout
        # kernel_regularizer = None
        # if self.l1_reg > 0 and self.l2_reg > 0:
        #     kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        # elif self.l1_reg > 0:
        #     kernel_regularizer = regularizers.l1(self.l1_reg)
        # elif self.l2_reg > 0:
        #     kernel_regularizer = regularizers.l2(self.l2_reg)
        #
        # inputs = tf.keras.Input(shape=self.input_shape)
        # x = inputs
        # for _ in range(self.num_conv_layers):
        #     prev_x = x
        #     x = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, padding='causal', activation='relu', kernel_regularizer=kernel_regularizer)(x)
        #     # x = BatchNormalization()(x) # Optionnel
        #     if self.dropout_rate > 0: # Ajout Dropout après Conv1D
        #         x = Dropout(self.dropout_rate)(x)
        #     if self.use_residual_connections and prev_x.shape == x.shape:
        #         x = Add()([prev_x, x])
        #     if self.use_squeeze_excitation:
        #         # Placeholder pour Squeeze-and-Excitation block
        #         pass
        #
        # if self.use_hourglass_architecture:
        #     # Placeholder pour architecture encodeur-décodeur
        #     pass
        #
        # x = GlobalAveragePooling1D()(x) # Ou Flatten()
        # outputs = Dense(self.output_units, activation=self.activation, kernel_regularizer=kernel_regularizer)(x) # Ajout regularizer à Dense
        # model = tf.keras.Model(inputs, outputs)
        # model.compile(optimizer='adam', loss='binary_crossentropy' if self.activation == 'sigmoid' else 'mse')
        # if self.temporal_regularization_factor:
        #     # Ajouter une logique de perte personnalisée pour la régularisation temporelle
        #     print(f"    Régularisation temporelle (facteur {self.temporal_regularization_factor}) à implémenter via perte personnalisée.")
        print(f"Placeholder: Modèle CNN Temporel construit avec {self.num_conv_layers} couches Conv1D.")
        print("  Fonctionnalités avancées (placeholders):")
        print(f"    Dropout: {self.dropout_rate}")
        print(f"    Régularisation L1: {self.l1_reg}, L2: {self.l2_reg}")
        print(f"    Connexions résiduelles: {self.use_residual_connections}")
        print(f"    Squeeze-and-Excitation: {self.use_squeeze_excitation}")
        print(f"    Architecture en sablier: {self.use_hourglass_architecture}")
        print(f"    Régularisation temporelle: {self.temporal_regularization_factor}")
        return "keras_tcn_model_placeholder" # Remplacer par le vrai modèle

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, early_stopping_patience=None, **kwargs):
        # Placeholder pour l'entraînement
        # callbacks = []
        # if validation_data and early_stopping_patience:
        #     from tensorflow.keras.callbacks import EarlyStopping
        #     early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        #     callbacks.append(early_stop_callback)
        #     print(f"  Early stopping activé avec patience={early_stopping_patience} sur val_loss.")
        #
        # self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, **kwargs)
        print(f"Placeholder: Entraînement du CNN Temporel pour {epochs} époques.")
        if early_stopping_patience:
            print(f"  Early stopping (placeholder) avec patience {early_stopping_patience}.")
        pass

    def predict(self, X, **kwargs):
        # Placeholder pour la prédiction
        # return self.model.predict(X, **kwargs)
        print("Placeholder: Prédiction avec CNN Temporel.")
        return np.random.rand(len(X), self.output_units)

    def predict_proba(self, X, **kwargs):
        # Placeholder pour la prédiction de probabilités
        print("Placeholder: Prédiction de probabilités avec CNN Temporel.")
        raw_preds = np.random.rand(len(X), self.output_units)
        if self.output_units == 1 and self.activation == 'sigmoid':
             return np.hstack((1-raw_preds, raw_preds))
        return raw_preds # Assurer que cette ligne est correctement indentée avec le if/else précédent

    def analyze_sensitivity(self, X_perturbed_sequence: np.ndarray, original_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Placeholder pour analyser la sensibilité du modèle CNN Temporel.
        Args:
            X_perturbed_sequence (np.ndarray): Données d'entrée perturbées.
            original_predictions (np.ndarray): Prédictions sur données originales.
        Returns:
            Dict[str, Any]: Métriques de sensibilité.
        """
        print(f"Placeholder: Analyse de sensibilité pour TemporalCNNModel.")
        # perturbed_predictions = self.predict(X_perturbed_sequence)
        # ... (calcul similaire à LSTM)
        return {"status": "sensitivity analysis placeholder for TemporalCNNModel"}


# Section 5.3.4: Modèles Bayésiens

class GaussianProcessRegressionModel:
    """
    Modèle de Régression par Processus Gaussien.
    Inspiré de la section 5.3.4 du document de référence.
    """
    def __init__(self, kernel_spec=None, inducing_points_ratio=None, random_state=None, **kwargs):
        self.kernel_spec = kernel_spec # Ex: {'type': 'RBF', 'params': {'input_dim': 1, 'variance': 1., 'lengthscale': 1.}}
                                      # Ou une liste pour noyaux composites
        self.inducing_points_ratio = inducing_points_ratio # Pour approximations (ex: 0.1 pour 10% de points)
        self.random_state = random_state # Pour la reproductibilité si nécessaire
        self.model = self._build_model(**kwargs)
        # Potentielle dépendance: GPy ou GPflow

    def _build_model(self, X_train_shape=None, **kwargs):
        # Placeholder pour la construction du modèle GP
        # Exemple avec GPy (nécessite X_train_shape pour initialiser le noyau si input_dim n'est pas dans kernel_spec)
        # if self.kernel_spec is None:
        #     input_dim = X_train_shape[1] if X_train_shape else 1
        #     kernel = GPy.kern.RBF(input_dim=input_dim) + GPy.kern.White(input_dim=input_dim)
        # else:
        #     # Logique pour construire un noyau composite à partir de kernel_spec
        #     kernel = GPy.kern.RBF(input_dim=X_train_shape[1] if X_train_shape else 1) # Placeholder
        #
        # if self.inducing_points_ratio and X_train_shape:
        #     num_inducing = int(X_train_shape[0] * self.inducing_points_ratio)
        #     # Z = GPy.util.linalg.tdot(np.random.permutation(X_train_shape[0])[:num_inducing]) # Pas sûr de ça
        #     # model = GPy.models.SparseGPRegression(X_placeholder, Y_placeholder, kernel, num_inducing_points=num_inducing)
        #     print(f"Placeholder: Modèle GP Sparsifié avec {num_inducing} points induisants.")
        # else:
        #     # model = GPy.models.GPRegression(X_placeholder, Y_placeholder, kernel)
        #     print("Placeholder: Modèle GP standard.")
        print(f"Placeholder: Modèle de Processus Gaussien construit.")
        print(f"  Spécification du noyau: {self.kernel_spec}")
        print(f"  Ratio de points induisants: {self.inducing_points_ratio}")
        return "gpy_model_placeholder" # Remplacer par le vrai modèle

    def fit(self, X, y, optimize_restarts=5, **kwargs):
        # Placeholder pour l'entraînement (optimisation des hyperparamètres du noyau)
        # self.model.X = X
        # self.model.Y = y.reshape(-1,1) if len(y.shape) == 1 else y
        # self.model.optimize_restarts(num_restarts=optimize_restarts, verbose=False)
        print(f"Placeholder: Entraînement du Processus Gaussien (optimisation des hyperparamètres du noyau avec {optimize_restarts} redémarrages).")
        pass

    def predict(self, X, **kwargs):
        # Placeholder pour la prédiction (moyenne et variance)
        # mean, variance = self.model.predict(X, **kwargs)
        # return mean # Ou (mean, variance) selon le besoin
        print("Placeholder: Prédiction avec Processus Gaussien (moyenne).")
        return np.random.rand(len(X), 1) # Simule la prédiction de la moyenne

    def predict_proba(self, X, **kwargs):
        # Pour les GP en régression, predict_proba retourne la moyenne et la variance.
        # mean, variance = self.model.predict(X, **kwargs) # La vraie implémentation GPy/GPflow
        print("Placeholder: Prédiction avec Processus Gaussien (moyenne et variance simulées).")
        mean = np.random.rand(len(X), 1) # Simule la prédiction de la moyenne
        variance = np.random.rand(len(X), 1) * 0.1 # Simule la variance, doit être positif
        # S'assurer que la variance est positive
        variance = np.abs(variance)
        return mean, variance # Retourne un tuple (moyenne, variance)

    def analyze_sensitivity(self, X_perturbed: np.ndarray, original_predictions_mean: np.ndarray) -> Dict[str, Any]:
        """
        Placeholder pour analyser la sensibilité du modèle GP.
        Pourrait évaluer comment la moyenne et la variance prédites changent.
        Args:
            X_perturbed (np.ndarray): Données d'entrée perturbées.
            original_predictions_mean (np.ndarray): Prédictions de moyenne sur données originales.
        Returns:
            Dict[str, Any]: Métriques de sensibilité.
        """
        print(f"Placeholder: Analyse de sensibilité pour GaussianProcessRegressionModel.")
        # perturbed_mean, perturbed_variance = self.predict_proba(X_perturbed)
        # ... (calculs de changement de moyenne, changement de variance)
        return {"status": "sensitivity analysis placeholder for GaussianProcessRegressionModel"}


class HierarchicalBayesianModelPlaceholder:
    """
    Placeholder pour un Modèle Bayésien Hiérarchique.
    Inspiré de la section 5.3.4 du document de référence.
    Dépendance potentielle: PyMC, Stan.
    """
    def __init__(self, model_specification=None, informative_priors=None, adaptive_mcmc=True, **kwargs):
        self.model_specification = model_specification # Pour définir la structure du modèle
        self.informative_priors = informative_priors # Pour spécifier les priors
        self.adaptive_mcmc = adaptive_mcmc # Option pour MCMC adaptatif
        self.trace = None # Pour stocker les résultats de l'échantillonnage
        # Potentielle dépendance: pymc ou stan
        print("Placeholder: Modèle Bayésien Hiérarchique initialisé.")
        print(f"  Spécification du modèle: {self.model_specification}")
        print(f"  Priors informatifs: {self.informative_priors}")
        print(f"  MCMC adaptatif: {self.adaptive_mcmc}")

    def fit(self, X, y, draws=2000, tune=1000, chains=2, **kwargs):
        # Placeholder pour l'échantillonnage MCMC
        # with pm.Model() as hierarchical_model:
        #     # Définir les priors (potentiellement informatifs)
        #     # Définir la vraisemblance
        #     # ...
        #     if self.adaptive_mcmc:
        #         # Utiliser des méthodes d'échantillonnage adaptatif (ex: NUTS pour PyMC)
        #         step = pm.NUTS()
        #         self.trace = pm.sample(draws, tune=tune, chains=chains, step=step, return_inferencedata=True, **kwargs)
        #     else:
        #         self.trace = pm.sample(draws, tune=tune, chains=chains, return_inferencedata=True, **kwargs)
        print(f"Placeholder: Entraînement (échantillonnage MCMC) du Modèle Bayésien Hiérarchique.")
        print(f"  Tirages: {draws}, Burn-in: {tune}, Chaînes: {chains}")
        self.trace = "mcmc_trace_placeholder" # Remplacer par les vrais résultats
        pass

    def predict(self, X, **kwargs):
        # Placeholder pour la prédiction (souvent basée sur la distribution a posteriori)
        # ppc = pm.sample_posterior_predictive(self.trace, samples=500, model=self.hierarchical_model, var_names=['prediction_target'])
        # return np.mean(ppc['prediction_target'], axis=0)
        print("Placeholder: Prédiction avec Modèle Bayésien Hiérarchique.")
        return np.random.rand(len(X), 1) # Simule des prédictions

    def predict_proba(self, X, **kwargs) -> np.ndarray: # Type de retour clarifié
        # Pour les modèles bayésiens, on a toute la distribution a posteriori.
        # Retourne des échantillons de la distribution prédictive a posteriori.
        # La forme sera (nombre d'échantillons de X, nombre d'échantillons MCMC par prédiction)
        # Exemple: si X a 10 lignes, et on tire 500 échantillons MCMC pour chaque, la sortie est (10, 500)
        print("Placeholder: Prédiction (échantillons de la distribution a posteriori) avec Modèle Bayésien Hiérarchique.")
        # ppc = pm.sample_posterior_predictive(self.trace, samples=num_mcmc_samples, model=self.hierarchical_model, var_names=['prediction_target'])
        # return ppc['prediction_target'].T # Transposer pour avoir (len(X), num_mcmc_samples)
        num_mcmc_samples = kwargs.get('num_mcmc_samples', 100) # Permet de spécifier le nombre d'échantillons
        return np.random.rand(len(X), num_mcmc_samples) # ex: 100 échantillons MCMC par point de données X

    def analyze_sensitivity(self, X_perturbed: np.ndarray, original_posterior_samples: np.ndarray) -> Dict[str, Any]:
        """
        Placeholder pour analyser la sensibilité du modèle Bayésien Hiérarchique.
        Pourrait comparer les distributions a posteriori.
        Args:
            X_perturbed (np.ndarray): Données d'entrée perturbées.
            original_posterior_samples (np.ndarray): Échantillons a posteriori pour les données originales.
        Returns:
            Dict[str, Any]: Métriques de sensibilité (ex: distance de Wasserstein entre distributions).
        """
        print(f"Placeholder: Analyse de sensibilité pour HierarchicalBayesianModelPlaceholder.")
        # perturbed_posterior_samples = self.predict_proba(X_perturbed)
        # ... (calculs de comparaison de distributions)
        return {"status": "sensitivity analysis placeholder for HierarchicalBayesianModelPlaceholder"}

# Fonctions utilitaires pour la traçabilité (placeholders)
def get_git_revision_hash() -> Optional[str]:
    """Tente de récupérer le hash du commit git actuel."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return None

def calculate_data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Calcule un hash simple pour un jeu de données X, y."""
    # Convertir en string et hasher. Pour de grands datasets, considérer un sous-ensemble ou des statistiques.
    # S'assurer que l'ordre des colonnes et des lignes est constant pour la reproductibilité du hash.
    X_sorted_cols = X.reindex(sorted(X.columns), axis=1)
    combined_str = X_sorted_cols.to_string() + y.to_string()
    return hashlib.md5(combined_str.encode()).hexdigest()


def prepare_data_for_model(df: pd.DataFrame, target_shift_days: int = 1, feature_columns: List[str] = None, target_column: str = None, price_change_threshold: float = 0.02, problem_type: str = 'classification') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les données pour l'entraînement du modèle.
    """
    df_copy = df.copy()

    if target_column is None:
        if 'close' not in df_copy.columns:
            raise ValueError("La colonne 'close' est requise dans le DataFrame lorsque target_column n'est pas spécifié.")
        df_copy['future_close'] = df_copy['close'].shift(-target_shift_days)
        df_copy['price_change'] = (df_copy['future_close'] - df_copy['close']) / df_copy['close']
        
        if problem_type == 'classification':
            df_copy['target'] = (df_copy['price_change'] > price_change_threshold).astype(int)
            target_column = 'target'
            target_dist = df_copy['target'].value_counts(dropna=False)
            print(f"Distribution de la target (classification, seuil: {price_change_threshold*100:.1f}%):")
            print(f"  Classe 0: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_copy)*100:.1f}%)")
            print(f"  Classe 1: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_copy)*100:.1f}%)")
        elif problem_type == 'regression':
            df_copy['target'] = df_copy['price_change']
            target_column = 'target'
            print(f"Target pour la régression: 'price_change' (min: {df_copy['target'].min():.4f}, max: {df_copy['target'].max():.4f})")
        else:
            raise ValueError(f"problem_type non supporté: {problem_type}. Choisissez 'classification' ou 'regression'.")
            
    elif target_column not in df_copy.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")

    if feature_columns is None:
        default_features = ['sma_10', 'sma_20', 'ema_10', 'ema_20', 'rsi_14']
        lowercase_cols = {col.lower(): col for col in df_copy.columns}
        feature_columns = [lowercase_cols.get(f.lower()) for f in default_features if f.lower() in lowercase_cols]
        if not feature_columns:
            raise ValueError("Aucune des features par défaut n'a été trouvée. Spécifiez feature_columns.")

    missing_features = [col for col in feature_columns if col not in df_copy.columns]
    if missing_features:
        raise ValueError(f"Colonnes de features manquantes : {missing_features}")

    df_processed = df_copy[feature_columns + [target_column]].copy()
    df_processed.dropna(inplace=True)

    X = df_processed[feature_columns]
    y = df_processed[target_column]

    return X, y

def _objective_optuna(trial: optuna.Trial, X_opt: pd.DataFrame, y_opt: pd.Series, model_type: str, cv_splitter: TimeSeriesSplit, scale_features_flag: bool, global_random_state: int, precomputed_class_weight_dict: Dict) -> float:
    """
    Fonction objectif à optimiser par Optuna.
    """
    scores = []
    
    if model_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 30, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.035),
            'class_weight': precomputed_class_weight_dict,
            'random_state': global_random_state,
            'n_jobs': -1
        }
        model_builder = RandomForestClassifier
        current_model_scale_features = False
    elif model_type == 'xgboost_classifier':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.35, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), # Ajout de min_child_weight
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.6),
            'lambda': trial.suggest_float('lambda', 1e-9, 1.0, log=True), # L2 reg
            'alpha': trial.suggest_float('alpha', 1e-9, 1.0, log=True), # L1 reg
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': global_random_state
        }
        if precomputed_class_weight_dict and len(precomputed_class_weight_dict) == 2:
            counts = y_opt.value_counts()
            if 0 in counts and 1 in counts and counts[1] > 0:
                params['scale_pos_weight'] = counts[0] / counts[1]
        model_builder = xgb.XGBClassifier
        current_model_scale_features = False
    else:
        trial.report(0.0, 0)
        print(f"Optimisation Optuna non configurée pour {model_type}. Retourne score 0.")
        return 0.0

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_opt, y_opt)):
        X_train_cv, X_val_cv = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
        y_train_cv, y_val_cv = y_opt.iloc[train_idx], y_opt.iloc[val_idx]

        scaler_cv = None
        X_train_cv_processed = X_train_cv
        X_val_cv_processed = X_val_cv

        if current_model_scale_features:
            scaler_cv = StandardScaler()
            X_train_cv_processed = pd.DataFrame(scaler_cv.fit_transform(X_train_cv), columns=X_train_cv.columns, index=X_train_cv.index)
            X_val_cv_processed = pd.DataFrame(scaler_cv.transform(X_val_cv), columns=X_val_cv.columns, index=X_val_cv.index)
        
        model_cv = model_builder(**params)
        
        fit_params = {}
        if model_type == 'xgboost_classifier':
            fit_params['eval_set'] = [(X_val_cv_processed, y_val_cv)]
            fit_params['early_stopping_rounds'] = trial.suggest_int('early_stopping_rounds', 10, 50)
            fit_params['verbose'] = False
        
        model_cv.fit(X_train_cv_processed, y_train_cv, **fit_params)
        
        if hasattr(model_cv, 'predict_proba'):
            probabilities_cv = model_cv.predict_proba(X_val_cv_processed)
            if probabilities_cv.shape[1] == 2:
                 probabilities_cv = probabilities_cv[:, 1]
            else:
                 score = 0.0
                 scores.append(score)
                 continue
        else: 
            predictions_cv = model_cv.predict(X_val_cv_processed)
            score = -mean_squared_error(y_val_cv, predictions_cv)
            scores.append(score)
            continue

        try:
            if y_val_cv.nunique() > 1:
                score = roc_auc_score(y_val_cv, probabilities_cv)
            else:
                score = 0.0 
        except ValueError:
            score = 0.0
        scores.append(score)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores) if scores else 0.0


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic_regression', model_params: Dict[str, Any] = None, model_path: str = 'models_store/model.joblib', test_size: float = 0.2, random_state: int = 42, scale_features: bool = True) -> Dict[str, Any]: # Changement du type de retour
    if model_params is None:
        model_params = {}
    
    # Placeholder pour la détection d'anomalies dans les features d'entrée
    # TODO: Implémenter des vérifications pour les anomalies (ex: valeurs extrêmes, distribution inattendue)
    #       avant l'entraînement. Pourrait retourner un rapport ou lever des avertissements.
    #       Par exemple, vérifier si X.describe() correspond aux attentes.

    print(f"Début de l'entraînement du modèle {model_type} avec {len(X)} échantillons et {len(X.columns)} features.")

    calculated_class_weight_dict = None 
    if y.dtype == 'int' and y.nunique() > 1 : 
        unique_classes = np.unique(y)
        class_weights_values = compute_class_weight('balanced', classes=unique_classes, y=y)
        calculated_class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_values)}
        print(f"Poids des classes (calculés sur y complet avant split): {calculated_class_weight_dict}")

    optimize_hyperparams = model_params.pop('optimize_hyperparameters', False)
    optuna_n_trials = model_params.pop('optuna_n_trials', 10) 
    optuna_direction = model_params.pop('optuna_direction', 'maximize')
    optuna_cv_splits = model_params.pop('optuna_cv_splits', 3)
    
    best_params_from_optuna = {} 

    if optimize_hyperparams:
        print(f"Optimisation des hyperparamètres pour {model_type} avec Optuna ({optuna_n_trials} essais)...")
        X_train_val, X_test_final_split, y_train_val, y_test_final_split = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
        
        optuna_class_weight_dict_for_objective = None
        if y_train_val.dtype == 'int' and y_train_val.nunique() > 1:
            unique_classes_opt = np.unique(y_train_val)
            class_weights_val_opt = compute_class_weight('balanced', classes=unique_classes_opt, y=y_train_val)
            optuna_class_weight_dict_for_objective = {cls: weight for cls, weight in zip(unique_classes_opt, class_weights_val_opt)}
        elif calculated_class_weight_dict: 
             optuna_class_weight_dict_for_objective = calculated_class_weight_dict

        tscv = TimeSeriesSplit(n_splits=optuna_cv_splits)
        study = optuna.create_study(direction=optuna_direction)
        
        study.optimize(lambda trial: _objective_optuna(trial, X_train_val, y_train_val, model_type, tscv, scale_features, random_state, optuna_class_weight_dict_for_objective), n_trials=optuna_n_trials)
        
        best_params_from_optuna = study.best_params
        print(f"Meilleurs hyperparamètres trouvés par Optuna: {best_params_from_optuna}")
    else:
        pass 

    if optimize_hyperparams:
        X_train_final = X_train_val
        y_train_final = y_train_val
        X_test_final = X_test_final_split
        y_test_final = y_test_final_split
    else:
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

    scaler = None
    X_train_processed = X_train_final 
    X_test_processed = X_test_final   

    model_requires_scaling = model_type in ['logistic_regression', 'elastic_net', 'sgd_classifier']
    if scale_features and model_requires_scaling:
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns, index=X_train_final.index)
        X_test_processed = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns, index=X_test_final.index)
        print(f"Features standardisées pour l'entraînement final du modèle {model_type}")
    elif scale_features and not model_requires_scaling:
         print(f"Note: La standardisation globale est activée mais non appliquée pour {model_type} car généralement non requise.")

    default_params_structure = {
        'logistic_regression': {'solver': 'liblinear', 'max_iter': 1000},
        'elastic_net': {'loss': 'log_loss', 'penalty': 'elasticnet', 'max_iter': 1000, 'tol': 1e-3}, # Ajout régularisation adaptative (l1_ratio, alpha via model_params)
        'random_forest': {'n_jobs': -1, 'ccp_alpha': 0.0}, # ccp_alpha pour élagage
        'xgboost_classifier': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
        'quantile_regression': {'loss': 'quantile'},
        'bidirectional_lstm': {'activation': 'sigmoid', 'output_units': 1, 'dropout_rate': 0.2, 'l1_reg': 0.0, 'l2_reg': 0.0},
        'temporal_cnn': {'activation': 'sigmoid', 'output_units': 1, 'dropout_rate': 0.2, 'l1_reg': 0.0, 'l2_reg': 0.0},
        'gaussian_process_regressor': {}, # Les params sont souvent dans le noyau
        'hierarchical_bayesian': {} # Les params sont spécifiques au modèle défini
    }
    
    current_constructor_params = default_params_structure.get(model_type, {}).copy()
    # Appliquer d'abord les paramètres fournis par l'utilisateur qui n'ont pas été pop() pour Optuna
    # model_params contient les paramètres originaux moins ceux utilisés par la logique Optuna.
    # best_params_from_optuna contient les paramètres optimisés.
    
    # On fusionne : params par défaut < params utilisateur restants < params optimisés
    # `model_params` a déjà eu les clés d'Optuna retirées.
    # `best_params_from_optuna` contient les clés optimisées.
    
    temp_params = model_params.copy() # Commence avec les params utilisateur restants
    temp_params.update(best_params_from_optuna) # Surcharge/ajoute avec les params optimisés
    
    current_constructor_params.update(temp_params) # Met à jour les defaults avec le résultat

    # Gestion spécifique de l'early stopping pour XGBoost pour l'entraînement final
    # Si 'early_stopping_rounds' est dans les meilleurs paramètres d'Optuna, on le garde pour le fit final.
    # Sinon, on vérifie s'il était dans les model_params originaux.
    final_fit_early_stopping_rounds = best_params_from_optuna.get('early_stopping_rounds')
    if final_fit_early_stopping_rounds is None: # S'il n'a pas été optimisé par Optuna
        final_fit_early_stopping_rounds = model_params.get('early_stopping_rounds')


    if 'random_state' not in current_constructor_params:
        current_constructor_params['random_state'] = random_state
    
    final_class_weight_dict = None
    if y_train_final.dtype == 'int' and y_train_final.nunique() > 1:
        unique_classes_final = np.unique(y_train_final)
        class_weights_final_values = compute_class_weight('balanced', classes=unique_classes_final, y=y_train_final)
        final_class_weight_dict = {cls: weight for cls, weight in zip(unique_classes_final, class_weights_final_values)}
        print(f"Poids des classes pour l'entraînement final (sur y_train_final): {final_class_weight_dict}")
 
    is_sklearn_classifier_type = model_type in ['logistic_regression', 'elastic_net', 'random_forest', 'xgboost_classifier']
    # Les nouveaux modèles de NN peuvent aussi être des classifieurs
    is_nn_classifier_type = model_type in ['bidirectional_lstm', 'temporal_cnn'] and current_constructor_params.get('activation') == 'sigmoid' # current_constructor_params car constructor_params n'est pas encore défini

    if is_sklearn_classifier_type and final_class_weight_dict:
        if model_type == 'xgboost_classifier':
            if 'scale_pos_weight' not in current_constructor_params:
                counts_train_final = y_train_final.value_counts()
                if 0 in counts_train_final and 1 in counts_train_final and counts_train_final[1] > 0:
                    current_constructor_params['scale_pos_weight'] = counts_train_final[0] / counts_train_final[1]
        elif 'class_weight' not in current_constructor_params:
            current_constructor_params['class_weight'] = final_class_weight_dict
            
    keys_to_remove_for_constructor = [
        'feature_groups', 'temporal_stratification_params',
        'custom_objective_params', 'temporal_weighting_params',
        # 'early_stopping_rounds', # On le gère séparément pour le fit final
        # Clés spécifiques à Optuna qui pourraient rester si Optuna n'est pas utilisé
        'optimize_hyperparameters', 'optuna_n_trials', 'optuna_direction', 'optuna_cv_splits'
    ]
    
    # On retire 'early_stopping_rounds' des constructor_params car il est pour le .fit()
    constructor_params = {k: v for k, v in current_constructor_params.items() if k not in keys_to_remove_for_constructor and k != 'early_stopping_rounds'}


    if current_constructor_params.get('feature_groups') and model_type == 'elastic_net':
        print(f"Info: 'feature_groups' pour ElasticNet ({current_constructor_params.get('feature_groups')}) n'est pas utilisé activement dans cette version.")
    if current_constructor_params.get('temporal_stratification_params') and model_type == 'random_forest':
        print(f"Info: 'temporal_stratification_params' ({current_constructor_params.get('temporal_stratification_params')}) pour RandomForest non implémenté. Envisager pré-échantillonnage.")
    
    final_sample_weights = None
    if current_constructor_params.get('temporal_weighting_params') and model_type == 'xgboost_classifier':
        print(f"Info: 'temporal_weighting_params' ({current_constructor_params.get('temporal_weighting_params')}) pour XGBoost. Calcul de final_sample_weights à implémenter si besoin pour le fit final.")

    # Pour les modèles NN, s'assurer que input_shape est fourni si nécessaire
    # X_train_processed est un DataFrame pandas. Les modèles NN attendent souvent des arrays numpy.
    # Et pour LSTM/CNN, souvent une forme 3D (samples, timesteps, features)
    # Ceci est une simplification; une préparation de données plus robuste serait nécessaire.
    if model_type in ['bidirectional_lstm', 'temporal_cnn']:
        if 'input_shape' not in constructor_params:
            # Suppose (timesteps=1, num_features) pour l'instant si non spécifié
            # Idéalement, cela devrait être géré par une fonction de préparation de données spécifique aux séquences
            constructor_params['input_shape'] = (1, X_train_processed.shape[1]) # (timesteps, features)
            print(f"Avertissement: 'input_shape' non fourni pour {model_type}, utilisation de {constructor_params['input_shape']}. Adapter les données en conséquence.")
        # Les données X_train_processed et X_test_processed devront peut-être être remodelées en 3D.
        # Exemple: X_train_nn = X_train_processed.values.reshape((X_train_processed.shape[0], constructor_params['input_shape'][0], constructor_params['input_shape'][1]))
        # Pour l'instant, les modèles placeholder ne l'utilisent pas activement.

    print(f"Construction du modèle final {model_type} avec les paramètres constructeur: {constructor_params}")
    if model_type == 'logistic_regression':
        model = LogisticRegression(**constructor_params)
    elif model_type == 'elastic_net':
        model = SGDClassifier(**constructor_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**constructor_params)
    elif model_type == 'xgboost_classifier':
        if callable(constructor_params.get('objective')):
            print("Utilisation d'une fonction objectif personnalisée pour XGBoost.")
        eval_metric_val = constructor_params.pop('eval_metric', default_params_structure.get(model_type,{}).get('eval_metric'))
        if eval_metric_val: constructor_params['eval_metric'] = eval_metric_val
        model = xgb.XGBClassifier(**constructor_params)
    elif model_type == 'quantile_regression':
        if 'alpha' not in constructor_params: constructor_params['alpha'] = 0.5
        constructor_params.pop('class_weight', None)
        model = GradientBoostingRegressor(**constructor_params)
    elif model_type == 'bidirectional_lstm':
        # Assurer que X_train_processed est au format attendu (ex: numpy array, potentiellement 3D)
        # X_train_nn = X_train_processed.values.reshape((X_train_processed.shape[0], constructor_params['input_shape'][0], constructor_params['input_shape'][1]))
        model = BidirectionalLSTMModel(**constructor_params)
    elif model_type == 'temporal_cnn':
        # X_train_nn = X_train_processed.values.reshape((X_train_processed.shape[0], constructor_params['input_shape'][0], constructor_params['input_shape'][1]))
        model = TemporalCNNModel(**constructor_params)
    elif model_type == 'gaussian_process_regressor':
        # GP peut nécessiter la forme de X_train pour l'initialisation du noyau si non spécifié autrement
        constructor_params['X_train_shape'] = X_train_processed.shape
        model = GaussianProcessRegressionModel(**constructor_params)
    elif model_type == 'hierarchical_bayesian':
        model = HierarchicalBayesianModelPlaceholder(**constructor_params)
    else:
        raise ValueError(f"Type de modèle non supporté : {model_type}.")

    print(f"Entraînement du modèle final {model_type} sur {len(X_train_processed)} échantillons.")
    fit_final_args = {}
    # Les modèles NN et Bayésiens peuvent avoir des paramètres de fit différents (ex: epochs, batch_size)
    # Ceux-ci devraient être passés via model_params et extraits ici si nécessaire.
    # Pour l'instant, les méthodes fit des placeholders sont simples.
    
    # Préparation des données pour les modèles NN (exemple simple, à améliorer)
    X_fit_data = X_train_processed
    y_fit_data = y_train_final
    X_test_data_eval = X_test_processed

    # Gestion de l'early stopping pour XGBoost lors du fit final
    if model_type == 'xgboost_classifier' and final_fit_early_stopping_rounds is not None:
        if isinstance(X_test_processed, pd.DataFrame) and not y_test_final.empty and X_test_data_eval is not None and len(X_test_data_eval) > 0: # Ajout vérification X_test_data_eval
            fit_final_args['eval_set'] = [(X_test_data_eval, y_test_final)] # Utiliser le vrai test set pour early stopping final
            fit_final_args['early_stopping_rounds'] = final_fit_early_stopping_rounds
            # fit_final_args['verbose'] = False # Déjà géré dans _objective_optuna et peut être redondant ici
            print(f"  Utilisation pour fit XGBoost final: early_stopping_rounds={final_fit_early_stopping_rounds} sur le jeu de test final.")
        else:
            print("Avertissement: Early stopping pour XGBoost final non appliqué car X_test_processed/y_test_final n'est pas prêt ou vide.")


    if model_type in ['bidirectional_lstm', 'temporal_cnn']:
        # Supposons que les données doivent être des numpy arrays et potentiellement remodelées
        if isinstance(X_train_processed, pd.DataFrame): X_fit_data = X_train_processed.values
        if isinstance(y_train_final, pd.Series): y_fit_data = y_train_final.values
        if isinstance(X_test_processed, pd.DataFrame): X_test_data_eval = X_test_processed.values
        
        # Remodelage si input_shape suggère des timesteps > 1
        # Ceci est une simplification. Une vraie implémentation nécessiterait une gestion des séquences plus robuste.
        # et que constructor_params['input_shape'] est bien (timesteps, features)
        if constructor_params.get('input_shape') and len(constructor_params['input_shape']) == 2 :
            timesteps, num_features = constructor_params['input_shape']
            if timesteps > 0 : # S'assurer que timesteps est défini et > 0
                 if X_fit_data is not None:
                    X_fit_data = X_fit_data.reshape((X_fit_data.shape[0], timesteps, num_features))
                 if X_test_data_eval is not None:
                    X_test_data_eval = X_test_data_eval.reshape((X_test_data_eval.shape[0], timesteps, num_features))

        fit_epochs = current_constructor_params.get('epochs', model_params.get('epochs', 10))
        fit_batch_size = current_constructor_params.get('batch_size', model_params.get('batch_size', 32))
        fit_early_stopping_patience = current_constructor_params.get('early_stopping_patience', model_params.get('early_stopping_patience'))

        fit_final_args.update({'epochs': fit_epochs, 'batch_size': fit_batch_size})
        print(f"  Utilisation pour fit NN: epochs={fit_epochs}, batch_size={fit_batch_size}")
        
        if fit_early_stopping_patience and not y_test_final.empty and X_test_data_eval is not None and len(X_test_data_eval) > 0 :
             fit_final_args['validation_data'] = (X_test_data_eval, y_test_final)
             fit_final_args['early_stopping_patience'] = fit_early_stopping_patience
             print(f"  Early stopping pour NN (placeholder) avec patience {fit_early_stopping_patience} sur le jeu de test final.")
        elif fit_early_stopping_patience:
            print("Avertissement: Early stopping pour NN non appliqué car y_test_final est vide ou X_test_data_eval est vide/None.")


    elif model_type == 'gaussian_process_regressor':
        fit_optimize_restarts = current_constructor_params.get('optimize_restarts', model_params.get('optimize_restarts', 5))
        fit_final_args.update({'optimize_restarts': fit_optimize_restarts})
        if isinstance(X_train_processed, pd.DataFrame): X_fit_data = X_train_processed.values
        if isinstance(y_train_final, pd.Series): y_fit_data = y_train_final.values.reshape(-1,1)
        if isinstance(X_test_processed, pd.DataFrame): X_test_data_eval = X_test_processed.values


    elif model_type == 'hierarchical_bayesian':
        fit_draws = current_constructor_params.get('draws', model_params.get('draws', 2000))
        fit_tune = current_constructor_params.get('tune', model_params.get('tune', 1000))
        fit_chains = current_constructor_params.get('chains', model_params.get('chains', 2))
        fit_final_args.update({'draws': fit_draws, 'tune': fit_tune, 'chains': fit_chains})
        if isinstance(X_train_processed, pd.DataFrame): X_fit_data = X_train_processed.values
        if isinstance(y_train_final, pd.Series): y_fit_data = y_train_final.values
        if isinstance(X_test_processed, pd.DataFrame): X_test_data_eval = X_test_processed.values

    model.fit(X_fit_data, y_fit_data, sample_weight=final_sample_weights, **fit_final_args)
    
    feature_importances_dict: Optional[Dict[str, float]] = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances_dict = dict(zip(X_train_final.columns, importances))
        print(f"Importance des features pour {model_type}: {feature_importances_dict}")
    elif model_type == 'logistic_regression' and hasattr(model, 'coef_'):
        # Pour la régression logistique, les coefficients peuvent servir d'indicateur d'importance
        # Note: Pour la classification multiclasse, coef_ aura une forme (n_classes, n_features)
        # Pour la binaire, (1, n_features) ou (n_features,)
        coefs = model.coef_
        if coefs.ndim == 1: # Cas binaire simple ou déjà aplati
            feature_importances_dict = dict(zip(X_train_final.columns, coefs))
        elif coefs.ndim == 2 and coefs.shape[0] == 1: # Cas binaire avec shape (1, n_features)
            feature_importances_dict = dict(zip(X_train_final.columns, coefs[0]))
        else: # Multiclasse ou autre cas non géré simplement ici
            print(f"Coefficients pour {model_type} (shape {coefs.shape}) non directement convertis en feature_importances simples.")
        if feature_importances_dict:
             print(f"Coefficients (comme importance) pour {model_type}: {feature_importances_dict}")


    # Stocker également les paramètres de fit qui ne sont pas des paramètres de constructeur (ex: early stopping)
    # pour référence, si nécessaire.
    # TODO: Ajouter des méta-informations pour la traçabilité (version du code, date, hash des données d'entraînement si possible)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': list(X_train_final.columns),
        'model_type': model_type,
        'training_constructor_params': constructor_params, # Paramètres passés au __init__
        'training_fit_params': fit_final_args, # Paramètres passés au .fit()
        'feature_importances': feature_importances_dict, # Ajout de l'importance des features
        'training_timestamp': pd.Timestamp.now().isoformat(), # Ajout timestamp d'entraînement
        'code_version': get_git_revision_hash(), # Tentative d'ajout de la version du code
        'data_hash': calculate_data_hash(X_train_final, y_train_final), # Tentative d'ajout du hash des données
        'dependencies_versions': { # Ajout des versions des dépendances
            'scikit-learn': sklearn.__version__,
            'xgboost': xgboost.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'optuna': optuna.__version__
            # Ajouter d'autres dépendances clés si nécessaire (ex: tensorflow, torch)
        }
    }
    
    # TODO: Ajouter un placeholder pour la détection de drift conceptuel après l'entraînement
    #       Par exemple, comparer les statistiques des données d'entraînement avec des données de référence
    #       ou des distributions attendues.
    #       `check_concept_drift(X_train_final, y_train_final, reference_data_stats)`

    if model_type in ['bidirectional_lstm', 'temporal_cnn'] and hasattr(model, 'model') and hasattr(model.model, 'save'): # Vérifie model.model
        print(f"Note: Pour les modèles Keras, utiliser model.model.save() est préférable. Joblib est utilisé pour la structure ici.")
    joblib.dump(model_data, model_path)

    results = {} # Renommé metrics en results pour inclure plus que les métriques
    # Ajuster la section des métriques pour les nouveaux modèles
    if is_sklearn_classifier_type or (model_type in ['bidirectional_lstm', 'temporal_cnn'] and model.activation == 'sigmoid'): # NN classifier
        # Utiliser X_test_data_eval qui est potentiellement un numpy array
        predictions_raw = model.predict(X_test_data_eval)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba_output = model.predict_proba(X_test_data_eval)
            if isinstance(proba_output, tuple):
                 print(f"predict_proba pour {model_type} a retourné un tuple, non utilisé pour AUC/classification binaire direct.")
            elif proba_output.ndim == 2 and proba_output.shape[1] == 2:
                probabilities = proba_output[:, 1]
            elif proba_output.ndim == 1:
                probabilities = proba_output
            else:
                print(f"Sortie de predict_proba pour {model_type} non standard pour AUC binaire, shape: {proba_output.shape}")
        elif model_type in ['bidirectional_lstm', 'temporal_cnn'] and model.activation == 'sigmoid' and model.output_units == 1:
             # Si predict_proba n'est pas là mais c'est un classifieur binaire NN, predict() donne les probas
             probabilities = predictions_raw.ravel() if predictions_raw.ndim > 1 else predictions_raw


        # Conversion des prédictions en classes
        if probabilities is not None and (y_test_final.dtype == 'int' or y_test_final.nunique() <=2) :
             predictions_classes = (probabilities > 0.5).astype(int)
        elif predictions_raw.ndim == 2 and predictions_raw.shape[1] > 1 and (y_test_final.dtype == 'int' or y_test_final.nunique() <=2): # Cas multiclasse
             predictions_classes = np.argmax(predictions_raw, axis=1)
        elif predictions_raw.ndim == 1 and (y_test_final.dtype == 'int' or y_test_final.nunique() <=2): # Cas binaire où predict retourne déjà des classes
             predictions_classes = predictions_raw.astype(int)
        else: # Fallback ou cas de régression traité ailleurs
             predictions_classes = predictions_raw


        results['accuracy'] = accuracy_score(y_test_final, predictions_classes)
        print(f"Modèle '{model_type}' entraîné et sauvegardé dans '{model_path}'.")
        print(f"Accuracy (test final): {results['accuracy']:.4f}")
        
        if probabilities is not None:
            try:
                if y_test_final.nunique() <= 2: # Assure que la cible est binaire pour AUC
                    results['auc'] = roc_auc_score(y_test_final, probabilities)
                    print(f"AUC Score (test final): {results['auc']:.4f}")
                else:
                    print("AUC non calculé car la cible n'est pas binaire.")
                    results['auc'] = None
            except ValueError as e:
                print(f"Impossible de calculer l'AUC (test final): {e}")
                results['auc'] = None
        results['classification_report'] = classification_report(y_test_final, predictions_classes, output_dict=True, zero_division=0)
        print("\nRapport de classification (test final):")
        print(classification_report(y_test_final, predictions_classes, zero_division=0))

    elif model_type == 'quantile_regression':
        predictions = model.predict(X_test_data_eval)
        results['mse'] = mean_squared_error(y_test_final, predictions)
        print(f"Modèle '{model_type}' (quantile: {model.alpha if hasattr(model, 'alpha') else 'N/A'}) sauvegardé.")
        print(f"MSE (test final): {results['mse']:.4f}")
    
    elif model_type == 'gaussian_process_regressor':
        predictions_mean = model.predict(X_test_data_eval)
        if predictions_mean.ndim > 1 and predictions_mean.shape[1] == 1:
            predictions_mean = predictions_mean.ravel()

        results['mse'] = mean_squared_error(y_test_final, predictions_mean)
        print(f"Modèle '{model_type}' entraîné et sauvegardé.")
        print(f"MSE (test final, basé sur la moyenne prédite): {results['mse']:.4f}")
        # if hasattr(model.model, 'log_likelihood'): results['log_likelihood'] = model.model.log_likelihood() # Si GPy/GPflow
    
    elif model_type == 'hierarchical_bayesian':
        predictions_mean_hbm = model.predict(X_test_data_eval)
        if predictions_mean_hbm.ndim > 1 and predictions_mean_hbm.shape[1] == 1:
            predictions_mean_hbm = predictions_mean_hbm.ravel()

        results['mse'] = mean_squared_error(y_test_final, predictions_mean_hbm)
        print(f"Modèle '{model_type}' (placeholder) entraîné et sauvegardé.")
        print(f"MSE (test final, basé sur la prédiction ponctuelle): {results['mse']:.4f}")
        # results['trace_summary'] = "Placeholder for Arviz summary" # Si PyMC et ArviZ sont utilisés
    
    results['feature_importances'] = feature_importances_dict
    # TODO: Ajouter d'autres informations pertinentes au retour si nécessaire (ex: rapport d'anomalie)

    return results


def load_model_and_predict(X_new: pd.DataFrame, model_path: str = 'models_store/model.joblib', return_probabilities: bool = True, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Charge un modèle sauvegardé et effectue des prédictions.

    Args:
        X_new (pd.DataFrame): Nouvelles données pour la prédiction.
        model_path (str): Chemin vers le fichier du modèle sauvegardé.
        return_probabilities (bool): Si True et applicable, retourne les probabilités.
                                     Pour les modèles de régression, cela peut être ignoré ou adapté.
        return_uncertainty (bool): Si True et que le modèle le supporte (ex: GP), retourne aussi une mesure d'incertitude.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
            - np.ndarray des prédictions (ou probabilités).
            - Tuple (prédictions, incertitude) si return_uncertainty est True et supporté.
    """
    # TODO: Ajouter un placeholder pour la détection de drift conceptuel avant la prédiction
    #       Comparer les statistiques de X_new avec celles des données d'entraînement (stockées dans model_data ou séparément)
    #       `check_data_drift(X_new, model_data.get('training_data_stats'))`

    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        expected_features = model_data.get('feature_columns')
        training_timestamp = model_data.get('training_timestamp')
        code_version = model_data.get('code_version', 'N/A')
        data_hash_train = model_data.get('data_hash', 'N/A')
        dependencies_versions = model_data.get('dependencies_versions', {})
        
        print(f"Modèle chargé depuis {model_path}:")
        print(f"  - Type: {model_data.get('model_type', 'N/A')}")
        print(f"  - Entraîné le: {training_timestamp}")
        print(f"  - Version code (entraînement): {code_version}")
        print(f"  - Hash données (entraînement): {data_hash_train}")
        if dependencies_versions:
            print(f"  - Versions des dépendances (entraînement):")
            for lib, version in dependencies_versions.items():
                print(f"    - {lib}: {version}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}.")
    except Exception as e:
        raise Exception(f"Erreur chargement modèle {model_path}: {e}")

    if expected_features:
        if list(X_new.columns) != expected_features:
            print("Attention: Les colonnes de X_new ne correspondent pas ou ne sont pas dans le bon ordre. Réorganisation...")
            try:
                X_new = X_new[expected_features]
            except KeyError as e:
                raise ValueError(f"Colonnes manquantes dans X_new pour la prédiction: {e}. Attendu: {expected_features}")

    X_scaled = X_new
    if scaler:
        X_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
    
    X_predict_data = X_scaled
    model_type = model_data.get('model_type')
    # training_params = model_data.get('training_params', {}) # Moins utilisé ici, plus pour le debug
    training_constructor_params = model_data.get('training_constructor_params', {})


    # Préparation des données pour les modèles NN (similaire à train_model)
    if model_type in ['bidirectional_lstm', 'temporal_cnn']:
        if isinstance(X_scaled, pd.DataFrame):
            X_predict_data = X_scaled.values
        
        input_shape_from_train = training_constructor_params.get('input_shape')
        if input_shape_from_train and len(input_shape_from_train) == 2 :
            timesteps, num_features = input_shape_from_train
            if timesteps > 0 and X_predict_data.shape[1] == num_features : # S'assurer que le nombre de features correspond
                 X_predict_data = X_predict_data.reshape((X_predict_data.shape[0], timesteps, num_features))
            elif timesteps > 0 and X_predict_data.shape[1] != num_features:
                 print(f"Avertissement: Incohérence de features pour NN. Attendu {num_features}, obtenu {X_predict_data.shape[1]}. Remodelage peut échouer.")
                 # Tenter le remodelage quand même, mais cela risque d'échouer si le nombre total d'éléments ne correspond pas.
                 # Une meilleure gestion des erreurs ou une validation plus stricte serait nécessaire ici.
                 try:
                     X_predict_data = X_predict_data.reshape((X_predict_data.shape[0], timesteps, num_features))
                 except ValueError as reshape_error:
                     raise ValueError(f"Erreur de remodelage pour NN: {reshape_error}. Vérifiez input_shape et les données d'entrée.") from reshape_error


    elif model_type in ['gaussian_process_regressor', 'hierarchical_bayesian']:
        if isinstance(X_scaled, pd.DataFrame):
            X_predict_data = X_scaled.values

    # Gestion de la prédiction et de l'incertitude
    predictions: np.ndarray
    uncertainty: Optional[np.ndarray] = None

    if hasattr(model, 'predict_proba'):
        proba_output = model.predict_proba(X_predict_data)
        
        if model_type == 'gaussian_process_regressor' and isinstance(proba_output, tuple) and len(proba_output) == 2:
            # GP retourne (moyenne, variance) via predict_proba placeholder
            predictions = proba_output[0].ravel() # Moyenne
            if return_uncertainty:
                uncertainty = np.sqrt(proba_output[1].ravel()) # Écart-type comme incertitude
        elif model_type == 'hierarchical_bayesian' and proba_output.ndim == 2 and proba_output.shape[1] > 1:
            # HBM placeholder retourne des échantillons. On prend la moyenne comme prédiction.
            predictions = np.mean(proba_output, axis=1)
            if return_uncertainty:
                uncertainty = np.std(proba_output, axis=1) # Écart-type des échantillons comme incertitude
        elif proba_output.ndim == 2 and proba_output.shape[1] == 2: # Cas classifieur binaire standard
            predictions = proba_output[:, 1] # Proba de la classe 1
        elif proba_output.ndim == 1: # Supposé probas classe positive déjà
            predictions = proba_output
        else: # Cas non standard
            print(f"Avertissement: Sortie de predict_proba pour {model_type} de forme {proba_output.shape}. Utilisation directe.")
            predictions = proba_output # Peut être multidimensionnel

    elif model_type in ['bidirectional_lstm', 'temporal_cnn'] and \
         training_constructor_params.get('activation') == 'sigmoid' and \
         training_constructor_params.get('output_units') == 1:
        # Classifieur binaire NN, predict() donne les probas
        predictions = model.predict(X_predict_data).ravel()
    else: # Modèles de régression ou classifieurs sans predict_proba clair
        predictions = model.predict(X_predict_data)
        if predictions.ndim > 1 and predictions.shape[1] == 1: # S'assurer que c'est 1D pour la régression simple
            predictions = predictions.ravel()

    if not return_probabilities and (model_type in ['logistic_regression', 'random_forest', 'xgboost_classifier'] or \
                                   (model_type in ['bidirectional_lstm', 'temporal_cnn'] and training_constructor_params.get('activation') == 'sigmoid')):
        # Convertir les probabilités en classes si return_probabilities est False pour les classifieurs
        predictions = (predictions > 0.5).astype(int)

    if return_uncertainty:
        return predictions, uncertainty
    else:
        return predictions

if __name__ == '__main__':
    data = {
        'timestamp': pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 31)] + [f'2023-02-{i:02d}' for i in range(1, 21)]),
        'close': np.random.rand(50) * 20 + 100,
        'sma_10': np.random.rand(50) * 5 + 100,
        'ema_10': np.random.rand(50) * 5 + 100,
        'rsi_14': np.random.rand(50) * 50 + 25,
        'other_feature': np.random.rand(50) * 10
    }
    sample_df = pd.DataFrame(data)
    sample_df.set_index('timestamp', inplace=True)
    
    actual_features_for_model = ['sma_10', 'ema_10', 'rsi_14', 'other_feature']

    print("--- Test de Classification ---")
    X_clf, y_clf = prepare_data_for_model(sample_df.copy(), feature_columns=actual_features_for_model, problem_type='classification')

    if not X_clf.empty:
        import os
        os.makedirs('models_store', exist_ok=True)

        print("\n--- Entraînement Random Forest (avec Optuna) ---")
        rf_opt_params = {
            'optimize_hyperparameters': True, 
            'optuna_n_trials': 5, 
            'optuna_direction': 'maximize',
        }
        metrics_rf_opt = train_model(X_clf, y_clf, model_type='random_forest', model_params=rf_opt_params, model_path='models_store/test_rf_opt_model.joblib', scale_features=False)
        print(f"\nMétriques (Random Forest Optuna): {metrics_rf_opt}")

        print("\n--- Entraînement XGBoost Classifier (avec Optuna) ---")
        xgb_opt_params = {
            'optimize_hyperparameters': True,
            'optuna_n_trials': 5,
            'objective': 'binary:logistic', 
        }
        metrics_xgb_opt = train_model(X_clf, y_clf, model_type='xgboost_classifier', model_params=xgb_opt_params, model_path='models_store/test_xgb_opt_model.joblib', scale_features=False)
        print(f"\nMétriques (XGBoost Optuna): {metrics_xgb_opt}")

        print("\n--- Entraînement Elastic Net ---")
        en_params = {'l1_ratio': 0.15, 'alpha': 0.1, 'feature_groups': ['groupA', 'groupB']} 
        metrics_en = train_model(X_clf, y_clf, model_type='elastic_net', model_params=en_params, model_path='models_store/test_elasticnet_model.joblib', scale_features=True)
        print(f"\nMétriques (Elastic Net): {metrics_en}")

    else:
        print("Pas assez de données de classification pour l'entraînement.")

    print("\n--- Test de Régression Quantile ---")
    X_reg, y_reg = prepare_data_for_model(sample_df.copy(), feature_columns=actual_features_for_model, problem_type='regression')
    if not X_reg.empty:
        qr_params = {'alpha': 0.5, 'n_estimators': 60} 
        metrics_qr = train_model(X_reg, y_reg, model_type='quantile_regression', model_params=qr_params, model_path='models_store/test_quantile_model.joblib', scale_features=False)
        print(f"\nMétriques (Régression Quantile): {metrics_qr}")

        if len(X_reg) > 3:
            X_new_reg_sample = X_reg.tail(3).copy()
            preds_qr = load_model_and_predict(X_new_reg_sample, model_path='models_store/test_quantile_model.joblib', return_probabilities=False)
            print(f"Prédictions Régression Quantile (valeurs):\n{preds_qr}")
    else:
        print("Pas assez de données de régression pour l'entraînement.")

    print("\n--- Test de chargement modèle XGBoost optimisé et prédiction ---")
    if not X_clf.empty and os.path.exists('models_store/test_xgb_opt_model.joblib'):
        if len(X_clf) > 3:
            X_new_clf_sample = X_clf.tail(3).copy()
            preds_xgb_probs = load_model_and_predict(X_new_clf_sample, model_path='models_store/test_xgb_opt_model.joblib', return_probabilities=True)
            print(f"Prédictions XGBoost Opt (probabilités):\n{preds_xgb_probs}")
            preds_xgb_classes = load_model_and_predict(X_new_clf_sample, model_path='models_store/test_xgb_opt_model.joblib', return_probabilities=False)
            print(f"Prédictions XGBoost Opt (classes):\n{preds_xgb_classes}")
    else:
        print("Modèle XGBoost optimisé non trouvé ou pas de données pour tester la prédiction.")
