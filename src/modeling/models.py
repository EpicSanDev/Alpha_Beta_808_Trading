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
# Pour les réseaux de neurones (LSTM, CNN) - avec compatibilité
from .tensorflow_compat import (
    tf, Sequential, LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, 
    GlobalAveragePooling1D, Reshape, Add, Activation, BatchNormalization, 
    regularizers, EarlyStopping, require_tensorflow, is_tensorflow_available,
    TENSORFLOW_AVAILABLE
)


# import torch
# import torch.nn as nn
# import torch.optim as optim

# Pour les Processus Gaussiens
try:
    import gpflow
    GPFLOW_AVAILABLE = True
except ImportError:
    GPFLOW_AVAILABLE = False
    class MockGPflow:
        def __getattr__(self, name):
            return MockGPflow()
        def __call__(self, *args, **kwargs):
            return MockGPflow()
    gpflow = MockGPflow()

# Pour les Modèles Bayésiens Hiérarchiques
import pymc as pm # Changé de pymc3 à pymc
import arviz as az
# import stan # Stan est plus complexe à intégrer directement, on se concentre sur PyMC

# --- Définitions des Nouveaux Modèles ---

# Section 5.3.3: Réseaux de Neurones Spécialisés

class BidirectionalLSTMModel:
    """
    Modèle LSTM Bidirectionnel pour les séries temporelles financières.

    Ce modèle utilise des couches LSTM bidirectionnelles pour capturer les dépendances temporelles
    dans les deux directions (passé et futur relatif à un point temporel).

    Args:
        input_shape (tuple): Forme des données d'entrée, typiquement (timesteps, num_features).
                             Pour une utilisation avec `timesteps=1`, cela devient (1, num_features).
        num_layers (int): Nombre de couches LSTM bidirectionnelles empilées.
        units (int): Nombre d'unités (neurones) dans chaque couche LSTM.
        output_units (int): Nombre d'unités dans la couche de sortie Dense. Typiquement 1 pour la classification binaire ou la régression.
        activation (str): Fonction d'activation pour la couche de sortie (ex: 'sigmoid' pour binaire, 'linear' pour régression).
        dropout_rate (float): Taux de dropout à appliquer après chaque couche LSTM (si variational_dropout=False).
        variational_dropout (bool): Si True, utilise le dropout récurrent au sein des couches LSTM (recurrent_dropout).
        l1_reg (float): Facteur de régularisation L1 pour les poids du noyau des couches LSTM et Dense.
        l2_reg (float): Facteur de régularisation L2 pour les poids du noyau des couches LSTM et Dense.
        use_temporal_attention (bool): Placeholder pour l'utilisation future de mécanismes d'attention temporelle.
        use_multi_resolution (bool): Placeholder pour l'utilisation future de features multi-résolution.
        temporal_regularization_factor (float): Placeholder pour un facteur de régularisation temporelle personnalisé.
        **kwargs: Arguments supplémentaires passés par Keras.
    """
    def __init__(self, input_shape, num_layers=1, units=50, output_units=1, activation='sigmoid', dropout_rate=0.2, variational_dropout=False, l1_reg=0.0, l2_reg=0.0, use_temporal_attention=False, use_multi_resolution=False, temporal_regularization_factor=None, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for BidirectionalLSTMModel but is not installed.")
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
        # Potentielle dépendance: tensorflow.kerasself.input_shape = input_shape
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
        kernel_regularizer = None
        if self.l1_reg > 0 and self.l2_reg > 0:
            kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        elif self.l1_reg > 0:
            kernel_regularizer = regularizers.l1(self.l1_reg)
        elif self.l2_reg > 0:
            kernel_regularizer = regularizers.l2(self.l2_reg)

        model = Sequential()
        for i in range(self.num_layers):
            return_sequences = True if i < self.num_layers - 1 else False
            # TODO: Gérer le dropout variationnel (recurrent_dropout dans LSTM)
            lstm_layer = LSTM(self.units, 
                              return_sequences=return_sequences, 
                              kernel_regularizer=kernel_regularizer,
                              recurrent_dropout=self.dropout_rate if self.variational_dropout else 0) # Dropout variationnel
            if i == 0:
                model.add(Bidirectional(lstm_layer, input_shape=self.input_shape))
            else:
                model.add(Bidirectional(lstm_layer))
            
            if self.dropout_rate > 0 and not self.variational_dropout: # Dropout standard si non variationnel
                model.add(Dropout(self.dropout_rate))
        
        # TODO: Implémenter l'attention temporelle si self.use_temporal_attention est True
        # if self.use_temporal_attention:
        #     model.add(AttentionLayer(...)) # Placeholder pour une couche d'attention

        model.add(Dense(self.output_units, activation=self.activation, kernel_regularizer=kernel_regularizer))
        
        loss_function = 'binary_crossentropy' if self.activation == 'sigmoid' else 'mse'
        # TODO: Implémenter la régularisation temporelle via une perte personnalisée si self.temporal_regularization_factor
        # if self.temporal_regularization_factor:
        #    def custom_loss(y_true, y_pred):
        #        base_loss = tf.keras.losses.get(loss_function)(y_true, y_pred)
        #        # Ajouter la pénalité de régularisation temporelle ici
        #        # Par exemple, pénaliser les grandes variations des poids cachés ou des activations sur le temps
        #        # temp_reg = self.temporal_regularization_factor * compute_temporal_penalty(model)
        #        # return base_loss + temp_reg
        #        return base_loss # Placeholder
        #    loss_to_compile = custom_loss
        # else:
        loss_to_compile = loss_function

        model.compile(optimizer='adam', loss=loss_to_compile, metrics=['accuracy' if loss_function == 'binary_crossentropy' else 'mae'])
        print(f"Modèle LSTM Bidirectionnel construit avec Keras: {self.num_layers} couches, {self.units} unités.")
        if self.variational_dropout: print(f"  Dropout variationnel (recurrent_dropout): {self.dropout_rate}")
        return model

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, early_stopping_patience=None, **kwargs):
        callbacks = []
        if validation_data and early_stopping_patience:
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
            callbacks.append(early_stop_callback)
            print(f"  Early stopping activé avec patience={early_stopping_patience} sur val_loss.")
        
        print(f"Entraînement du LSTM Bidirectionnel pour {epochs} époques avec Keras.")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, verbose=kwargs.get('verbose', 1))

    def predict(self, X, **kwargs):
        print("Prédiction avec LSTM Bidirectionnel (Keras).")
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        print("Prédiction de probabilités avec LSTM Bidirectionnel (Keras).")
        # Pour un modèle Keras avec sortie sigmoïde, predict() retourne déjà les probabilités pour la classe positive.
        # Pour la compatibilité avec scikit-learn (qui attend (N, 2) pour binaire), on ajuste.
        raw_preds = self.model.predict(X, **kwargs)
        if self.output_units == 1 and self.activation == 'sigmoid': # Cas binaire
            if raw_preds.ndim == 1: # Si la sortie est déjà (N,)
                return np.vstack((1 - raw_preds, raw_preds)).T # (N, 2)
            elif raw_preds.ndim == 2 and raw_preds.shape[1] == 1: # Si la sortie est (N, 1)
                return np.hstack((1 - raw_preds, raw_preds)) # (N, 2)
        return raw_preds # Cas multi-classe (N, C) ou régression (N, output_units)

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
    Modèle CNN Temporel (similaire à un TCN simplifié) pour les séries temporelles financières.

    Ce modèle utilise des couches de convolution 1D causales pour extraire des features temporelles.
    Des techniques comme BatchNormalization, Dropout, et potentiellement des connexions résiduelles
    peuvent être utilisées.

    Args:
        input_shape (tuple): Forme des données d'entrée, typiquement (timesteps, num_features).
                             Pour une utilisation avec `timesteps=1`, cela devient (1, num_features).
        num_filters (int): Nombre de filtres (neurones) dans chaque couche de convolution.
        kernel_size (int): Taille du noyau (filtre) pour les convolutions 1D.
        num_conv_layers (int): Nombre de couches de convolution 1D empilées.
        output_units (int): Nombre d'unités dans la couche de sortie Dense. Typiquement 1 pour la classification binaire ou la régression.
        activation (str): Fonction d'activation pour la couche de sortie (ex: 'sigmoid' pour binaire, 'linear' pour régression).
        dropout_rate (float): Taux de dropout à appliquer après chaque couche de convolution.
        l1_reg (float): Facteur de régularisation L1 pour les poids du noyau des couches Conv1D et Dense.
        l2_reg (float): Facteur de régularisation L2 pour les poids du noyau des couches Conv1D et Dense.
        use_residual_connections (bool): Si True, tente d'ajouter des connexions résiduelles.
        use_squeeze_excitation (bool): Placeholder pour l'utilisation future de blocs Squeeze-and-Excitation.
        use_hourglass_architecture (bool): Placeholder pour une architecture en sablier.
        temporal_regularization_factor (float): Placeholder pour un facteur de régularisation temporelle personnalisé.
        **kwargs: Arguments supplémentaires.
    """
    def __init__(self, input_shape, num_filters=64, kernel_size=3, num_conv_layers=2, output_units=1, activation='sigmoid', dropout_rate=0.2, l1_reg=0.0, l2_reg=0.0, use_residual_connections=False, use_squeeze_excitation=False, use_hourglass_architecture=False, temporal_regularization_factor=None, **kwargs):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.output_units = output_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_residual_connections = use_residual_connections
        self.use_squeeze_excitation = use_squeeze_excitation
        self.use_hourglass_architecture = use_hourglass_architecture
        self.temporal_regularization_factor = temporal_regularization_factor
        self.model = self._build_model()
        # Potentielle dépendance: tensorflow.keras ou torch

    def _build_model(self):
        kernel_regularizer = None
        if self.l1_reg > 0 and self.l2_reg > 0:
            kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        elif self.l1_reg > 0:
            kernel_regularizer = regularizers.l1(self.l1_reg)
        elif self.l2_reg > 0:
            kernel_regularizer = regularizers.l2(self.l2_reg)

        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        
        # Placeholder pour architecture en sablier (encodeur-décodeur)
        # if self.use_hourglass_architecture:
        #     # Logique d'encodeur
        #     # ...
        #     # Logique de décodeur
        #     # ...
        # else: # Architecture CNN standard
        for i in range(self.num_conv_layers):
            prev_x = x # Pour connexions résiduelles
            # TODO: Implémenter TCN avec des convolutions causales et dilatées si nécessaire
            x = Conv1D(filters=self.num_filters, 
                       kernel_size=self.kernel_size, 
                       padding='causal', # Causal pour les séries temporelles
                       activation='relu', # Souvent ReLU ou variantes dans les TCN
                       kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization()(x) # Souvent utilisé dans les TCN
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            
            # Connexion résiduelle simple (si les dimensions correspondent)
            if self.use_residual_connections:
                if prev_x.shape[-1] == x.shape[-1]: # Même nombre de filtres
                    x = Add()([prev_x, x])
                elif i > 0 : # Si pas la première couche, on peut projeter prev_x
                    prev_x_proj = Conv1D(filters=self.num_filters, kernel_size=1, padding='same')(prev_x)
                    x = Add()([prev_x_proj, x])
            
            # TODO: Implémenter Squeeze-and-Excitation si self.use_squeeze_excitation
            # if self.use_squeeze_excitation:
            #    x = SqueezeExcitationBlock()(x) # Placeholder

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.output_units, activation=self.activation, kernel_regularizer=kernel_regularizer)(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        loss_function = 'binary_crossentropy' if self.activation == 'sigmoid' else 'mse'
        # TODO: Régularisation temporelle via perte personnalisée
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy' if loss_function == 'binary_crossentropy' else 'mae'])
        print(f"Modèle CNN Temporel (TCN-like) construit avec Keras: {self.num_conv_layers} couches Conv1D.")
        return model

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, early_stopping_patience=None, **kwargs):
        callbacks = []
        if validation_data and early_stopping_patience:
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
            callbacks.append(early_stop_callback)
            print(f"  Early stopping activé avec patience={early_stopping_patience} sur val_loss.")

        print(f"Entraînement du CNN Temporel pour {epochs} époques avec Keras.")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, verbose=kwargs.get('verbose', 1))

    def predict(self, X, **kwargs):
        print("Prédiction avec CNN Temporel (Keras).")
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        print("Prédiction de probabilités avec CNN Temporel (Keras).")
        raw_preds = self.model.predict(X, **kwargs)
        if self.output_units == 1 and self.activation == 'sigmoid':
            if raw_preds.ndim == 1:
                return np.vstack((1 - raw_preds, raw_preds)).T
            elif raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
                return np.hstack((1 - raw_preds, raw_preds))
        return raw_preds

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
    def __init__(self, kernel_spec=None, inducing_points_ratio=None, random_state=None, num_features=None, **kwargs): # Ajout num_features
        self.kernel_spec = kernel_spec
        self.inducing_points_ratio = inducing_points_ratio
        self.random_state = random_state
        self.num_features = num_features # Nécessaire pour initialiser le noyau GPflow
        self.model = self._build_model(**kwargs)
        # Dépendance: gpflow

    def _build_model(self, X_train_sample_for_inducing_variable=None, **kwargs): # X_train_shape renommé
        if self.num_features is None:
            raise ValueError("num_features doit être fourni pour initialiser GaussianProcessRegressionModel.")

        # Construction du noyau GPflow
        if self.kernel_spec is None: # Noyau par défaut
            kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*self.num_features) + gpflow.kernels.White(variance=0.1)
            print(f"Utilisation du noyau GPflow par défaut: SquaredExponential + White pour {self.num_features} features.")
        else:
            # TODO: Logique plus avancée pour construire des noyaux composites à partir de kernel_spec
            # Exemple simple:
            if self.kernel_spec.get('type') == 'Matern52':
                kernel = gpflow.kernels.Matern52(lengthscales=self.kernel_spec.get('params', {}).get('lengthscale', [1.0]*self.num_features))
            else: # Fallback sur RBF (SquaredExponential)
                kernel = gpflow.kernels.SquaredExponential(lengthscales=self.kernel_spec.get('params', {}).get('lengthscale', [1.0]*self.num_features))
            print(f"Noyau GPflow construit selon kernel_spec: {self.kernel_spec}")

        # Construction du modèle GPflow
        # Les données X et Y seront passées lors du .fit()
        # Pour les modèles clairsemés (Sparse), il faut des points induisants Z
        if self.inducing_points_ratio and X_train_sample_for_inducing_variable is not None:
            num_inducing = int(X_train_sample_for_inducing_variable.shape[0] * self.inducing_points_ratio)
            if num_inducing == 0 and X_train_sample_for_inducing_variable.shape[0] > 0 : num_inducing = 1 # Au moins un point
            
            if num_inducing > 0:
                # Sélectionner des points induisants (par exemple, aléatoirement depuis X_train_sample)
                # S'assurer que X_train_sample_for_inducing_variable est un array numpy
                if isinstance(X_train_sample_for_inducing_variable, pd.DataFrame):
                    X_numpy_sample = X_train_sample_for_inducing_variable.values
                else:
                    X_numpy_sample = X_train_sample_for_inducing_variable

                if self.random_state is not None:
                    np.random.seed(self.random_state)
                
                idx = np.random.choice(X_numpy_sample.shape[0], size=min(num_inducing, X_numpy_sample.shape[0]), replace=False)
                Z = X_numpy_sample[idx, :].copy()
                
                # Utiliser SVGP (Stochastic Variational Gaussian Process) pour la scalabilité
                # Nécessite des données (X,Y) pour l'initialisation, mais on les passera au fit.
                # On initialise avec des placeholders ou des données factices de la bonne forme.
                # Pour l'instant, on retourne le noyau et Z, le modèle sera créé dans fit.
                print(f"Modèle GPflow (SVGP-like) sera construit avec {num_inducing} points induisants.")
                return {"kernel": kernel, "inducing_variable": Z, "type": "sparse"}
            else:
                 print("Ratio de points induisants spécifié mais num_inducing est 0. Utilisation d'un modèle GP complet.")
                 return {"kernel": kernel, "type": "full"}
        else:
            print("Modèle GPflow (GPR) standard sera construit.")
            return {"kernel": kernel, "type": "full"} # Le modèle GPR sera créé dans fit

    def fit(self, X, y, optimize_restarts=1, max_iterations=100, **kwargs): # optimize_restarts moins pertinent pour gpflow.train.ScipyOptimizer
        # X et y doivent être des tf.Tensor
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        y_tf = tf.convert_to_tensor(y.reshape(-1,1) if y.ndim == 1 else y, dtype=tf.float64)

        model_components = self.model # C'est un dict {"kernel": ..., "type": ..., "inducing_variable": ...}
        kernel = model_components["kernel"]

        if model_components["type"] == "sparse":
            Z = model_components["inducing_variable"]
            if Z.shape[1] != X_tf.shape[1]: # S'assurer que Z a le bon nombre de features
                 raise ValueError(f"Les points induisants Z ont {Z.shape[1]} features, mais X en a {X_tf.shape[1]}.")
            self.gpflow_model = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=X_tf.shape[0])
            print(f"Entraînement du modèle GPflow SVGP avec {Z.shape[0]} points induisants.")
        else: # full GPR
            self.gpflow_model = gpflow.models.GPR((X_tf, y_tf), kernel=kernel, mean_function=None)
            print("Entraînement du modèle GPflow GPR standard.")

        # Optimisation des hyperparamètres du modèle
        opt = gpflow.optimizers.Scipy()
        # Pour SVGP, on optimise les paramètres variationnels et les hyperparamètres du noyau/likelihood
        # Pour GPR, on optimise les hyperparamètres du noyau/likelihood
        
        # Créer une fonction de perte à minimiser (log marginal likelihood négatif)
        if hasattr(self.gpflow_model, 'training_loss'): # Pour SVGP et SGPR
            loss_fn = self.gpflow_model.training_loss_closure((X_tf, y_tf) if model_components["type"] == "sparse" else None)
        else: # Pour GPR, il n'y a pas de training_loss_closure direct, on utilise log_marginal_likelihood
            @tf.function
            def objective_closure():
                return -self.gpflow_model.log_marginal_likelihood()
            loss_fn = objective_closure

        print(f"Optimisation des hyperparamètres du modèle GPflow (max {max_iterations} itérations)...")
        opt_logs = opt.minimize(loss_fn,
                                self.gpflow_model.trainable_variables,
                                options=dict(maxiter=max_iterations),
                                method="L-BFGS-B") # L-BFGS-B est souvent utilisé
        print("Optimisation terminée.")
        if hasattr(opt_logs, 'success') and hasattr(opt_logs, 'message'):
            print(f"  Succès: {opt_logs.success}, Message: {opt_logs.message}")


    def predict(self, X, **kwargs):
        if not hasattr(self, 'gpflow_model'):
            raise RuntimeError("Le modèle GPflow n'a pas été entraîné. Appelez fit() d'abord.")
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        mean, _ = self.gpflow_model.predict_y(X_tf) # predict_y retourne (mean, var) des prédictions Y
        print("Prédiction avec GPflow (moyenne).")
        return mean.numpy()

    def predict_proba(self, X, **kwargs):
        if not hasattr(self, 'gpflow_model'):
            raise RuntimeError("Le modèle GPflow n'a pas été entraîné. Appelez fit() d'abord.")
        X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
        mean, variance = self.gpflow_model.predict_y(X_tf)
        print("Prédiction avec GPflow (moyenne et variance).")
        return mean.numpy(), variance.numpy()

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
    def __init__(self, model_specification_func=None, informative_priors=None, adaptive_mcmc=True, num_features=None, **kwargs): # Ajout num_features
        self.model_specification_func = model_specification_func # Fonction qui définit le modèle PyMC
        self.informative_priors = informative_priors
        self.adaptive_mcmc = adaptive_mcmc
        self.num_features = num_features # Peut être utile pour la fonction de spécification
        self.pymc_model = None # Le modèle PyMC construit
        self.trace = None
        # Dépendance: pymc, arviz
        print("Modèle Bayésien Hiérarchique (PyMC) initialisé.")

    def fit(self, X, y, draws=1000, tune=1000, chains=2, target_accept=0.8, **kwargs): # Réduction draws pour tests rapides
        if self.model_specification_func is None:
            raise ValueError("Une fonction `model_specification_func(X, y, num_features, priors)` doit être fournie pour définir le modèle PyMC.")

        # Convertir X et y en arrays numpy si ce sont des DataFrames/Series pandas
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        if y_np.ndim == 1: y_np = y_np.reshape(-1,1)


        # Construire le modèle PyMC en appelant la fonction fournie
        # La fonction de spécification est responsable de la création du contexte `pm.Model()`
        self.pymc_model = self.model_specification_func(X_np, y_np, self.num_features, self.informative_priors)

        if not isinstance(self.pymc_model, pm.Model):
             raise TypeError("`model_specification_func` doit retourner une instance de `pm.Model`.")

        with self.pymc_model:
            step_method = None
            if self.adaptive_mcmc:
                # NUTS est généralement un bon choix pour les modèles continus complexes
                step_method = pm.NUTS(target_accept=target_accept)
                print(f"Utilisation de l'échantillonneur NUTS adaptatif (target_accept={target_accept}).")
            
            print(f"Début de l'échantillonnage MCMC avec PyMC: {draws} tirages, {tune} burn-in, {chains} chaînes.")
            self.trace = pm.sample(draws, tune=tune, chains=chains, step=step_method, 
                                   return_inferencedata=True, 
                                   random_seed=kwargs.get('random_state', None), # Pour reproductibilité
                                   idata_kwargs={"log_likelihood": True}) # Pour calculs de WAIC/LOO
            print("Échantillonnage MCMC terminé.")
            # Afficher un résumé si arviz est disponible
            if self.trace:
                try:
                    summary = az.summary(self.trace, round_to=2)
                    print("Résumé du tracé MCMC (arviz):")
                    print(summary)
                except Exception as e:
                    print(f"Impossible d'afficher le résumé arviz: {e}")


    def predict(self, X, num_samples_ppc=500, **kwargs): # num_samples_ppc pour le nombre d'échantillons de la prédictive a posteriori
        if self.pymc_model is None or self.trace is None:
            raise RuntimeError("Le modèle PyMC n'a pas été entraîné ou le tracé est manquant. Appelez fit() d'abord.")

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        # Pour faire des prédictions, il faut généralement mettre à jour les données observées
        # ou utiliser pm.set_data si le modèle a été construit avec des pm.MutableData.
        # Ici, on suppose que la fonction de spécification du modèle a une variable `X_shared`
        # et une variable `out_shared` (ou similaire) pour la prédiction.
        # C'est une simplification; une implémentation robuste nécessite une gestion soignée des données partagées.
        
        # Placeholder pour la logique de mise à jour des données partagées dans le modèle PyMC
        # with self.pymc_model:
        #     pm.set_data({'X_input_data': X_np}) # Supposant que 'X_input_data' est le nom de la donnée partagée pour X
        #     ppc = pm.sample_posterior_predictive(self.trace, var_names=["likelihood_obs"], # "likelihood_obs" ou le nom de votre variable de sortie
        #                                           samples=num_samples_ppc, random_seed=kwargs.get('random_state'))
        #
        # # Les prédictions sont souvent la moyenne des échantillons de la distribution prédictive a posteriori
        # # La forme de ppc.posterior_predictive["likelihood_obs"] sera (chaînes, tirages, X_np.shape[0], output_dim)
        # # Il faut l'aplatir et prendre la moyenne sur les échantillons MCMC.
        # pred_samples = ppc.posterior_predictive["likelihood_obs"].stack(samples=("chain", "draw")).values
        # # pred_samples aura shape (num_mcmc_total_samples, X_np.shape[0], output_dim)
        # # On veut (X_np.shape[0], output_dim) en moyennant sur les échantillons MCMC
        # final_predictions = np.mean(pred_samples, axis=0)

        print("Placeholder: Prédiction avec Modèle Bayésien Hiérarchique (PyMC).")
        print("  La prédiction réelle nécessite une gestion des données partagées dans le modèle PyMC.")
        # Simule une sortie de la forme (len(X), 1) pour la régression
        return np.random.rand(len(X_np), 1)


    def predict_proba(self, X, num_samples_ppc=500, **kwargs) -> np.ndarray:
        if self.pymc_model is None or self.trace is None:
            raise RuntimeError("Le modèle PyMC n'a pas été entraîné ou le tracé est manquant.")

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        # Similaire à predict(), la prédiction de "probabilités" (ou plutôt d'échantillons de la distribution a posteriori)
        # nécessite une gestion des données partagées.
        # with self.pymc_model:
        #     pm.set_data({'X_input_data': X_np})
        #     ppc = pm.sample_posterior_predictive(self.trace, var_names=["likelihood_obs"], # ou le nom de votre variable de sortie
        #                                           samples=num_samples_ppc, random_seed=kwargs.get('random_state'))
        # pred_samples = ppc.posterior_predictive["likelihood_obs"].stack(samples=("chain", "draw")).values
        # # pred_samples a shape (num_mcmc_total_samples, X_np.shape[0], output_dim)
        # # Pour predict_proba, on retourne souvent ces échantillons directement, transposés pour avoir (len(X), num_mcmc_samples) si output_dim=1
        # if pred_samples.shape[2] == 1: # Si output_dim est 1
        #     return pred_samples.squeeze(axis=2).T # (len(X), num_mcmc_total_samples)
        # else: # Pour output_dim > 1, la gestion est plus complexe
        #     return pred_samples # Retourner (num_mcmc_total_samples, len(X), output_dim) ou adapter

        print("Placeholder: Prédiction (échantillons PPC) avec Modèle Bayésien Hiérarchique (PyMC).")
        # Simule une sortie de la forme (len(X), num_samples_ppc)
        return np.random.rand(len(X_np), num_samples_ppc)

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

def _objective_optuna(trial: optuna.Trial, X_opt: pd.DataFrame, y_opt: pd.Series, model_type: str, cv_splitter: TimeSeriesSplit, scale_features_flag: bool, global_random_state: int, precomputed_class_weight_dict: Dict, socketio_instance=None, optimization_id_for_event=None, symbol_for_event=None) -> float:
    """
    Fonction objectif à optimiser par Optuna, avec émission d'événements SocketIO.
    """
    scores = []
    
    # Définir les paramètres pour le constructeur du modèle
    constructor_cv_params = {}
    # Paramètres spécifiques pour la méthode .fit()
    fit_cv_params = {} 
    callbacks_cv = [] # Callbacks pour .fit()

    if model_type == 'random_forest':
        constructor_cv_params = {
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
        constructor_cv_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.35, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.6),
            'lambda': trial.suggest_float('lambda', 1e-9, 1.0, log=True), 
            'alpha': trial.suggest_float('alpha', 1e-9, 1.0, log=True), 
            'objective': 'binary:logistic',
            'eval_metric': 'logloss', # Métrique pour l'évaluation, peut être 'auc', 'logloss', etc.
            'random_state': global_random_state,
        }
        if precomputed_class_weight_dict and len(precomputed_class_weight_dict) == 2:
            counts = y_opt.value_counts()
            if 0 in counts and 1 in counts and counts[1] > 0:
                constructor_cv_params['scale_pos_weight'] = counts[0] / counts[1]
        
        # Early stopping via constructor parameter (XGBoost v3.0.2+ approach)
        early_stopping_rounds_val = trial.suggest_int('early_stopping_rounds_cv', 10, 50)
        constructor_cv_params['early_stopping_rounds'] = early_stopping_rounds_val
        
        # Verbose parameter for fit method
        fit_cv_params['verbose'] = False # ou 0 pour pas de sortie
        
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
        
        is_scaling_needed_for_current_cv_model = current_model_scale_features if 'current_model_scale_features' in locals() else False

        if is_scaling_needed_for_current_cv_model: 
            scaler_cv = StandardScaler()
            X_train_cv_processed = pd.DataFrame(scaler_cv.fit_transform(X_train_cv), columns=X_train_cv.columns, index=X_train_cv.index)
            X_val_cv_processed = pd.DataFrame(scaler_cv.transform(X_val_cv), columns=X_val_cv.columns, index=X_val_cv.index)
        
        model_cv = model_builder(**constructor_cv_params) 
        
        current_fold_fit_params = fit_cv_params.copy() 
        if model_type == 'xgboost_classifier':
            current_fold_fit_params['eval_set'] = [(X_val_cv_processed, y_val_cv)]
            # Note: early_stopping_rounds est maintenant passé via le constructeur
            # Retirer les paramètres de callbacks de current_fold_fit_params s'ils y sont par erreur
            current_fold_fit_params.pop('callbacks', None) 
            
        model_cv.fit(X_train_cv_processed, y_train_cv, **current_fold_fit_params)
        
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
        trial.report(score, fold) # Important pour Optuna lui-même

        # Émettre une mise à jour de progression via SocketIO si disponible
        if socketio_instance and optimization_id_for_event:
            progress_data = {
                'optimization_id': optimization_id_for_event,
                'symbol': symbol_for_event,
                'trial_number': trial.number,
                'fold_number': fold + 1,
                'current_score_fold': score,
                'intermediate_mean_score': np.mean(scores) if scores else 0.0,
                'status_message': f"Optuna: Essai {trial.number}, Fold {fold+1}/{cv_splitter.get_n_splits()} pour {symbol_for_event} - Score: {score:.4f}"
            }
            # print(f"DEBUG: Emitting optimization_trial_update: {progress_data}") # Pour le débogage
            socketio_instance.emit('optimization_trial_update', progress_data)
            # time.sleep(0.01) # Petit délai pour s'assurer que le message est envoyé si beaucoup d'événements rapides

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores) if scores else 0.0


def train_model(X: pd.DataFrame, y: Optional[pd.Series], model_type: str = 'logistic_regression', model_params: Dict[str, Any] = None, model_path: str = 'models_store/model.joblib', test_size: float = 0.2, random_state: int = 42, scale_features: bool = True) -> Dict[str, Any]:
    if model_params is None:
        model_params = {}

    # X_input et y_input sont les données finales après préparation (si y initial est None)
    X_input: pd.DataFrame
    y_input: pd.Series

    if y is None:
        # Si y n'est pas fourni, X est supposé être un DataFrame complet à partir duquel X (features) et y (target) seront dérivés.
        print(f"INFO: y est None. Appel de prepare_data_for_model sur le DataFrame d'entrée (shape: {X.shape}).")
        
        required_params_for_prepare = ['feature_columns', 'target_column_name', 'problem_type']
        missing_prepare_params = [p for p in required_params_for_prepare if p not in model_params]
        if missing_prepare_params:
            raise ValueError(f"Paramètres manquants dans model_params pour prepare_data_for_model: {missing_prepare_params}")

        X_prepared, y_prepared = prepare_data_for_model(
            df=X, # X est le DataFrame complet ici
            feature_columns=model_params['feature_columns'],
            target_column=None, # MODIFIÉ: Indiquer à prepare_data_for_model de créer la colonne target
            price_change_threshold=model_params.get('price_change_threshold', 0.02),
            target_shift_days=model_params.get('target_shift_days', 1),
            problem_type=model_params['problem_type']
        )
        X_input = X_prepared
        y_input = y_prepared
        print(f"INFO: Après prepare_data_for_model, X_input shape: {X_input.shape}, y_input shape: {y_input.shape if y_input is not None else 'None'}")
    else:
        # Si y est fourni, X est supposé être déjà les features.
        X_input = X
        y_input = y

    if y_input is None or y_input.empty:
        raise ValueError("La variable cible (y_input) est vide ou None après la préparation des données.")
    if X_input is None or X_input.empty:
        raise ValueError("Les features (X_input) sont vides ou None après la préparation des données.")

    # Placeholder pour la détection d'anomalies
    # TODO: Implémenter des vérifications pour les anomalies sur X_input

    print(f"Début de l'entraînement du modèle {model_type} avec {len(X_input)} échantillons et {len(X_input.columns)} features.")

    calculated_class_weight_dict = None
    if y_input.dtype == 'int' and y_input.nunique() > 1:
        unique_classes = np.unique(y_input)
        class_weights_values = compute_class_weight('balanced', classes=unique_classes, y=y_input)
        calculated_class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_values)}
        print(f"Poids des classes (calculés sur y_input complet avant split): {calculated_class_weight_dict}")

    optimize_hyperparams = model_params.pop('optimize_hyperparameters', False)
    optuna_n_trials = model_params.pop('optuna_n_trials', 10)
    optuna_direction = model_params.pop('optuna_direction', 'maximize')
    optuna_cv_splits = model_params.pop('optuna_cv_splits', 3)
    
    best_params_from_optuna = {} 

    if optimize_hyperparams:
        print(f"Optimisation des hyperparamètres pour {model_type} avec Optuna ({optuna_n_trials} essais)...")
        # Utiliser X_input et y_input pour le split Optuna
        X_train_val, X_test_final_split, y_train_val, y_test_final_split = train_test_split(X_input, y_input, test_size=test_size, random_state=random_state, shuffle=False)
        
        optuna_class_weight_dict_for_objective = None
        # Utiliser y_train_val pour calculer les poids pour Optuna
        if y_train_val.dtype == 'int' and y_train_val.nunique() > 1:
            unique_classes_opt = np.unique(y_train_val)
            class_weights_val_opt = compute_class_weight('balanced', classes=unique_classes_opt, y=y_train_val)
            optuna_class_weight_dict_for_objective = {cls: weight for cls, weight in zip(unique_classes_opt, class_weights_val_opt)}
        # Si y_train_val n'est pas binaire/int, calculated_class_weight_dict (basé sur y_input) pourrait être utilisé s'il est pertinent
        elif calculated_class_weight_dict: # Fallback sur les poids calculés sur y_input complet
             optuna_class_weight_dict_for_objective = calculated_class_weight_dict

        tscv = TimeSeriesSplit(n_splits=optuna_cv_splits)
        study = optuna.create_study(direction=optuna_direction)
        
        # Récupérer socketio_instance et optimization_id depuis model_params si disponibles
        # Ces paramètres devront être passés à train_model depuis app_enhanced.py
        socketio_instance_for_optuna = model_params.pop('socketio_instance', None)
        optimization_id_for_optuna = model_params.pop('optimization_id', None)
        symbol_for_optuna_event = model_params.pop('symbol_for_event', None)


        study.optimize(lambda trial: _objective_optuna(
            trial, X_train_val, y_train_val, model_type, tscv,
            scale_features, random_state, optuna_class_weight_dict_for_objective,
            socketio_instance=socketio_instance_for_optuna,
            optimization_id_for_event=optimization_id_for_optuna,
            symbol_for_event=symbol_for_optuna_event
        ), n_trials=optuna_n_trials)
        
        best_params_from_optuna = study.best_params
        print(f"Meilleurs hyperparamètres trouvés par Optuna: {best_params_from_optuna}")
    # else: # Pas besoin de 'else: pass', X_train_final etc. seront définis à partir de X_input, y_input

    if optimize_hyperparams:
        # Si Optuna a été utilisé, X_train_val etc. sont déjà définis
        X_train_final = X_train_val
        y_train_final = y_train_val
        X_test_final = X_test_final_split
        y_test_final = y_test_final_split
    else:
        # Si Optuna n'est pas utilisé, faire le split sur X_input et y_input
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_input, y_input, test_size=test_size, random_state=random_state, shuffle=False)

    scaler = None
    # Utiliser X_train_final et X_test_final pour la standardisation
    X_train_scaled_for_fit = X_train_final 
    X_test_scaled_for_eval = X_test_final   

    model_requires_scaling = model_type in ['logistic_regression', 'elastic_net', 'sgd_classifier']
    if scale_features and model_requires_scaling:
        scaler = StandardScaler()
        X_train_scaled_for_fit = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns, index=X_train_final.index)
        X_test_scaled_for_eval = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns, index=X_test_final.index)
        print(f"Features standardisées pour l'entraînement final du modèle {model_type}")
    elif scale_features and not model_requires_scaling:
         print(f"Note: La standardisation globale est activée mais non appliquée pour {model_type} car généralement non requise.")

    default_params_structure = {
        'logistic_regression': {'solver': 'liblinear', 'max_iter': 1000},
        'elastic_net': {'loss': 'log_loss', 'penalty': 'elasticnet', 'max_iter': 1000, 'tol': 1e-3},
        'random_forest': {'n_jobs': -1, 'ccp_alpha': 0.0},
        'xgboost_classifier': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
        'quantile_regression': {'loss': 'quantile'},
        'bidirectional_lstm': {'activation': 'sigmoid', 'output_units': 1, 'dropout_rate': 0.2, 'l1_reg': 0.0, 'l2_reg': 0.0},
        'temporal_cnn': {'activation': 'sigmoid', 'output_units': 1, 'dropout_rate': 0.2, 'l1_reg': 0.0, 'l2_reg': 0.0},
        # S'assurer que X_train_final est défini avant d'accéder à .shape[1]
        'gaussian_process_regressor': {'num_features': X_train_final.shape[1] if X_train_final is not None and not X_train_final.empty else X_input.shape[1]},
        'hierarchical_bayesian': {'num_features': X_train_final.shape[1] if X_train_final is not None and not X_train_final.empty else X_input.shape[1]}
    }
    
    current_constructor_params = default_params_structure.get(model_type, {}).copy()
    # Appliquer d'abord les paramètres fournis par l'utilisateur qui n'ont pas été pop() pour Optuna
    # model_params contient les paramètres originaux moins ceux utilisés par la logique Optuna.
    # best_params_from_optuna contient les paramètres optimisés.
    
    # On fusionne : params par défaut < params utilisateur restants < params optimisés
    # `model_params` a déjà eu les clés d'Optuna retirées.
    # `best_params_from_optuna` contient les clés optimisées.
    
    temp_params = model_params.copy() 
    temp_params.update(best_params_from_optuna) 
    
    current_constructor_params.update(temp_params)

    final_fit_early_stopping_rounds = best_params_from_optuna.pop('early_stopping_rounds_cv', None) 
    if final_fit_early_stopping_rounds is None: 
        final_fit_early_stopping_rounds = model_params.get('early_stopping_rounds')

    if 'random_state' not in current_constructor_params:
        current_constructor_params['random_state'] = random_state
    
    # Utiliser y_train_final pour calculer les poids pour l'entraînement final
    final_class_weight_dict = None
    if y_train_final.dtype == 'int' and y_train_final.nunique() > 1:
        unique_classes_final = np.unique(y_train_final)
        class_weights_final_values = compute_class_weight('balanced', classes=unique_classes_final, y=y_train_final)
        final_class_weight_dict = {cls: weight for cls, weight in zip(unique_classes_final, class_weights_final_values)}
        print(f"Poids des classes pour l'entraînement final (sur y_train_final): {final_class_weight_dict}")
 
    is_sklearn_classifier_type = model_type in ['logistic_regression', 'elastic_net', 'random_forest', 'xgboost_classifier']
    is_nn_classifier_type = model_type in ['bidirectional_lstm', 'temporal_cnn'] and current_constructor_params.get('activation') == 'sigmoid'

    if is_sklearn_classifier_type and final_class_weight_dict:
        if model_type == 'xgboost_classifier':
            if 'scale_pos_weight' not in current_constructor_params: # Ne pas écraser si déjà optimisé
                counts_train_final = y_train_final.value_counts()
                if 0 in counts_train_final and 1 in counts_train_final and counts_train_final[1] > 0:
                    current_constructor_params['scale_pos_weight'] = counts_train_final[0] / counts_train_final[1]
        elif 'class_weight' not in current_constructor_params: # Ne pas écraser si déjà optimisé
            current_constructor_params['class_weight'] = final_class_weight_dict
            
    keys_to_remove_for_constructor = [
        'feature_groups', 'temporal_stratification_params',
        'custom_objective_params', 'temporal_weighting_params',
        'optimize_hyperparameters', 'optuna_n_trials', 'optuna_direction', 'optuna_cv_splits',
        # Retirer aussi les clés spécifiques à prepare_data_for_model si elles sont dans model_params
        'feature_columns', 'target_column_name', 'problem_type', 
        'price_change_threshold', 'target_shift_days'
    ]
    
    constructor_params = {k: v for k, v in current_constructor_params.items() if k not in keys_to_remove_for_constructor}

    if current_constructor_params.get('feature_groups') and model_type == 'elastic_net':
        print(f"Info: 'feature_groups' pour ElasticNet ({current_constructor_params.get('feature_groups')}) n'est pas utilisé activement dans cette version.")
    if current_constructor_params.get('temporal_stratification_params') and model_type == 'random_forest':
        print(f"Info: 'temporal_stratification_params' ({current_constructor_params.get('temporal_stratification_params')}) pour RandomForest non implémenté. Envisager pré-échantillonnage.")
    
    final_sample_weights = None
    if current_constructor_params.get('temporal_weighting_params') and model_type == 'xgboost_classifier':
        print(f"Info: 'temporal_weighting_params' ({current_constructor_params.get('temporal_weighting_params')}) pour XGBoost. Calcul de final_sample_weights à implémenter si besoin pour le fit final.")

    if model_type in ['bidirectional_lstm', 'temporal_cnn']:
        if 'input_shape' not in constructor_params:
            num_features_for_shape = X_train_scaled_for_fit.shape[1]
            default_timesteps = constructor_params.get('timesteps', 1)
            constructor_params['input_shape'] = (default_timesteps, num_features_for_shape)
            print(f"Avertissement/Info: 'input_shape' pour {model_type} est {constructor_params['input_shape']}. Si timesteps > 1, les données seront remodelées.")
    elif model_type == 'gaussian_process_regressor':
        if 'num_features' not in constructor_params: # Devrait être déjà là via default_params_structure
            constructor_params['num_features'] = X_train_final.shape[1] if X_train_final is not None and not X_train_final.empty else X_input.shape[1]
        if constructor_params.get('inducing_points_ratio') and 'X_train_sample_for_inducing_variable' not in constructor_params:
             # Utiliser X_train_final (qui est X_input si pas d'Optuna, ou X_train_val si Optuna)
             sample_source_for_inducing = X_train_final if X_train_final is not None and not X_train_final.empty else X_input
             constructor_params['X_train_sample_for_inducing_variable'] = sample_source_for_inducing.head(100)
    elif model_type == 'hierarchical_bayesian':
        if 'num_features' not in constructor_params: # Devrait être déjà là
            constructor_params['num_features'] = X_train_final.shape[1] if X_train_final is not None and not X_train_final.empty else X_input.shape[1]
        if 'model_specification_func' not in constructor_params:
            print(f"Avertissement: 'model_specification_func' non fournie pour HierarchicalBayesianModel. Le fit échouera.")

    print(f"Construction du modèle final {model_type} avec les paramètres constructeur: {constructor_params}")
    model: Any # Déclaration de type pour model
    if model_type == 'logistic_regression':
        model = LogisticRegression(**constructor_params)
    elif model_type == 'elastic_net':
        model = SGDClassifier(**constructor_params) # SGDClassifier est utilisé pour ElasticNet en classification
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**constructor_params)
    elif model_type == 'xgboost_classifier':
        model = xgb.XGBClassifier(**constructor_params)
    elif model_type == 'quantile_regression':
        if 'alpha' not in constructor_params: constructor_params['alpha'] = 0.5 # alpha est le quantile pour QuantileRegressor
        constructor_params.pop('class_weight', None) # Pas pertinent pour la régression
        model = GradientBoostingRegressor(**constructor_params) # GradientBoostingRegressor peut faire de la régression quantile
    elif model_type == 'bidirectional_lstm':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for BidirectionalLSTMModel but is not installed. Install with: pip install tensorflow>=2.13.0")
        model = BidirectionalLSTMModel(**constructor_params)
    elif model_type == 'temporal_cnn':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for TemporalCNNModel but is not installed. Install with: pip install tensorflow>=2.13.0")
        model = TemporalCNNModel(**constructor_params)
    elif model_type == 'gaussian_process_regressor':
        temp_inducing_sample = constructor_params.pop('X_train_sample_for_inducing_variable', None)
        model = GaussianProcessRegressionModel(**constructor_params)
        # fit_final_args sera utilisé plus bas pour passer temp_inducing_sample si besoin
    elif model_type == 'hierarchical_bayesian':
        model = HierarchicalBayesianModelPlaceholder(**constructor_params)
    else:
        raise ValueError(f"Type de modèle non supporté : {model_type}.")

    print(f"Entraînement du modèle final {model_type} sur {len(X_train_scaled_for_fit)} échantillons.")
    fit_final_args = {}
    
    # Utiliser X_train_scaled_for_fit et y_train_final pour l'entraînement
    X_fit_data = X_train_scaled_for_fit.values if isinstance(X_train_scaled_for_fit, pd.DataFrame) else X_train_scaled_for_fit
    y_fit_data = y_train_final.values if isinstance(y_train_final, pd.Series) else y_train_final
    
    # Utiliser X_test_scaled_for_eval et y_test_final pour l'évaluation (ex: early stopping)
    X_test_data_eval = X_test_scaled_for_eval.values if isinstance(X_test_scaled_for_eval, pd.DataFrame) else X_test_scaled_for_eval
    y_test_data_eval = y_test_final.values if isinstance(y_test_final, pd.Series) else y_test_final

    if model_type == 'xgboost_classifier' and final_fit_early_stopping_rounds is not None:
        # constructor_params a déjà early_stopping_rounds si optimisé. Sinon, on l'ajoute ici si fourni.
        if 'early_stopping_rounds' not in constructor_params and final_fit_early_stopping_rounds:
             # Ceci est redondant si XGBClassifier est recréé, mais gardé pour clarté si model est modifié en place.
             # En fait, le modèle est déjà créé. On ne peut pas changer les params du constructeur ici.
             # La bonne pratique est de s'assurer que constructor_params est final avant la création du modèle.
             # Pour XGBoost, early_stopping_rounds est un paramètre de fit() ou du constructeur (versions récentes).
             # On va supposer qu'il est dans constructor_params.
             pass
        if X_test_data_eval is not None and len(X_test_data_eval) > 0 and y_test_data_eval is not None and len(y_test_data_eval) > 0 :
            fit_final_args['eval_set'] = [(X_test_data_eval, y_test_data_eval)]
            # Si early_stopping_rounds n'est pas dans constructor_params, il faut le passer à fit (anciennes versions XGB)
            # Pour les versions récentes, il est préférable de le passer au constructeur.
            # On suppose que constructor_params contient déjà early_stopping_rounds si applicable.
            print(f"  Utilisation pour fit XGBoost final: eval_set sur le jeu de test.")
        else:
            print("Avertissement: eval_set pour XGBoost final non appliqué car les données de test ne sont pas prêtes ou vides.")

    elif model_type in ['bidirectional_lstm', 'temporal_cnn']:
        input_shape_for_fit = constructor_params.get('input_shape') # constructor_params est déjà nettoyé
        if input_shape_for_fit and len(input_shape_for_fit) == 2:
            timesteps, num_features_nn = input_shape_for_fit # Renommer num_features pour éviter conflit
            if timesteps > 0 and X_fit_data.shape[1] == num_features_nn:
                try:
                    X_fit_data = X_fit_data.reshape((X_fit_data.shape[0], timesteps, num_features_nn))
                    if X_test_data_eval is not None and X_test_data_eval.shape[1] == num_features_nn:
                        X_test_data_eval = X_test_data_eval.reshape((X_test_data_eval.shape[0], timesteps, num_features_nn))
                    elif X_test_data_eval is not None:
                         print(f"Avertissement: X_test_data_eval n'a pas le bon nombre de features ({X_test_data_eval.shape[1]} vs {num_features_nn}) pour le remodelage NN. Validation_data ne sera pas utilisé.")
                         X_test_data_eval = None
                except ValueError as e_reshape:
                    print(f"Erreur de remodelage pour {model_type}: {e_reshape}. Vérifier 'input_shape' et les données.")
                    return {'error': f"Erreur de remodelage des données pour {model_type}"}
            elif timesteps > 0 :
                 print(f"Avertissement: Nombre de features ({X_fit_data.shape[1]}) ne correspond pas à input_shape ({num_features_nn}) pour {model_type}. Remodelage ignoré.")

        fit_epochs = current_constructor_params.get('epochs', model_params.get('epochs', 10)) # current_constructor_params a encore tout
        fit_batch_size = current_constructor_params.get('batch_size', model_params.get('batch_size', 32))
        fit_early_stopping_patience = current_constructor_params.get('early_stopping_patience', model_params.get('early_stopping_patience'))

        fit_final_args.update({'epochs': fit_epochs, 'batch_size': fit_batch_size})
        print(f"  Utilisation pour fit NN: epochs={fit_epochs}, batch_size={fit_batch_size}")
        
        if fit_early_stopping_patience and X_test_data_eval is not None and y_test_data_eval is not None and len(X_test_data_eval) > 0 and len(y_test_data_eval) > 0:
             fit_final_args['validation_data'] = (X_test_data_eval, y_test_data_eval)
             fit_final_args['early_stopping_patience'] = fit_early_stopping_patience
             print(f"  Early stopping pour NN avec patience {fit_early_stopping_patience} sur le jeu de test final.")
        elif fit_early_stopping_patience:
            print("Avertissement: Early stopping pour NN non appliqué car les données de validation ne sont pas prêtes ou vides.")

    elif model_type == 'gaussian_process_regressor':
        fit_optimize_restarts = current_constructor_params.get('optimize_restarts', model_params.get('optimize_restarts', 5))
        fit_final_args.update({'optimize_restarts': fit_optimize_restarts})
        if y_fit_data.ndim == 1: y_fit_data = y_fit_data.reshape(-1,1)
        # Passer temp_inducing_sample à fit si besoin (déjà géré dans la création de `model` si c'était dans constructor_params)
        # La logique de `temp_inducing_sample` a été retirée de la création du modèle,
        # car le modèle GPflow est maintenant entièrement construit dans sa propre méthode `fit`.
        # Si `inducing_points_ratio` est défini, `GaussianProcessRegressionModel.fit` utilisera X_fit_data pour Z.

    elif model_type == 'hierarchical_bayesian':
        fit_draws = current_constructor_params.get('draws', model_params.get('draws', 2000))
        fit_tune = current_constructor_params.get('tune', model_params.get('tune', 1000))
        fit_chains = current_constructor_params.get('chains', model_params.get('chains', 2))
        fit_final_args.update({'draws': fit_draws, 'tune': fit_tune, 'chains': fit_chains})

    model.fit(X_fit_data, y_fit_data, sample_weight=final_sample_weights, **fit_final_args)
    
    feature_importances_dict: Optional[Dict[str, float]] = None
    if hasattr(model, 'feature_importances_'): # Pour RF, XGBoost, etc.
        importances = model.feature_importances_
        feature_importances_dict = dict(zip(X_train_final.columns, importances)) # Utiliser X_train_final pour les noms de colonnes
        print(f"Importance des features pour {model_type}: {feature_importances_dict}")
    elif model_type == 'logistic_regression' and hasattr(model, 'coef_'):
        coefs = model.coef_
        if coefs.ndim == 1:
            feature_importances_dict = dict(zip(X_train_final.columns, coefs))
        elif coefs.ndim == 2 and coefs.shape[0] == 1:
            feature_importances_dict = dict(zip(X_train_final.columns, coefs[0]))
        else:
            print(f"Coefficients pour {model_type} (shape {coefs.shape}) non directement convertis en feature_importances simples.")
        if feature_importances_dict:
             print(f"Coefficients (comme importance) pour {model_type}: {feature_importances_dict}")

    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': list(X_train_final.columns), # Colonnes utilisées pour l'entraînement final
        'model_type': model_type,
        'training_constructor_params': constructor_params,
        'training_fit_params': fit_final_args,
        'feature_importances': feature_importances_dict,
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'code_version': get_git_revision_hash(),
        'data_hash': calculate_data_hash(X_train_final, y_train_final), # Hash sur les données finales d'entraînement
        'dependencies_versions': {
            'scikit-learn': sklearn.__version__,
            'xgboost': xgboost.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'optuna': optuna.__version__,
            'tensorflow': tf.__version__ if 'tf' in globals() else 'N/A',
            'gpflow': gpflow.__version__ if 'gpflow' in globals() else 'N/A',
            'pymc': pm.__version__ if 'pm' in globals() else 'N/A',
            'arviz': az.__version__ if 'az' in globals() else 'N/A'
        }
    }
    
    # TODO: Ajouter un placeholder pour la détection de drift conceptuel après l'entraînement
    #       Par exemple, comparer les statistiques des données d'entraînement avec des données de référence
    #       ou des distributions attendues.
    #       `check_concept_drift(X_train_final, y_train_final, reference_data_stats)`

    if model_type in ['bidirectional_lstm', 'temporal_cnn'] and hasattr(model, 'model') and hasattr(model.model, 'save'): # Vérifie model.model
        print(f"Note: Pour les modèles Keras, utiliser model.model.save() est préférable. Joblib est utilisé pour la structure ici.")
    joblib.dump(model_data, model_path)

    results = {} 
    
    # Utiliser y_test_data_eval (numpy array) pour les métriques
    # S'assurer que X_test_data_eval est prêt pour la prédiction (ex: remodelé pour NN)
    # La variable X_test_data_eval a déjà été potentiellement remodelée pour les NN plus haut.
    # Si le remodelage a échoué (ex: X_test_data_eval mis à None), les prédictions/métriques ne seront pas calculées.

    if X_test_data_eval is None and model_type in ['bidirectional_lstm', 'temporal_cnn']:
        print(f"Avertissement: X_test_data_eval est None pour {model_type}, impossible de calculer les métriques de test.")
    elif y_test_data_eval is not None and len(y_test_data_eval) > 0 : # S'assurer qu'on a des données de test pour évaluer
        is_classifier_model = is_sklearn_classifier_type or \
                              (model_type in ['bidirectional_lstm', 'temporal_cnn'] and model.activation == 'sigmoid')
        
        if is_classifier_model:
            predictions_raw_test = model.predict(X_test_data_eval) # Peut être des classes ou des probas brutes pour NN
            
            probabilities_test = None
            if hasattr(model, 'predict_proba'):
                proba_output_test = model.predict_proba(X_test_data_eval)
                if isinstance(proba_output_test, tuple): # Ex: GP
                    print(f"predict_proba pour {model_type} (test) a retourné un tuple.")
                elif proba_output_test.ndim == 2 and proba_output_test.shape[1] == 2: # Classifieur binaire std
                    probabilities_test = proba_output_test[:, 1]
                elif proba_output_test.ndim == 1: # Déjà proba classe positive
                    probabilities_test = proba_output_test
                else: # Cas non standard (ex: multiclasse où predict_proba retourne (N, C))
                    probabilities_test = proba_output_test # Garder tel quel pour log_loss multiclasse
            elif model_type in ['bidirectional_lstm', 'temporal_cnn'] and model.activation == 'sigmoid' and model.output_units == 1:
                 probabilities_test = predictions_raw_test.ravel() if predictions_raw_test.ndim > 1 else predictions_raw_test

            # Conversion en classes pour accuracy, classification_report
            # y_test_data_eval est déjà un array numpy
            if probabilities_test is not None and probabilities_test.ndim == 1 and (y_test_data_eval.dtype == 'int' or pd.Series(y_test_data_eval).nunique() <=2) :
                 predictions_classes_test = (probabilities_test > 0.5).astype(int)
            elif predictions_raw_test.ndim == 2 and predictions_raw_test.shape[1] > 1 and (y_test_data_eval.dtype == 'int' or pd.Series(y_test_data_eval).nunique() > 2): # Cas multiclasse
                 predictions_classes_test = np.argmax(predictions_raw_test, axis=1)
            elif predictions_raw_test.ndim == 1 and (y_test_data_eval.dtype == 'int' or pd.Series(y_test_data_eval).nunique() <=2): # Cas binaire où predict retourne déjà des classes
                 predictions_classes_test = predictions_raw_test.astype(int)
            else: 
                 predictions_classes_test = predictions_raw_test # Fallback

            if len(y_test_data_eval) == len(predictions_classes_test):
                results['accuracy'] = accuracy_score(y_test_data_eval, predictions_classes_test)
                print(f"Modèle '{model_type}' entraîné et sauvegardé dans '{model_path}'.")
                print(f"Accuracy (test final): {results['accuracy']:.4f}")
                try:
                    results['classification_report'] = classification_report(y_test_data_eval, predictions_classes_test, output_dict=True, zero_division=0)
                    print("\nRapport de classification (test final):")
                    print(classification_report(y_test_data_eval, predictions_classes_test, zero_division=0))
                except ValueError as e_cls_report:
                     print(f"Erreur rapport de classification: {e_cls_report}")


            if probabilities_test is not None:
                try:
                    # Pour AUC, y_test_data_eval doit être binaire, probabilities_test les scores de la classe positive
                    # Pour multiclasse, probabilities_test est (N, C) et multi_class='ovr' ou 'ovo'
                    if pd.Series(y_test_data_eval).nunique() <= 2: 
                        if probabilities_test.ndim == 2 and probabilities_test.shape[1] == 2: # (N,2) -> prendre proba classe 1
                             auc_probs = probabilities_test[:,1]
                        elif probabilities_test.ndim == 1: # (N,) -> proba classe 1
                             auc_probs = probabilities_test
                        else: auc_probs = None

                        if auc_probs is not None and len(y_test_data_eval) == len(auc_probs):
                             results['auc'] = roc_auc_score(y_test_data_eval, auc_probs)
                             print(f"AUC Score (test final): {results['auc']:.4f}")
                        else: print("AUC non calculé (probas non adaptées ou problème de longueur).")

                    elif probabilities_test.ndim == 2 and probabilities_test.shape[1] > 2 and probabilities_test.shape[0] == len(y_test_data_eval): # Multiclasse
                        results['auc'] = roc_auc_score(y_test_data_eval, probabilities_test, multi_class='ovr')
                        print(f"AUC Score (test final, multiclasse OVR): {results['auc']:.4f}")
                    else:
                        print("AUC non calculé (cible non binaire ou probas non adaptées).")
                        results['auc'] = None
                except ValueError as e_auc:
                    print(f"Impossible de calculer l'AUC (test final): {e_auc}")
                    results['auc'] = None
            
        elif model_type in ['quantile_regression', 'gaussian_process_regressor'] or \
             (model_type in ['bidirectional_lstm', 'temporal_cnn', 'hierarchical_bayesian'] and model.activation != 'sigmoid'): # Modèles de régression
            
            predictions_test = model.predict(X_test_data_eval) # Moyenne pour GP, HBM
            if predictions_test.ndim > 1 and predictions_test.shape[1] == 1:
                predictions_test = predictions_test.ravel()

            if len(y_test_data_eval) == len(predictions_test):
                results['mse'] = mean_squared_error(y_test_data_eval, predictions_test)
                print(f"Modèle '{model_type}' entraîné et sauvegardé.")
                print(f"MSE (test final): {results['mse']:.4f}")
                if model_type == 'gaussian_process_regressor' and hasattr(model, 'predict_proba'):
                    _, variance_test = model.predict_proba(X_test_data_eval)
                    if variance_test is not None:
                         results['mean_variance'] = np.mean(variance_test)
                         print(f"Mean Predicted Variance (test final): {results['mean_variance']:.4f}")
            else:
                print(f"Avertissement: Longueurs de y_test_data_eval ({len(y_test_data_eval)}) et predictions_test ({len(predictions_test)}) ne correspondent pas. Métriques non calculées.")

    results['feature_importances'] = feature_importances_dict
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


    # Préparation des données pour la prédiction (conversion en numpy, remodelage)
    X_predict_data = X_scaled.values if isinstance(X_scaled, pd.DataFrame) else X_scaled
    
    model_type = model_data.get('model_type')
    training_constructor_params = model_data.get('training_constructor_params', {})

    if model_type in ['bidirectional_lstm', 'temporal_cnn']:
        input_shape_from_train = training_constructor_params.get('input_shape')
        if input_shape_from_train and len(input_shape_from_train) == 2:
            timesteps, num_features = input_shape_from_train
            if timesteps > 0 and X_predict_data.ndim == 2 and X_predict_data.shape[1] == num_features:
                try:
                    X_predict_data = X_predict_data.reshape((X_predict_data.shape[0], timesteps, num_features))
                except ValueError as reshape_error:
                    raise ValueError(f"Erreur de remodelage pour {model_type} lors de la prédiction: {reshape_error}. Attendu {num_features} features, obtenu {X_predict_data.shape[1]} avant remodelage pour {timesteps} timesteps.") from reshape_error
            elif timesteps > 0:
                 raise ValueError(f"Incohérence de features ou de dimensions pour {model_type}. Attendu {num_features} features pour {timesteps} timesteps, X_predict_data a shape {X_predict_data.shape}. Remodelage impossible.")

    # Gestion de la prédiction et de l'incertitude
    predictions: np.ndarray
    uncertainty: Optional[np.ndarray] = None

    if return_probabilities and hasattr(model, 'predict_proba'):
        proba_output = model.predict_proba(X_predict_data)
        
        if model_type == 'gaussian_process_regressor' and isinstance(proba_output, tuple) and len(proba_output) == 2:
            predictions = proba_output[0] # Moyenne
            if predictions.ndim > 1 and predictions.shape[1] == 1: predictions = predictions.ravel()
            
            raw_variance = proba_output[1] # Variance
            if raw_variance.ndim > 1 and raw_variance.shape[1] == 1: raw_variance = raw_variance.ravel()
            
            if return_uncertainty: # Variance retournée par GPR, prendre sqrt pour std dev
                uncertainty = np.sqrt(np.maximum(0, raw_variance)) # Assurer non-négatif avant sqrt
            # Si return_probabilities=True pour un régresseur GP, 'predictions' est la moyenne.
            # Si return_probabilities=False, 'predictions' est aussi la moyenne (pas de classes ici).
            # Donc, la logique de conversion en classes plus bas ne s'appliquera pas.

        elif model_type == 'hierarchical_bayesian' and proba_output.ndim == 2 and proba_output.shape[0] == len(X_predict_data):
            # HBM placeholder retourne des échantillons (N_samples_X, N_mcmc_samples)
            predictions = np.mean(proba_output, axis=1) # Moyenne des échantillons MCMC
            if return_uncertainty:
                uncertainty = np.std(proba_output, axis=1) # Écart-type des échantillons MCMC
            # Idem, la logique de conversion en classes ne s'appliquera pas.

        elif proba_output.ndim == 2 and proba_output.shape[1] == 2: # Classifieur binaire standard
            predictions = proba_output[:, 1] # Proba de la classe 1
        elif proba_output.ndim == 1: # Supposé probas classe positive déjà
            predictions = proba_output
        else: # Cas non standard (ex: multiclasse où predict_proba retourne (N, C))
            predictions = proba_output 
            # Pour multiclasse, si on veut la classe prédite, il faudrait argmax, mais ici on retourne les probas.
            # Si return_probabilities=False, la conversion en classe se fait plus bas.

    elif model_type in ['bidirectional_lstm', 'temporal_cnn'] and \
         training_constructor_params.get('activation') == 'sigmoid' and \
         training_constructor_params.get('output_units') == 1 and \
         return_probabilities:
        # Classifieur binaire NN, predict() donne les probas
        predictions = model.predict(X_predict_data)
        if predictions.ndim > 1 and predictions.shape[1] == 1 : predictions = predictions.ravel()

    else: # Prédictions directes (classes pour classifieurs, valeurs pour régresseurs)
        predictions = model.predict(X_predict_data)
        if predictions.ndim > 1 and predictions.shape[1] == 1: # S'assurer que c'est 1D pour la régression simple
            predictions = predictions.ravel()

    # Conversion en classes si return_probabilities est False pour les classifieurs
    is_classifier_outputting_probs = (model_type in ['logistic_regression', 'random_forest', 'xgboost_classifier'] or \
                                   (model_type in ['bidirectional_lstm', 'temporal_cnn'] and \
                                    training_constructor_params.get('activation') == 'sigmoid' and \
                                    training_constructor_params.get('output_units') == 1))
    
    if not return_probabilities and is_classifier_outputting_probs:
        # Si 'predictions' contient des probabilités (0 à 1), convertir en classes 0/1
        # Cela suppose une classification binaire avec seuil 0.5
        # Pour multiclasse, si 'predictions' est (N,C) de probas, il faudrait np.argmax(predictions, axis=1)
        if predictions.ndim == 1 : # Probas binaires
            predictions = (predictions > 0.5).astype(int)
        elif predictions.ndim == 2 and predictions.shape[1] > 1 : # Probas multiclasse
            predictions = np.argmax(predictions, axis=1)


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
