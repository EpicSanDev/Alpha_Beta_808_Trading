# src/modeling/__init__.py

# Rendre les fonctions et classes importantes accessibles directement depuis le package modeling
from .models import (
    prepare_data_for_model,
    train_model,
    load_model_and_predict,
    BidirectionalLSTMModel,
    TemporalCNNModel,
    GaussianProcessRegressionModel,
    HierarchicalBayesianModelPlaceholder
)
from .ensemble import (
    BaseModelsManager,
    MetaModel,
    SignalCalibrator,
    HierarchicalEnsemble
)
from .augmentation import (
    ajouter_bruit_calibre,
    bootstrap_temporel_blocs
)
from .xgboost_callbacks import (
    CustomMetricLogger,
    CustomEarlyStopping,
    AdvancedXGBoostModel
)
from .xgboost_model import (
    XGBoostCustomCallback, # Note: Peut y avoir redondance de nom avec celui dans xgboost_callbacks si les deux sont utilisés au même niveau
    XGBoostModel
)


__all__ = [
    # from models.py
    "prepare_data_for_model",
    "train_model",
    "load_model_and_predict",
    "BidirectionalLSTMModel",
    "TemporalCNNModel",
    "GaussianProcessRegressionModel",
    "HierarchicalBayesianModelPlaceholder",
    # from ensemble.py
    "BaseModelsManager",
    "MetaModel",
    "SignalCalibrator",
    "HierarchicalEnsemble",
    # from augmentation.py
    "ajouter_bruit_calibre",
    "bootstrap_temporel_blocs",
    # from xgboost_callbacks.py
    "CustomMetricLogger",
    "CustomEarlyStopping",
    "AdvancedXGBoostModel",
    # from xgboost_model.py
    "XGBoostCustomCallback", # Attention si XGBoostCustomCallback de xgboost_callbacks est aussi exposé et utilisé
    "XGBoostModel"
]
# Il est à noter que XGBoostCustomCallback est défini dans xgboost_model.py
# et CustomMetricLogger/CustomEarlyStopping sont dans xgboost_callbacks.py.
# AdvancedXGBoostModel est dans xgboost_callbacks.py et XGBoostModel est dans xgboost_model.py.
# S'il y a une intention d'utiliser une seule version de callback ou de modèle XGBoost wrapper,
# il faudrait choisir lequel exposer ou renommer pour éviter la confusion.
# Pour l'instant, j'expose les deux comme demandé par l'analyse des fichiers.
