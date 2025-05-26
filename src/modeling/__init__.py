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

__all__ = [
    "prepare_data_for_model",
    "train_model",
    "load_model_and_predict",
    "BidirectionalLSTMModel",
    "TemporalCNNModel",
    "GaussianProcessRegressionModel",
    "HierarchicalBayesianModelPlaceholder",
    "BaseModelsManager",
    "MetaModel",
    "SignalCalibrator",
    "HierarchicalEnsemble"
]
