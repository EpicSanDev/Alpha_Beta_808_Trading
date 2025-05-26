import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calcule le Coefficient d'Information (IC) entre les valeurs réelles et prédites.
    L'IC est généralement défini comme la corrélation de Spearman entre les prédictions
    et les rendements réels.

    Args:
        y_true: Séries des valeurs réelles (par exemple, rendements futurs).
        y_pred: Séries des valeurs prédites (par exemple, signaux du modèle).

    Returns:
        float: Le coefficient de corrélation de Spearman (IC).
               Retourne np.nan si le calcul échoue (par exemple, variance nulle).
    """
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Les séries y_true et y_pred doivent avoir la même longueur.")
    if y_true.shape[0] < 2: # Spearman nécessite au moins 2 points
        return np.nan

    # Supprimer les NaNs conjointement pour s'assurer que les paires sont valides
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
    
    if df.shape[0] < 2: # Pas assez de données après suppression des NaNs
        return np.nan
        
    # Vérifier la variance pour éviter les erreurs dans spearmanr
    if df['true'].nunique() <= 1 or df['pred'].nunique() <= 1:
        # Si l'une des séries a une variance nulle (toutes les valeurs sont identiques après dropna),
        # la corrélation de Spearman n'est pas bien définie ou sera 0 ou NaN.
        # On peut retourner 0 ou NaN. NaN est plus informatif d'un problème.
        return np.nan

    try:
        ic_value, _ = spearmanr(df['true'], df['pred'])
        return ic_value
    except Exception: # Gérer toute autre exception potentielle de spearmanr
        return np.nan

def predictive_sharpe_ratio(predictions: pd.Series, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calcule le ratio de Sharpe prédictif basé sur les signaux du modèle.
    Simule les rendements d'une stratégie simple : long si signal > 0, short si signal < 0.
    Les rendements sont annualisés pour le calcul du ratio de Sharpe.

    Args:
        predictions: Séries des signaux de trading prédits par le modèle.
                     Les valeurs positives indiquent un signal d'achat/long,
                     les valeurs négatives un signal de vente/short.
        returns: Séries des rendements réels correspondants aux périodes des prédictions.
                 Doit être aligné avec `predictions`.
        risk_free_rate: Taux de rendement sans risque annualisé (par défaut 0.0).

    Returns:
        float: Ratio de Sharpe prédictif annualisé.
               Retourne np.nan si le calcul est impossible (par exemple, pas de trades).
    """
    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions)
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    if predictions.shape[0] != returns.shape[0]:
        raise ValueError("Les séries predictions et returns doivent avoir la même longueur.")
    
    # Aligner les index et supprimer les NaNs
    df = pd.DataFrame({'predictions': predictions, 'returns': returns}).dropna()
    
    if df.empty:
        return np.nan

    # Simuler les positions basées sur les signaux
    # Position = 1 pour long, -1 pour short, 0 pour neutre (si signal est 0)
    positions = np.sign(df['predictions'])
    
    # Calculer les rendements de la stratégie
    strategy_returns = positions * df['returns']
    
    if strategy_returns.empty or strategy_returns.std() == 0:
        # Pas de trades ou pas de variance dans les rendements de la stratégie
        return np.nan

    # Calculer le ratio de Sharpe
    # Supposons que les rendements sont journaliers, annualisation par sqrt(252)
    # Si les rendements sont d'une autre fréquence, ajuster le facteur d'annualisation.
    # Pour l'instant, on suppose que la fréquence est implicite dans `returns`.
    # L'utilisateur doit fournir des `returns` à la fréquence souhaitée pour l'analyse.
    # Le `risk_free_rate` doit correspondre à la période des `returns`.
    # Si `returns` sont des rendements journaliers, `risk_free_rate` doit être le taux journalier.
    # Pour simplifier, on suppose que `returns` sont des rendements périodiques (ex: journaliers)
    # et on annualise le ratio de Sharpe.
    
    excess_returns = strategy_returns - (risk_free_rate / 252) # Exemple de taux journalier si RF annuel
    
    # Vérifier si excess_returns est vide ou a une std nulle après soustraction du taux sans risque
    if excess_returns.empty or excess_returns.std() == 0:
        return np.nan
        
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Annualisation (si les rendements sont journaliers et 252 jours de trading par an)
    # Ce facteur d'annualisation dépend de la fréquence des `returns`.
    # Si les `returns` sont déjà à la fréquence désirée pour le Sharpe (ex: mensuels),
    # alors pas besoin d'annualiser ici.
    # Pour un Sharpe prédictif, on s'intéresse souvent à la performance par période de prédiction.
    # Si on veut un Sharpe annualisé à partir de rendements journaliers:
    annualization_factor = np.sqrt(252) # Supposant des données journalières
    
    return sharpe * annualization_factor


def get_calibration_metrics(y_true: np.ndarray, proba_pred: np.ndarray, n_bins: int = 10) -> dict:
    """
    Calcule les métriques de calibration des probabilités.

    Args:
        y_true: Valeurs réelles binaires (0 ou 1).
        proba_pred: Probabilités prédites pour la classe positive.
        n_bins: Nombre de bins pour la courbe de calibration.

    Returns:
        dict: Un dictionnaire contenant:
            - 'brier_score': Le score de Brier.
            - 'fraction_of_positives': Fraction réelle de positifs dans chaque bin.
            - 'mean_predicted_value': Probabilité moyenne prédite dans chaque bin.
    """
    if len(y_true) == 0 or len(proba_pred) == 0:
        return {
            'brier_score': np.nan,
            'fraction_of_positives': np.array([]),
            'mean_predicted_value': np.array([])
        }

    # S'assurer que y_true est binaire (0 ou 1)
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true doit contenir des valeurs binaires (0 ou 1) pour la calibration.")

    brier = brier_score_loss(y_true, proba_pred)
    
    # calibration_curve peut échouer si proba_pred n'a pas assez de valeurs uniques
    # ou si toutes les valeurs sont identiques.
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, proba_pred, n_bins=n_bins, strategy='uniform' # ou 'quantile'
        )
    except ValueError: # Peut arriver si proba_pred est constant ou a peu de valeurs uniques
        fraction_of_positives = np.array([np.nan] * n_bins)
        mean_predicted_value = np.array([np.nan] * n_bins)
        # Si y_true a des valeurs, on peut calculer la fraction de positifs globale
        if len(y_true) > 0:
            global_fraction = np.mean(y_true)
            fraction_of_positives = np.array([global_fraction if i == 0 else np.nan for i in range(n_bins)])
            # Pour mean_predicted_value, si proba_pred est constant, on peut utiliser cette valeur
            if len(proba_pred) > 0 and pd.Series(proba_pred).nunique() == 1:
                 mean_predicted_value = np.array([proba_pred[0] if i == 0 else np.nan for i in range(n_bins)])


    return {
        'brier_score': brier,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }

# Exemple d'utilisation (peut être mis dans des tests unitaires)
if __name__ == '__main__':
    # Exemple pour information_coefficient
    y_true_ic = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
    y_pred_ic = pd.Series([0.5, -0.3, 0.6, 0.1, -0.2])
    ic = information_coefficient(y_true_ic, y_pred_ic)
    print(f"Information Coefficient (IC): {ic:.4f}")

    # Exemple pour predictive_sharpe_ratio
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    predictions_sr = pd.Series([1, -1, 1, 1, -1], index=dates) # Signaux: 1 pour long, -1 pour short
    returns_sr = pd.Series([0.01, 0.005, -0.002, 0.015, -0.003], index=dates) # Rendements journaliers
    sharpe = predictive_sharpe_ratio(predictions_sr, returns_sr)
    print(f"Predictive Sharpe Ratio (annualized): {sharpe:.4f}")

    # Exemple pour get_calibration_metrics
    y_true_calib = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 10) # 100 observations
    # Probabilités bien calibrées
    proba_pred_calib_good = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.05] * 10)
    np.random.shuffle(proba_pred_calib_good) # Mélanger pour simuler des prédictions réelles
    
    # Probabilités mal calibrées (toujours confiant)
    proba_pred_calib_bad = np.clip(np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9, 0.99, 0.99] * 10) + np.random.normal(0, 0.05, 100), 0, 1)

    calib_metrics_good = get_calibration_metrics(y_true_calib, proba_pred_calib_good)
    print(f"\nCalibration Metrics (Good):")
    print(f"  Brier Score: {calib_metrics_good['brier_score']:.4f}")
    # print(f"  Fraction of Positives: {calib_metrics_good['fraction_of_positives']}")
    # print(f"  Mean Predicted Value: {calib_metrics_good['mean_predicted_value']}")

    calib_metrics_bad = get_calibration_metrics(y_true_calib, proba_pred_calib_bad)
    print(f"\nCalibration Metrics (Bad):")
    print(f"  Brier Score: {calib_metrics_bad['brier_score']:.4f}")
    # print(f"  Fraction of Positives: {calib_metrics_bad['fraction_of_positives']}")
    # print(f"  Mean Predicted Value: {calib_metrics_bad['mean_predicted_value']}")

    # Cas avec variance nulle pour IC
    y_true_ic_zero_var = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    ic_zero_var = information_coefficient(y_true_ic_zero_var, y_pred_ic)
    print(f"IC with zero variance in y_true: {ic_zero_var}") # Devrait être nan

    y_pred_ic_zero_var = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
    ic_zero_var_pred = information_coefficient(y_true_ic, y_pred_ic_zero_var)
    print(f"IC with zero variance in y_pred: {ic_zero_var_pred}") # Devrait être nan

    # Cas avec peu de données pour IC
    y_true_ic_short = pd.Series([0.01])
    y_pred_ic_short = pd.Series([0.5])
    ic_short = information_coefficient(y_true_ic_short, y_pred_ic_short)
    print(f"IC with 1 data point: {ic_short}") # Devrait être nan

    # Cas où Sharpe ne peut être calculé
    predictions_sr_no_trade = pd.Series([0, 0, 0, 0, 0], index=dates)
    sharpe_no_trade = predictive_sharpe_ratio(predictions_sr_no_trade, returns_sr)
    print(f"Predictive Sharpe Ratio (no trades): {sharpe_no_trade}") # Devrait être nan

    returns_sr_zero_std = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=dates)
    predictions_sr_const_signal = pd.Series([1, 1, 1, 1, 1], index=dates)
    sharpe_zero_std_returns = predictive_sharpe_ratio(predictions_sr_const_signal, returns_sr_zero_std)
    print(f"Predictive Sharpe Ratio (zero std strategy returns): {sharpe_zero_std_returns}") # Devrait être nan