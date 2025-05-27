from typing import Union, Tuple

def check_single_trade_notional_vs_capital(
    trade_notional_value: float,
    total_capital: float,
    max_notional_as_pct_of_capital: float = 0.05
) -> bool:
    """
    Vérifie si la valeur notionnelle d'un trade proposé dépasse un pourcentage maximum du capital total.
    Utilisé pour limiter le risque sur un seul trade basé sur le capital.

    Args:
        trade_notional_value (float): La valeur notionnelle du trade proposé (taille * prix * multiplicateur_contrat).
        total_capital (float): Le capital total disponible (valeur du portefeuille).
        max_notional_as_pct_of_capital (float): Le pourcentage maximum du capital total autorisé
                                                pour la valeur notionnelle d'un seul trade.
                                                Par défaut à 0.05 (5%).

    Returns:
        bool: True si le trade est dans la limite, False sinon.
    """
    if total_capital <= 0:
        # Ne peut pas prendre de position si le capital est nul ou négatif
        return False
    
    max_allowed_notional_for_trade = total_capital * max_notional_as_pct_of_capital
    return trade_notional_value <= max_allowed_notional_for_trade

def check_stop_loss(
    current_price: float,
    entry_price: float,
    stop_loss_percentage: float = 0.02,
    position_type: str = 'long',
    contract_multiplier: float = 1.0, # Ajout pour futures
    leverage: float = 1.0 # Ajout pour futures
) -> bool:
    """
    Vérifie si un stop-loss de base est atteint, en considérant l'impact du levier sur le P&L.
    Note: Ce stop-loss est basé sur le pourcentage de perte par rapport au prix d'entrée,
    pas directement sur la marge. Pour des stops basés sur la marge, voir dynamic_stops.py.

    Args:
        current_price (float): Le prix actuel de l'actif.
        entry_price (float): Le prix d'entrée de la position.
        stop_loss_percentage (float): Le pourcentage de perte acceptable sur la valeur notionnelle
                                     avant de déclencher le stop-loss. Par défaut à 0.02 (2%).
        position_type (str): Le type de position, 'long' ou 'short'. Par défaut à 'long'.
        contract_multiplier (float): Multiplicateur de contrat pour les futures. Par défaut à 1.0.
        leverage (float): Levier utilisé pour la position. Par défaut à 1.0.

    Returns:
        bool: True si le stop-loss est atteint, False sinon.
    """
    if position_type not in ['long', 'short']:
        raise ValueError("position_type doit être 'long' ou 'short'")

    # Le stop_loss_percentage s'applique à la variation de prix de l'actif sous-jacent.
    # L'effet de levier amplifie le P&L, mais le seuil de stop-loss est défini sur le prix.
    if position_type == 'long':
        stop_price = entry_price * (1 - stop_loss_percentage)
        return current_price <= stop_price
    else:  # position_type == 'short'
        stop_price = entry_price * (1 + stop_loss_percentage)
        return current_price >= stop_price

# La fonction check_position_limit (lignes 59-77 du fichier original) est supprimée
# car sa logique est mieux couverte par check_max_total_notional_exposure
# et check_single_trade_notional_vs_capital.

def check_total_leverage_limit(
    current_portfolio_value: float,
    total_notional_exposure: float,
    max_portfolio_leverage: float
) -> Tuple[bool, str]:
    """
    Vérifie si le levier global du portefeuille dépasse une limite maximale.

    Args:
        current_portfolio_value (float): La valeur actuelle du portefeuille (équité).
        total_notional_exposure (float): La somme des valeurs notionnelles de toutes les positions ouvertes.
        max_portfolio_leverage (float): Le levier maximum autorisé pour le portefeuille.

    Returns:
        Tuple[bool, str]: (True si le levier est dans la limite, raison si False, "" si True)
    """
    if current_portfolio_value <= 0:
        if total_notional_exposure > 0:
            return False, "Portfolio equity is zero or negative with active notional exposure."
        return True, "" # Pas de levier si pas d'exposition et pas de capital

    effective_leverage = total_notional_exposure / current_portfolio_value
    if effective_leverage > max_portfolio_leverage:
        return False, f"Effective leverage {effective_leverage:.2f}x exceeds limit {max_portfolio_leverage:.2f}x."
    return True, ""

def check_max_notional_exposure_per_asset(
    asset_notional_value: float,
    max_notional_per_asset: float
) -> Tuple[bool, str]:
    """
    Vérifie si la valeur notionnelle d'une position sur un actif donné dépasse une limite.

    Args:
        asset_notional_value (float): La valeur notionnelle de la position sur l'actif.
        max_notional_per_asset (float): La valeur notionnelle maximale autorisée pour cet actif.

    Returns:
        Tuple[bool, str]: (True si l'exposition est dans la limite, raison si False, "" si True)
    """
    if asset_notional_value > max_notional_per_asset:
        return False, f"Asset notional value {asset_notional_value:.2f} exceeds limit {max_notional_per_asset:.2f}."
    return True, ""

def check_max_total_notional_exposure(
    total_notional_exposure: float,
    max_allowed_total_notional: float
) -> Tuple[bool, str]:
    """
    Vérifie si l'exposition notionnelle totale du portefeuille dépasse une limite absolue.

    Args:
        total_notional_exposure (float): L'exposition notionnelle totale actuelle du portefeuille.
        max_allowed_total_notional (float): La limite maximale d'exposition notionnelle totale autorisée.

    Returns:
        Tuple[bool, str]: (True si dans la limite, raison si False, "" si True)
    """
    if total_notional_exposure > max_allowed_total_notional:
        return False, f"Total notional exposure {total_notional_exposure:.2f} exceeds limit {max_allowed_total_notional:.2f}."
    return True, ""

def check_max_leverage_per_trade(
    trade_leverage: float,
    max_leverage_allowed: float
) -> Tuple[bool, str]:
    """
    Vérifie si le levier demandé pour un trade spécifique dépasse la limite autorisée.

    Args:
        trade_leverage (float): Le levier demandé pour le trade.
        max_leverage_allowed (float): Le levier maximum autorisé (peut être par actif ou global).

    Returns:
        Tuple[bool, str]: (True si le levier du trade est acceptable, raison si False, "" si True)
    """
    if trade_leverage <= 0:
        return False, f"Trade leverage {trade_leverage:.2f}x must be positive."
    if trade_leverage > max_leverage_allowed:
        return False, f"Trade leverage {trade_leverage:.2f}x exceeds allowed limit {max_leverage_allowed:.2f}x."
    return True, ""

def check_margin_usage_limit(
    current_used_margin: float,
    portfolio_equity: float,
    max_margin_usage_pct_limit: float
) -> Tuple[bool, str]:
    """
    Vérifie si le pourcentage de la marge utilisée par rapport à l'équité du portefeuille
    dépasse une limite configurée.

    Args:
        current_used_margin (float): La marge actuellement utilisée par les positions ouvertes.
        portfolio_equity (float): L'équité actuelle du portefeuille.
        max_margin_usage_pct_limit (float): Le pourcentage maximum de l'équité du portefeuille
                                           qui peut être utilisé comme marge (ex: 0.5 pour 50%).

    Returns:
        Tuple[bool, str]: (True si l'utilisation de la marge est dans les limites, raison si False, "" si True)
    """
    if portfolio_equity <= 0:
        if current_used_margin > 0:
             return False, "Cannot use margin with zero or negative portfolio equity."
        return True, "" # Pas d'utilisation de marge si équité nulle et marge nulle

    margin_usage_pct = current_used_margin / portfolio_equity
    
    if margin_usage_pct > max_margin_usage_pct_limit:
        return False, f"Margin usage {margin_usage_pct*100:.2f}% exceeds limit {max_margin_usage_pct_limit*100:.2f}%."
    return True, ""