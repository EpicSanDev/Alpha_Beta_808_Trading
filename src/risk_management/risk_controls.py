from typing import Union

def check_position_limit(
    proposed_position_size: Union[int, float],
    total_capital: Union[int, float],
    max_position_percentage: float = 0.05
) -> bool:
    """
    Vérifie si la taille d'une position proposée dépasse un pourcentage maximum du capital total.

    Args:
        proposed_position_size: La taille de la position proposée.
        total_capital: Le capital total disponible.
        max_position_percentage: Le pourcentage maximum du capital total autorisé pour une seule position.
                                 Par défaut à 0.05 (5%).

    Returns:
        True si la position est dans la limite (taille <= total_capital * max_position_percentage),
        False sinon.
    """
    if total_capital <= 0:
        # Ne peut pas prendre de position si le capital est nul ou négatif
        return False
    
    max_allowed_size = total_capital * max_position_percentage
    return proposed_position_size <= max_allowed_size

def check_stop_loss(
    current_price: Union[int, float],
    entry_price: Union[int, float],
    stop_loss_percentage: float = 0.02,
    position_type: str = 'long'
) -> bool:
    """
    Vérifie si un stop-loss de base est atteint.

    Args:
        current_price: Le prix actuel de l'actif.
        entry_price: Le prix d'entrée de la position.
        stop_loss_percentage: Le pourcentage de perte acceptable avant de déclencher le stop-loss.
                              Par défaut à 0.02 (2%).
        position_type: Le type de position, 'long' ou 'short'. Par défaut à 'long'.

    Returns:
        True si le stop-loss est atteint, False sinon.
        Pour une position 'long', le stop-loss est atteint si current_price <= entry_price * (1 - stop_loss_percentage).
        Pour une position 'short', le stop-loss est atteint si current_price >= entry_price * (1 + stop_loss_percentage).
    """
    if position_type not in ['long', 'short']:
        raise ValueError("position_type doit être 'long' ou 'short'")

    if position_type == 'long':
        stop_price = entry_price * (1 - stop_loss_percentage)
        return current_price <= stop_price
    else:  # position_type == 'short'
        stop_price = entry_price * (1 + stop_loss_percentage)
        return current_price >= stop_price

def check_position_limit(
    new_position_size: Union[int, float],
    current_total_exposure: Union[int, float],
    max_exposure_limit: Union[int, float]
) -> bool:
    """
    Vérifie si l'ajout d'une nouvelle position dépasse la limite d'exposition totale.

    Args:
        new_position_size: La taille de la nouvelle position proposée.
        current_total_exposure: L'exposition totale actuelle.
        max_exposure_limit: La limite maximale d'exposition autorisée.

    Returns:
        True si la position peut être ajoutée sans dépasser la limite d'exposition,
        False sinon.
    """
    projected_exposure = current_total_exposure + new_position_size
    return projected_exposure <= max_exposure_limit