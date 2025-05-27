# src/signal_generation/signal_generator.py
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from dataclasses import dataclass, field

@dataclass
class TradingSignal:
    """
    Représente un signal de trading généré.
    """
    timestamp: pd.Timestamp
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: Optional[float] = None  # Nombre de contrats ou montant
    price: Optional[float] = None  # Prix d'exécution suggéré/limite
    target_leverage: Optional[float] = None
    signal_source: Optional[str] = None # Nom du modèle ou de la stratégie
    confidence: Optional[float] = None # Confiance du signal (ex: 0.0 à 1.0)
    # Potentiellement d'autres champs: stop_loss, take_profit, order_type, etc.

def generate_base_signals_from_predictions(
    predictions: Union[pd.Series, np.ndarray, List[float]],
    symbol: str,
    threshold: float = 0.5,
    prediction_type: str = 'probability' # 'probability' or 'class'
) -> List[TradingSignal]:
    """
    Génère une liste de TradingSignal de base (action, symbole, timestamp)
    à partir des prédictions d'un modèle.
    Le dimensionnement de la position et le levier sont déterminés ultérieurement.

    Args:
        predictions (Union[pd.Series, np.ndarray, List[float]]):
            Une série, un array NumPy ou une liste de prédictions.
            L'index de la série (si pd.Series) est utilisé pour les timestamps.
        symbol (str): Le symbole de l'actif concerné.
        threshold (float, optional):
            Seuil pour convertir les probabilités en signaux.
            Ignoré si prediction_type est 'class'. Par défaut à 0.5.
        prediction_type (str, optional):
            Type de prédictions. Peut être 'probability' ou 'class'.
            Par défaut à 'probability'.

    Returns:
        List[TradingSignal]: Une liste d'objets TradingSignal.
    """
    if not isinstance(predictions, pd.Series):
        # Si ce n'est pas une Series, on suppose que l'index est un simple range.
        # Idéalement, les timestamps devraient être fournis.
        predictions_series = pd.Series(predictions, index=pd.to_datetime(pd.RangeIndex(start=0, stop=len(predictions))))
    else:
        predictions_series = predictions

    if not isinstance(predictions_series.index, pd.DatetimeIndex):
        raise ValueError("L'index des prédictions doit être un DatetimeIndex.")

    action_signals = pd.Series(['HOLD'] * len(predictions_series), index=predictions_series.index)

    if prediction_type == 'probability':
        action_signals[predictions_series > threshold] = 'BUY'
        action_signals[predictions_series < (1 - threshold)] = 'SELL' # Seuil symétrique pour SELL
    elif prediction_type == 'class':
        action_signals[predictions_series == 1] = 'BUY' # Supposant 1 = BUY
        action_signals[predictions_series == 0] = 'SELL' # Supposant 0 = SELL
    else:
        raise ValueError("prediction_type doit être 'probability' ou 'class'")

    trading_signals: List[TradingSignal] = []
    for timestamp, action in action_signals.items():
        # La confiance pourrait être la probabilité elle-même si disponible
        confidence_value = None
        if prediction_type == 'probability':
            # Utiliser la distance par rapport à 0.5 comme mesure de confiance simple
            # ou la probabilité brute si elle est directionnelle (ex: proba de hausse pour BUY)
            raw_pred = predictions_series.loc[timestamp]
            if action == 'BUY':
                confidence_value = float(raw_pred)
            elif action == 'SELL':
                confidence_value = float(1 - raw_pred) # ou raw_pred si c'est proba de baisse

        trading_signals.append(
            TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                confidence=confidence_value
                # quantity, price, target_leverage seront définis plus tard
            )
        )
    return trading_signals

def determine_position_size_and_leverage(
    base_signals: List[TradingSignal],
    total_capital: float,
    current_prices: dict, # Ex: {"BTCUSDT": 20000, "ETHUSDT": 1500}
    contract_multipliers: dict, # Ex: {"BTCUSDT": 0.001, "ETHUSDT": 0.01}
    fraction_per_trade: float = 0.02, # Fraction du capital à risquer/allouer par trade
    default_target_leverage: float = 1.0,
    max_leverage_cap: float = 20.0, # Plafond de levier global
    # risk_manager_limits: Optional[dict] = None # Pourrait contenir des limites du RiskManager
    # market_data_for_sizing: Optional[pd.DataFrame] = None # Pourrait contenir OI, funding rates etc.
) -> List[TradingSignal]:
    """
    Détermine la taille de la position (nombre de contrats) et le levier cible
    pour une liste de TradingSignal de base.

    Args:
        base_signals (List[TradingSignal]): Liste des signaux de base (action, symbole, timestamp).
        total_capital (float): Capital total disponible.
        current_prices (dict): Dictionnaire des prix actuels des actifs.
                               Clé: symbole, Valeur: prix.
        contract_multipliers (dict): Dictionnaire des multiplicateurs de contrat.
                                     Clé: symbole, Valeur: multiplicateur.
        fraction_per_trade (float, optional): Fraction du capital à allouer par trade.
                                              Par défaut à 0.02 (2%).
        default_target_leverage (float, optional): Levier cible par défaut si non spécifié
                                                   par le signal ou la logique. Par défaut à 1.0.
        max_leverage_cap (float, optional): Levier maximum autorisé. Par défaut à 20.0.
        # risk_manager_limits: Pourrait être utilisé pour vérifier les limites d'exposition.
        # market_data_for_sizing: Données supplémentaires (OI, funding) pour affiner le dimensionnement.

    Returns:
        List[TradingSignal]: Liste des TradingSignal mis à jour avec quantity et target_leverage.
    """
    sized_signals: List[TradingSignal] = []

    for signal in base_signals:
        # Copie pour ne pas modifier l'objet original dans la liste base_signals
        # si base_signals est réutilisé ailleurs.
        # Alternativement, modifier signal directement si c'est l'intention.
        # Pour l'instant, modifions directement.
        # updated_signal = dataclasses.replace(signal)

        if signal.action == 'HOLD':
            signal.quantity = 0
            signal.target_leverage = 0 # Pas de levier pour HOLD
            sized_signals.append(signal)
            continue

        current_price = current_prices.get(signal.symbol)
        contract_multiplier = contract_multipliers.get(signal.symbol)

        if current_price is None or contract_multiplier is None:
            print(f"Avertissement: Prix actuel ({current_price}) ou multiplicateur de contrat ({contract_multiplier}) manquant pour {signal.symbol}. Signal ignoré pour le dimensionnement.")
            signal.quantity = 0
            signal.target_leverage = 0
            signal.action = 'HOLD' # Si données manquantes, on ne trade pas
            sized_signals.append(signal)
            continue
        
        # 1. Déterminer le levier cible
        target_leverage = signal.target_leverage if signal.target_leverage is not None else default_target_leverage
        
        # Logique pour ajuster le levier en fonction de la confiance (exemple)
        if signal.confidence is not None:
            # Exemple simple: levier proportionnel à la confiance, mais plafonné
            # Supposons que la confiance est entre 0 et 1.
            # Si confiance < 0.6, levier = default_target_leverage
            # Si confiance >= 0.6, levier = default_target_leverage + (confiance - 0.6) * 10
            # Ceci est un exemple, la logique réelle peut être plus complexe.
            if signal.confidence >= 0.6: # Seuil de confiance pour augmenter le levier
                 suggested_leverage = default_target_leverage + (signal.confidence - 0.5) * 10 # Pente d'ajustement
                 target_leverage = max(default_target_leverage, min(suggested_leverage, max_leverage_cap))
            # else: target_leverage reste default_target_leverage ou celui du signal

        target_leverage = min(target_leverage, max_leverage_cap) # Plafonner au max autorisé
        target_leverage = max(1.0, target_leverage) # Levier minimum de 1


        # 2. Calculer le capital réel alloué pour ce trade
        # On peut moduler fraction_per_trade en fonction de signal.confidence
        # Par exemple: allocation_fraction = fraction_per_trade * (signal.confidence or 0.7) / 0.7
        # Pour l'instant, utilisons une fraction fixe.
        actual_capital_for_trade = total_capital * fraction_per_trade
        
        # 3. Calculer la valeur notionnelle de la position souhaitée
        notional_value_target = actual_capital_for_trade * target_leverage

        # 4. Calculer la taille de la position en nombre de contrats
        if current_price <= 0 or contract_multiplier <= 0:
            print(f"Avertissement: Prix ({current_price}) ou multiplicateur ({contract_multiplier}) invalide pour {signal.symbol}. Signal ignoré.")
            signal.quantity = 0
            signal.target_leverage = 0
            signal.action = 'HOLD'
            sized_signals.append(signal)
            continue
            
        num_contracts = notional_value_target / (current_price * contract_multiplier)
        
        signal.quantity = round(num_contracts)
        signal.target_leverage = target_leverage
        signal.price = current_price # Enregistrer le prix utilisé pour le calcul de la taille

        # Interaction avec RiskManager (conceptuel):
        # if risk_manager_limits:
        #     if not RiskManager.is_trade_allowed(signal, total_capital, risk_manager_limits):
        #         signal.quantity = 0 # Ou ajuster la quantité
        
        if signal.quantity > 0:
            sized_signals.append(signal)
        else:
            signal.quantity = 0
            signal.target_leverage = 0
            signal.action = 'HOLD'
            sized_signals.append(signal)
            
    return sized_signals

class AbstractSignalGenerator:
    """
    Classe abstraite pour les générateurs de signaux.
    Les classes filles doivent implémenter la méthode generate_signals.
    """
    def __init__(self, signal_source_name: str = "AbstractSignalGenerator"):
        self.signal_source_name = signal_source_name

    def generate_signals(self, predictions: Union[pd.Series, np.ndarray, List[float]],
                         symbol: str,
                         total_capital: float,
                         current_prices: dict,
                         contract_multipliers: dict,
                         **kwargs) -> List[TradingSignal]:
        """
        Méthode principale pour générer des signaux de trading complets.
        Doit être implémentée par les classes filles.
        """
        raise NotImplementedError("La méthode generate_signals doit être implémentée par la classe fille.")

    def _add_source_to_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        for signal in signals:
            if signal.signal_source is None:
                signal.signal_source = self.signal_source_name
        return signals


class BasicSignalGenerator(AbstractSignalGenerator):
    """
    Un générateur de signaux simple qui utilise les fonctions de base
    pour la génération d'actions et le dimensionnement des positions/levier.
    """
    def __init__(self,
                 threshold: float = 0.5,
                 prediction_type: str = 'probability',
                 fraction_per_trade: float = 0.02,
                 default_target_leverage: float = 1.0,
                 max_leverage_cap: float = 20.0,
                 signal_source_name: str = "BasicSignalGenerator"):
        super().__init__(signal_source_name)
        self.threshold = threshold
        self.prediction_type = prediction_type
        self.fraction_per_trade = fraction_per_trade
        self.default_target_leverage = default_target_leverage
        self.max_leverage_cap = max_leverage_cap

    def generate_signals(self,
                         predictions: Union[pd.Series, np.ndarray, List[float]],
                         symbol: str,
                         total_capital: float,
                         current_prices: dict, # Ex: {"BTCUSDT": 60000}
                         contract_multipliers: dict, # Ex: {"BTCUSDT": 0.001}
                         # Les kwargs peuvent être utilisés pour passer des paramètres spécifiques
                         # aux fonctions sous-jacentes si nécessaire, ou pour des configurations
                         # plus avancées non gérées par les attributs de l'instance.
                         **kwargs) -> List[TradingSignal]:
        """
        Génère des signaux de trading complets en utilisant les prédictions du modèle.

        Args:
            predictions: Prédictions du modèle.
            symbol: Symbole de l'actif.
            total_capital: Capital total.
            current_prices: Prix actuels des actifs.
            contract_multipliers: Multiplicateurs de contrat.
            **kwargs: Peut inclure 'prediction_type', 'threshold' pour surcharger
                      les valeurs de l'instance pour cet appel spécifique.
                      Peut aussi inclure des paramètres pour `determine_position_size_and_leverage`
                      comme `default_target_leverage`, `max_leverage_cap`, `fraction_per_trade`.

        Returns:
            List[TradingSignal]: Liste des signaux de trading dimensionnés.
        """
        # Utiliser les kwargs pour potentiellement surcharger les paramètres de l'instance
        current_prediction_type = kwargs.get('prediction_type', self.prediction_type)
        current_threshold = kwargs.get('threshold', self.threshold)
        
        base_signals = generate_base_signals_from_predictions(
            predictions=predictions,
            symbol=symbol,
            threshold=current_threshold,
            prediction_type=current_prediction_type
        )
        
        # Mettre à jour les signaux avec la source avant le dimensionnement
        base_signals = self._add_source_to_signals(base_signals)

        # Paramètres pour le dimensionnement, surchargeables par kwargs
        current_fraction_per_trade = kwargs.get('fraction_per_trade', self.fraction_per_trade)
        current_default_target_leverage = kwargs.get('default_target_leverage', self.default_target_leverage)
        current_max_leverage_cap = kwargs.get('max_leverage_cap', self.max_leverage_cap)

        sized_signals = determine_position_size_and_leverage(
            base_signals=base_signals,
            total_capital=total_capital,
            current_prices=current_prices,
            contract_multipliers=contract_multipliers,
            fraction_per_trade=current_fraction_per_trade,
            default_target_leverage=current_default_target_leverage,
            max_leverage_cap=current_max_leverage_cap
            # On pourrait aussi passer risk_manager_limits ou market_data_for_sizing ici
            # si disponibles dans kwargs ou via la configuration du générateur.
        )
        return sized_signals

# TODO: Implémenter EnsembleSignalGenerator qui pourrait agréger des signaux
# de plusieurs BasicSignalGenerator (ou d'autres types) et potentiellement
# leurs suggestions de levier.

# TODO: La fonction determine_position_size_and_leverage pourrait être étendue pour:
# 1. Interagir avec un RiskManager pour valider les tailles de position/levier
#    par rapport aux limites de risque globales du portefeuille.
# 2. Utiliser des données de marché supplémentaires (passées via market_data_for_sizing)
#    comme l'open interest, les funding rates, la volatilité, pour affiner
#    dynamiquement fraction_per_trade ou target_leverage.
#    Par exemple, réduire le levier en période de forte volatilité,
#    ou augmenter la taille si l'open interest confirme un mouvement.
# 3. Gérer plus finement l'arrondi des contrats (selon les spécifications du broker).
# 4. Permettre des stratégies de dimensionnement plus complexes (ex: Kelly Criterion, Volatility Targeting).

if __name__ == '__main__':
    # Exemple d'utilisation pour generate_base_signals_from_predictions (probabilités)
    sample_probabilities = pd.Series(
        [0.1, 0.4, 0.65, 0.9, 0.5],
        index=pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:05', '2023-01-01 10:10', '2023-01-01 10:15', '2023-01-01 10:20'])
    )
    print("Prédictions (probabilités):")
    print(sample_probabilities)
    # Test direct de generate_base_signals_from_predictions
    base_signals_prob_direct = generate_base_signals_from_predictions(sample_probabilities, symbol="BTCUSDT", threshold=0.55)
    print("\nSignaux de base générés directement (BTCUSDT, probabilités, seuil 0.55):")
    for signal in base_signals_prob_direct:
        print(signal)

    # Configuration pour le dimensionnement et le générateur
    current_capital_main = 100000
    trade_fraction_main = 0.02
    current_prices_market_main = {"BTCUSDT": 60000, "ETHUSDT": 3500, "ADAUSDT": 0.40}
    contract_multipliers_assets_main = {"BTCUSDT": 0.001, "ETHUSDT": 0.01, "ADAUSDT": 10}
    default_leverage_main = 5.0
    max_leverage_main = 15.0

    print(f"\n--- Configuration Globale pour les Tests ---")
    print(f"Capital: {current_capital_main}, Fraction/Trade: {trade_fraction_main}")
    print(f"Levier Défaut: {default_leverage_main}, Levier Max: {max_leverage_main}")
    print(f"Prix: {current_prices_market_main}")
    print(f"Multiplicateurs: {contract_multipliers_assets_main}")

    # Instanciation du BasicSignalGenerator
    basic_generator = BasicSignalGenerator(
        threshold=0.55,
        prediction_type='probability',
        fraction_per_trade=trade_fraction_main,
        default_target_leverage=default_leverage_main,
        max_leverage_cap=max_leverage_main,
        signal_source_name="ProbModelBTC"
    )

    print("\n--- Test BasicSignalGenerator avec prédictions de probabilités (BTCUSDT) ---")
    sized_signals_btc_prob = basic_generator.generate_signals(
        predictions=sample_probabilities,
        symbol="BTCUSDT",
        total_capital=current_capital_main,
        current_prices=current_prices_market_main, # Doit contenir BTCUSDT
        contract_multipliers=contract_multipliers_assets_main # Doit contenir BTCUSDT
    )
    for signal in sized_signals_btc_prob:
        print(signal)

    # Test BasicSignalGenerator avec prédictions de classes (ETHUSDT)
    sample_classes_eth = pd.Series(
        [0, 1, 0, 1, 1],
        index=pd.to_datetime(['2023-01-02 10:00', '2023-01-02 10:05', '2023-01-02 10:10', '2023-01-02 10:15', '2023-01-02 10:20'])
    )
    eth_generator = BasicSignalGenerator(
        prediction_type='class', # Important pour interpréter 0 et 1
        fraction_per_trade=trade_fraction_main,
        default_target_leverage=default_leverage_main,
        max_leverage_cap=max_leverage_main,
        signal_source_name="ClassModelETH"
    )
    print("\n--- Test BasicSignalGenerator avec prédictions de classes (ETHUSDT) ---")
    # Simuler un modèle qui fournit un levier cible pour un signal spécifique via la prédiction elle-même
    # Pour cela, il faudrait que `generate_base_signals_from_predictions` puisse extraire cela,
    # ou que `TradingSignal` soit enrichi avant `determine_position_size_and_leverage`.
    # Actuellement, `target_leverage` sur `TradingSignal` est rempli par `determine_position_size_and_leverage`.
    # Pour simuler un modèle qui suggère un levier, on pourrait modifier le signal *après* `generate_base_signals`
    # et *avant* `determine_position_size_and_leverage`.
    # Le BasicSignalGenerator ne fait pas cela directement, mais on peut le montrer.
    
    base_eth_signals = generate_base_signals_from_predictions(sample_classes_eth, "ETHUSDT", prediction_type='class')
    # Supposons qu'un modèle plus intelligent ait analysé le deuxième signal (BUY) et suggère un levier
    if len(base_eth_signals) > 1 and base_eth_signals[1].action == 'BUY':
        base_eth_signals[1].target_leverage = 10.0 # Suggestion du modèle
        base_eth_signals[1].confidence = 0.90 # Confiance associée
        print(f"Signal ETHUSDT (index 1) enrichi manuellement avant dimensionnement: {base_eth_signals[1]}")

    # On utilise maintenant ces signaux de base (potentiellement enrichis) avec determine_position_size_and_leverage
    # via le générateur (qui appellera determine_position_size_and_leverage en interne)
    # Pour que le générateur utilise le target_leverage du signal de base, il faut que
    # determine_position_size_and_leverage le priorise. C'est déjà le cas.
    
    sized_signals_eth_class = eth_generator.generate_signals(
        predictions=sample_classes_eth, # Ces prédictions généreront des signaux de base SANS levier initial
        symbol="ETHUSDT",               # Le levier sera appliqué par determine_position_size_and_leverage
        total_capital=current_capital_main,
        current_prices=current_prices_market_main,
        contract_multipliers=contract_multipliers_assets_main
    )
    # Pour tester le cas où le signal de base a déjà un target_leverage, il faudrait
    # que BasicSignalGenerator permette de passer des signaux déjà partiellement formés,
    # ou modifier sa logique interne.
    # Alternative: on appelle determine_position_size_and_leverage directement avec les signaux enrichis.
    print("Signaux ETH dimensionnés par le générateur (levier par défaut ou basé sur confiance interne):")
    for signal in sized_signals_eth_class:
        print(signal)
        
    print("\nDimensionnement direct des signaux ETH de base (dont un enrichi avec target_leverage=10):")
    sized_eth_manual_enrich = determine_position_size_and_leverage(
        base_signals_eth_class, # Contient le signal avec target_leverage=10.0
        current_capital_main,
        current_prices_market_main,
        contract_multipliers_assets_main,
        fraction_per_trade=trade_fraction_main,
        default_target_leverage=default_leverage_main,
        max_leverage_cap=max_leverage_main
    )
    for signal in sized_eth_manual_enrich:
        print(signal) # On s'attend à ce que le signal enrichi ait un levier de 10.

    # Cas HOLD avec le générateur
    hold_predictions_ada = pd.Series(
        [0.5, 0.5, 0.5],
        index=pd.to_datetime(['2023-01-03 10:00', '2023-01-03 10:05', '2023-01-03 10:10'])
    )
    ada_generator = BasicSignalGenerator(signal_source_name="HoldModelADA", threshold=0.55)
    print("\n--- Test BasicSignalGenerator avec prédictions HOLD (ADAUSDT) ---")
    sized_signals_ada_hold = ada_generator.generate_signals(
        predictions=hold_predictions_ada,
        symbol="ADAUSDT",
        total_capital=current_capital_main,
        current_prices=current_prices_market_main,
        contract_multipliers=contract_multipliers_assets_main
    )
    for signal in sized_signals_ada_hold:
        print(signal)

    # Exemple avec un signal ayant une confiance qui modifie le levier, via le générateur
    print("\n--- Test BasicSignalGenerator: levier modulé par la confiance (BTCUSDT) ---")
    # Le générateur utilise la confiance issue de generate_base_signals_from_predictions
    # et determine_position_size_and_leverage l'utilise.
    # La prédiction de 0.9 pour BTCUSDT devrait donner une confiance de 0.9.
    # Levier attendu: default (5.0) + (0.9 - 0.5)*10 = 5.0 + 4.0 = 9.0
    # Le signal avec prédiction 0.65 devrait donner confiance 0.65.
    # Levier attendu: default (5.0) + (0.65 - 0.5)*10 = 5.0 + 1.5 = 6.5
    
    # On réutilise sized_signals_btc_prob déjà calculé avec le générateur
    print("Signaux BTC (prob) déjà calculés par le générateur:")
    for signal in sized_signals_btc_prob:
        if signal.action == 'BUY': # On s'intéresse aux BUYs pour voir le levier
            print(f"Signal: {signal.action} {signal.symbol} @ {signal.timestamp}, Conf: {signal.confidence:.2f}, Lev: {signal.target_leverage}, Qty: {signal.quantity}")

    print("\n--- Test BasicSignalGenerator: surcharge des paramètres via kwargs ---")
    custom_btc_generator = BasicSignalGenerator(
        threshold=0.6, # Seuil plus strict
        default_target_leverage=3.0, # Levier par défaut plus bas
        fraction_per_trade=0.01, # Moins de capital par trade
        signal_source_name="CustomBTC"
    )
    sized_signals_btc_custom_instance = custom_btc_generator.generate_signals(
        predictions=sample_probabilities, # mêmes prédictions BTC
        symbol="BTCUSDT",
        total_capital=current_capital_main,
        current_prices=current_prices_market_main,
        contract_multipliers=contract_multipliers_assets_main
    )
    print("Signaux BTC avec générateur aux paramètres d'instance custom:")
    for signal in sized_signals_btc_custom_instance:
        if signal.action != 'HOLD': print(signal)

    sized_signals_btc_custom_kwargs = basic_generator.generate_signals(
        predictions=sample_probabilities, # mêmes prédictions BTC
        symbol="BTCUSDT",
        total_capital=current_capital_main,
        current_prices=current_prices_market_main,
        contract_multipliers=contract_multipliers_assets_main,
        # Surcharge des paramètres du basic_generator pour cet appel:
        threshold=0.6,
        default_target_leverage=2.0,
        fraction_per_trade=0.005
    )
    print("\nSignaux BTC avec générateur de base mais paramètres surchargés via kwargs:")
    for signal in sized_signals_btc_custom_kwargs:
        if signal.action != 'HOLD': print(signal)