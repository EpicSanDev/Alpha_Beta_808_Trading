#!/usr/bin/env python3
"""
Multi-Asset Portfolio Management Module
Implements advanced portfolio construction and management for multiple assets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import cvxpy as cp

# Suppression des warnings
warnings.filterwarnings("ignore")

class AssetClass(Enum):
    """Classes d'actifs supportées"""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"

class RebalancingMethod(Enum):
    """Méthodes de rééquilibrage"""
    CALENDAR = "calendar"          # Rééquilibrage périodique
    THRESHOLD = "threshold"        # Basé sur des seuils de dérive
    VOLATILITY = "volatility"      # Basé sur la volatilité
    SIGNAL_STRENGTH = "signal_strength"  # Basé sur la force des signaux

@dataclass
class Asset:
    """Représente un actif dans le portefeuille"""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    base_currency: str
    quote_currency: str
    min_trade_size: float
    tick_size: float
    commission_rate: float # Pourcentage de commission (ex: 0.001 pour 0.1%)
    is_active: bool = True
    leverage: float = 1.0 # Levier par défaut pour cet actif si non spécifié ailleurs
    weight_constraint_min: float = 0.0
    weight_constraint_max: float = 1.0
    is_future: bool = False # Indique si l'actif est un contrat à terme
    contract_multiplier: float = 1.0 # Multiplicateur de contrat (ex: pour les indices ou certains futures)
    margin_currency: Optional[str] = None # Devise de marge pour les futures

@dataclass
class Position:
    """Représente une position dans le portefeuille"""
    symbol: str
    quantity: float  # Nombre de contrats/actions. Positif pour LONG, négatif pour SHORT (convention à clarifier/adapter si besoin)
    entry_price: float # Prix d'entrée moyen par contrat/action
    current_price: float # Prix actuel du marché par contrat/action
    timestamp: datetime # Timestamp de la dernière mise à jour ou de l'entrée
    
    # Champs spécifiques aux futures (peuvent être None ou 0 pour les actifs spot)
    position_type: Optional[str] = None # 'LONG' ou 'SHORT', pertinent surtout pour les futures
    leverage_used: Optional[float] = None # Levier effectivement utilisé pour cette position
    initial_margin_used: Optional[float] = None # Marge initiale bloquée pour cette position
    contract_multiplier: float = 1.0 # Multiplicateur du contrat (hérité de Asset)

    # Valeurs calculées
    # La définition de position_value doit être claire: valeur notionnelle ou valeur de l'équité?
    # Pour les futures, la valeur notionnelle est souvent quantity * current_price * contract_multiplier
    # L'équité de la position serait initial_margin_used + unrealized_pnl
    notional_value: float = 0.0 # Valeur notionnelle actuelle de la position
    unrealized_pnl: float = 0.0
    
    # Ces champs peuvent être redondants ou nécessiter une clarification de leur calcul
    # pour les futures par rapport aux actifs spot.
    position_value: float = 0.0 # À clarifier: est-ce la valeur notionnelle ou l'équité?
    weight: float = 0.0 # Poids dans le portefeuille
    unrealized_pnl_pct: float = 0.0 # P&L non réalisé en pourcentage

    def __post_init__(self):
        # Assurer la cohérence pour les positions futures
        if self.position_type == 'SHORT' and self.quantity > 0:
            self.quantity *= -1 # Convention: quantité négative pour SHORT
        
        # Calcul initial de la valeur notionnelle et du P&L (peut être recalculé par le manager)
        self.notional_value = abs(self.quantity) * self.current_price * self.contract_multiplier
        
        pnl_per_contract = 0
        if self.position_type == 'LONG':
            pnl_per_contract = (self.current_price - self.entry_price) * self.contract_multiplier
        elif self.position_type == 'SHORT':
            pnl_per_contract = (self.entry_price - self.current_price) * self.contract_multiplier
        self.unrealized_pnl = pnl_per_contract * abs(self.quantity)

        if self.initial_margin_used is not None and self.initial_margin_used > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / self.initial_margin_used) if self.initial_margin_used != 0 else 0.0
        elif self.entry_price > 0 and self.position_type is not None : # Pour les actifs spot ou si la marge n'est pas le dénominateur
            entry_notional = abs(self.quantity) * self.entry_price * self.contract_multiplier
            self.unrealized_pnl_pct = (self.unrealized_pnl / entry_notional) if entry_notional != 0 else 0.0
        
        # Laisser position_value être calculé par le PortfolioManager pour la cohérence globale du portefeuille
        # ou définir une convention claire ici. Pour l'instant, on peut l'aligner sur la valeur notionnelle.
        self.position_value = self.notional_value

class PortfolioOptimizer:
    """
    Optimiseur de portefeuille utilisant différentes méthodes d'allocation
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
        
    def mean_variance_optimization(self,
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 constraints: Dict = None) -> np.ndarray:
        """
        Optimisation moyenne-variance de Markowitz
        
        Args:
            expected_returns: Rendements attendus
            covariance_matrix: Matrice de covariance
            constraints: Contraintes d'optimisation
            
        Returns:
            Vecteur de poids optimaux
        """
        n_assets = len(expected_returns)
        
        # Variables d'optimisation
        weights = cp.Variable(n_assets)
        
        # Fonction objectif: maximiser l'utilité espérée
        portfolio_return = expected_returns.T @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
        
        # Contraintes de base
        constraints_list = [
            weights >= 0,  # Pas de vente à découvert
            cp.sum(weights) == 1  # Somme des poids = 1
        ]
        
        # Contraintes additionnelles
        if constraints:
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
        
        # Résolution
        problem = cp.Problem(cp.Maximize(utility), constraints_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if weights.value is None:
            # Fallback: poids égaux
            return np.ones(n_assets) / n_assets
        
        return np.array(weights.value).flatten()
    
    def risk_parity_optimization(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Optimisation par parité de risque
        
        Args:
            covariance_matrix: Matrice de covariance
            
        Returns:
            Vecteur de poids risk parity
        """
        n_assets = covariance_matrix.shape[0]
        
        def objective(weights):
            """Minimiser la différence entre les contributions au risque"""
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            
            # Différences au carré par rapport à la contribution égale (1/n)
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Somme = 1
        ]
        
        bounds = [(0.001, 0.5) for _ in range(n_assets)]  # Bornes réalistes
        
        # Point de départ: poids égaux
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return initial_weights
    
    def black_litterman_optimization(self,
                                   market_caps: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   views: Dict = None,
                                   tau: float = 0.025) -> np.ndarray:
        """
        Modèle de Black-Litterman
        
        Args:
            market_caps: Capitalisations de marché
            covariance_matrix: Matrice de covariance
            views: Vues du gestionnaire {'asset_idx': expected_return}
            tau: Paramètre d'incertitude
            
        Returns:
            Vecteur de poids Black-Litterman
        """
        # Poids de marché (baseline)
        market_weights = market_caps / np.sum(market_caps)
        
        # Rendements implicites du marché
        risk_aversion = 3.0  # Paramètre typique
        pi = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        if views is None or len(views) == 0:
            # Pas de vues: retourner les poids de marché ajustés
            expected_returns = pi
        else:
            # Intégrer les vues du gestionnaire
            n_assets = len(market_weights)
            n_views = len(views)
            
            # Matrice P (picking matrix)
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)  # Vecteur des vues
            
            for i, (asset_idx, view_return) in enumerate(views.items()):
                P[i, asset_idx] = 1.0
                Q[i] = view_return
            
            # Matrice Omega (incertitude des vues)
            Omega = np.diag(np.diag(P @ (tau * covariance_matrix) @ P.T))
            
            # Calcul des nouveaux rendements attendus
            tau_sigma = tau * covariance_matrix
            M1 = np.linalg.inv(tau_sigma)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M3 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
            
            expected_returns = np.linalg.inv(M1 + M2) @ M3
        
        # Optimisation avec les nouveaux rendements attendus
        return self.mean_variance_optimization(expected_returns, covariance_matrix)


class MultiAssetPortfolioManager:
    """
    Gestionnaire de portefeuille multi-actifs avec optimisation et rééquilibrage automatique
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 rebalancing_method: RebalancingMethod = RebalancingMethod.CALENDAR,
                 rebalancing_frequency_days: int = 30,
                 drift_threshold: float = 0.05,  # 5% de dérive avant rééquilibrage
                 volatility_lookback: int = 30,
                 max_position_size: float = 0.2,  # 20% max par actif
                 min_position_size: float = 0.01,  # 1% min par actif
                 transaction_cost: float = 0.001,  # 0.1% de frais de transaction
                 optimization_method: str = 'mean_variance'):
        """
        Initialise le gestionnaire de portefeuille multi-actifs
        
        Args:
            initial_capital: Capital initial
            rebalancing_method: Méthode de rééquilibrage
            rebalancing_frequency_days: Fréquence de rééquilibrage en jours
            drift_threshold: Seuil de dérive pour déclencher un rééquilibrage
            volatility_lookback: Période pour le calcul de volatilité
            max_position_size: Taille maximale d'une position (% du portefeuille)
            min_position_size: Taille minimale d'une position (% du portefeuille)
            transaction_cost: Coût de transaction par trade
            optimization_method: Méthode d'optimisation ('mean_variance', 'risk_parity', 'black_litterman')
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.rebalancing_method = rebalancing_method
        self.rebalancing_frequency_days = rebalancing_frequency_days
        self.drift_threshold = drift_threshold
        self.volatility_lookback = volatility_lookback
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.transaction_cost = transaction_cost
        self.optimization_method = optimization_method
        
        # Assets et positions
        self.assets = {}  # Dict[str, Asset]
        self.positions = {}  # Dict[str, Position]
        self.target_weights = {}  # Dict[str, float]
        self.current_weights = {}  # Dict[str, float]
        
        # Historique et métriques
        self.portfolio_history = []
        self.rebalancing_history = []
        self.last_rebalance_date = None
        self.transaction_costs_paid = 0.0
        
        # Données de marché
        self.price_data = {}  # Dict[str, pd.DataFrame]
        self.correlation_matrix = None
        self.covariance_matrix = None
        
        # Optimiseur
        self.optimizer = PortfolioOptimizer()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_asset(self, 
                  symbol: str = None, 
                  price_data: pd.DataFrame = None, 
                  returns_data: pd.Series = None, 
                  asset: Asset = None,
                  is_future: bool = False,
                  contract_multiplier: float = 1.0,
                  margin_currency: Optional[str] = None,
                  asset_class: AssetClass = AssetClass.CRYPTO, # Permet de spécifier la classe d'actif
                  name: Optional[str] = None, # Permet de spécifier un nom différent du symbole
                  exchange: str = "binance", # Exchange par défaut
                  base_currency: Optional[str] = None,
                  quote_currency: Optional[str] = "USDT", # Devise de cotation par défaut
                  min_trade_size: float = 0.001,
                  tick_size: float = 0.01,
                  commission_rate: float = 0.001,
                  is_active: bool = True,
                  leverage: float = 1.0):
        """
        Ajoute un actif au portefeuille
        
        Args:
            symbol: Symbol of the asset (if creating asset from parameters)
            price_data: Price data for the asset
            returns_data: Returns data for the asset
            asset: Asset object (if providing complete asset)
            is_future: True if the asset is a future contract
            contract_multiplier: Contract multiplier for futures
            margin_currency: Margin currency for futures
            asset_class: Class of the asset
            name: Name of the asset
            exchange: Exchange where the asset is traded
            base_currency: Base currency of the asset
            quote_currency: Quote currency of the asset
            min_trade_size: Minimum trade size
            tick_size: Tick size for price changes
            commission_rate: Commission rate for trading
            is_active: Whether the asset is active
            leverage: Default leverage for the asset
        """
        if asset is not None:
            # Use provided Asset object
            self.assets[asset.symbol] = asset
            self.target_weights[asset.symbol] = 0.0
            self.current_weights[asset.symbol] = 0.0
            log_msg = f"Actif ajouté (objet fourni): {asset.symbol} ({asset.asset_class.value}"
            if asset.is_future:
                log_msg += f", Future, Multiplier: {asset.contract_multiplier}, Marge: {asset.margin_currency})"
            else:
                log_msg += ")"
            self.logger.info(log_msg)
        elif symbol is not None:
            # Create Asset from parameters
            asset_name = name if name else symbol
            final_base_currency = base_currency if base_currency else (symbol[:-len(quote_currency)] if quote_currency and symbol.endswith(quote_currency) else symbol)
            
            new_asset = Asset(
                symbol=symbol,
                name=asset_name,
                asset_class=asset_class,
                exchange=exchange,
                base_currency=final_base_currency,
                quote_currency=quote_currency,
                min_trade_size=min_trade_size,
                tick_size=tick_size,
                commission_rate=commission_rate,
                is_active=is_active,
                leverage=leverage, # Utilisation du paramètre leverage
                is_future=is_future,
                contract_multiplier=contract_multiplier,
                margin_currency=margin_currency
            )
            self.assets[symbol] = new_asset
            self.target_weights[symbol] = 0.0
            self.current_weights[symbol] = 0.0
            
            # Store price and returns data if provided
            if price_data is not None:
                self.update_price_data(symbol, price_data)
            
            log_msg = f"Actif ajouté (paramètres): {symbol} ({new_asset.asset_class.value}"
            if new_asset.is_future:
                log_msg += f", Future, Multiplier: {new_asset.contract_multiplier}, Marge: {new_asset.margin_currency})"
            else:
                log_msg += ")"
            self.logger.info(log_msg)
        else:
            raise ValueError("Must provide either 'asset' object or 'symbol' and other required parameters")
    
    def update_price_data(self, symbol: str, price_data: pd.DataFrame):
        """Met à jour les données de prix pour un actif"""
        self.price_data[symbol] = price_data.copy()
        
        # Mettre à jour le prix actuel des positions
        if symbol in self.positions:
            current_price = price_data['close'].iloc[-1]
            position = self.positions[symbol]
            position.current_price = current_price
            # La mise à jour de position_value et P&L doit tenir compte du contract_multiplier pour les futures
            asset_details = self.assets.get(symbol)
            contract_mult = asset_details.contract_multiplier if asset_details else 1.0

            if asset_details and asset_details.is_future:
                position.notional_value = abs(position.quantity) * current_price * contract_mult
                # Le P&L pour les futures est (current_price - entry_price) * quantity * multiplier
                # La quantité peut être négative pour les shorts
                pnl_per_contract = (current_price - position.entry_price) * contract_mult
                position.unrealized_pnl = pnl_per_contract * position.quantity # quantity est déjà signée
                
                # position_value pour les futures est souvent la valeur notionnelle ou l'équité.
                # Ici, nous utilisons la valeur notionnelle pour la cohérence avec __post_init__
                position.position_value = position.notional_value
                
                if position.initial_margin_used is not None and position.initial_margin_used > 0:
                    position.unrealized_pnl_pct = (position.unrealized_pnl / position.initial_margin_used) if position.initial_margin_used != 0 else 0.0
                elif position.entry_price > 0 : # Fallback si marge non définie
                     entry_notional = abs(position.quantity) * position.entry_price * contract_mult
                     position.unrealized_pnl_pct = (position.unrealized_pnl / entry_notional) if entry_notional != 0 else 0.0

            else: # Actifs Spot
                position.position_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                if position.entry_price > 0:
                    position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
                else:
                    position.unrealized_pnl_pct = 0.0
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calcule les métriques du portefeuille"""
        if not self.positions:
            return {
                'total_value': self.current_capital, # Devrait être la somme du capital disponible et de l'équité des positions
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'num_positions': 0,
                'total_notional_value': 0.0,
                'total_used_margin': 0.0,
            }
        
        # Pour les futures, la "valeur" du portefeuille est plus complexe.
        # On peut considérer l'équité totale = capital disponible + somme(marge utilisée) + somme(P&L non réalisé)
        # Ou simplement le capital initial + P&L total réalisé et non réalisé.
        # Pour l'instant, nous allons calculer la somme des valeurs notionnelles et des P&L.
        
        total_notional_value = 0.0
        total_unrealized_pnl = 0.0
        total_used_margin = 0.0 # Marge totale utilisée par les positions futures

        for symbol, position in self.positions.items():
            asset = self.assets.get(symbol)
            if asset and asset.is_future:
                # Mettre à jour la position avec les dernières infos de l'asset (contract_multiplier)
                position.contract_multiplier = asset.contract_multiplier
                # Recalculer P&L et valeur notionnelle au cas où le prix aurait changé
                # et que update_price_data n'aurait pas été appelé explicitement pour cet actif
                # ou pour s'assurer que les calculs sont à jour avec les bons multiplicateurs.
                
                # Recalcul du P&L basé sur la logique de __post_init__
                pnl_per_contract = 0
                if position.position_type == 'LONG':
                    pnl_per_contract = (position.current_price - position.entry_price) * position.contract_multiplier
                elif position.position_type == 'SHORT': # quantity sera négative
                    pnl_per_contract = (position.entry_price - position.current_price) * position.contract_multiplier
                
                # La quantité dans la dataclass Position est déjà signée pour les shorts.
                # Pour le P&L, on utilise la quantité absolue pour le calcul par contrat, puis on multiplie par la quantité signée.
                # Correction: P&L = (prix_actuel - prix_entrée) * quantité * multiplicateur pour LONG
                # P&L = (prix_entrée - prix_actuel) * abs(quantité) * multiplicateur pour SHORT
                # Ou plus simple: (prix_actuel - prix_entrée) * quantité_signée * multiplicateur
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * position.contract_multiplier

                position.notional_value = abs(position.quantity) * position.current_price * position.contract_multiplier
                total_notional_value += position.notional_value
                if position.initial_margin_used is not None:
                    total_used_margin += position.initial_margin_used
                
                # Mise à jour de position.position_value (convention: valeur notionnelle pour futures)
                position.position_value = position.notional_value

            else: # Actifs Spot
                position.position_value = position.quantity * position.current_price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                total_notional_value += position.position_value # Pour les spots, la valeur notionnelle est la valeur de la position

            total_unrealized_pnl += position.unrealized_pnl

        # La valeur totale du portefeuille pour un compte avec futures est souvent:
        # Capital disponible + Marge utilisée + P&L non réalisé des positions ouvertes
        # self.current_capital ici représente les fonds non alloués à la marge.
        # Donc, portfolio_equity = self.current_capital + total_used_margin + total_unrealized_pnl
        # Si self.current_capital est le capital total initial, alors
        # portfolio_equity = self.initial_capital + total_realized_pnl (non suivi ici) + total_unrealized_pnl
        # Pour simplifier et aligner avec le simulateur, utilisons une approche basée sur l'équité:
        portfolio_equity = self.current_capital + total_used_margin + total_unrealized_pnl
        
        # Calculer les poids actuels. Pour les futures, le poids peut être basé sur la marge utilisée ou la valeur notionnelle.
        # Utilisons la valeur notionnelle pour le calcul du poids pour l'instant.
        # Si total_notional_value est nul (ex: portefeuille vide ou cash uniquement), éviter division par zéro.
        if portfolio_equity > 0: # Utiliser l'équité du portefeuille comme base pour les poids
            for symbol, position in self.positions.items():
                asset = self.assets.get(symbol)
                # Le poids d'une position future peut être sa contribution à la marge ou sa valeur notionnelle / équité totale.
                # Pour la diversification, la valeur notionnelle est plus représentative de l'exposition.
                if asset and asset.is_future:
                     # Poids basé sur la valeur notionnelle par rapport à l'équité totale du portefeuille
                    self.current_weights[symbol] = position.notional_value / portfolio_equity if portfolio_equity > 0 else 0.0
                else: # Spot
                    self.current_weights[symbol] = position.position_value / portfolio_equity if portfolio_equity > 0 else 0.0
                position.weight = self.current_weights[symbol]
        else: # Si l'équité est nulle ou négative, les poids sont nuls.
            for symbol in self.positions:
                self.current_weights[symbol] = 0.0
                self.positions[symbol].weight = 0.0
        
        return {
            'total_value': portfolio_equity, # Représente l'équité totale du portefeuille
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': total_unrealized_pnl / self.initial_capital if self.initial_capital > 0 else 0.0,
            'num_positions': len(self.positions),
            'current_weights': self.current_weights.copy(),
            'target_weights': self.target_weights.copy(),
            'total_notional_value': total_notional_value, # Exposition notionnelle totale
            'total_used_margin': total_used_margin, # Marge totale utilisée
            'available_capital': self.current_capital # Capital non utilisé comme marge
        }
    
    def calculate_expected_returns(self, lookback_days: int = 60) -> Dict[str, float]:
        """
        Calcule les rendements attendus pour chaque actif
        
        Args:
            lookback_days: Période de calcul en jours
            
        Returns:
            Dict avec les rendements attendus annualisés
        """
        expected_returns = {}
        
        for symbol, price_data in self.price_data.items():
            if len(price_data) < lookback_days:
                expected_returns[symbol] = 0.0
                continue
            
            # Calculer les rendements quotidiens
            returns = price_data['close'].pct_change().dropna()
            recent_returns = returns.tail(lookback_days)
            
            # Rendement moyen annualisé
            avg_return = recent_returns.mean() * 252  # 252 jours de trading par an
            expected_returns[symbol] = avg_return
        
        return expected_returns
    
    def calculate_covariance_matrix(self, lookback_days: int = 60) -> np.ndarray:
        """
        Calcule la matrice de covariance des rendements
        
        Args:
            lookback_days: Période de calcul
            
        Returns:
            Matrice de covariance
        """
        symbols = list(self.price_data.keys())
        returns_data = []
        
        for symbol in symbols:
            if symbol not in self.price_data or len(self.price_data[symbol]) < lookback_days:
                # Remplacer par des rendements nuls si pas assez de données
                returns_data.append(np.zeros(lookback_days))
            else:
                returns = self.price_data[symbol]['close'].pct_change().dropna()
                recent_returns = returns.tail(lookback_days)
                returns_data.append(recent_returns.values)
        
        if not returns_data:
            return np.array([[]])
        
        # Aligner les séries sur la même longueur
        min_length = min(len(series) for series in returns_data)
        aligned_returns = np.array([series[-min_length:] for series in returns_data]).T
        
        # Utiliser l'estimateur de Ledoit-Wolf pour la robustesse
        cov_estimator = LedoitWolf()
        covariance_matrix = cov_estimator.fit(aligned_returns).covariance_
        
        # Annualiser la covariance
        self.covariance_matrix = covariance_matrix * 252
        
        return self.covariance_matrix
    
    def optimize_portfolio(self, 
                         method: str = None,
                         expected_returns: Dict[str, float] = None,
                         views: Dict[str, float] = None) -> Dict[str, float]:
        """
        Optimise l'allocation du portefeuille
        
        Args:
            method: Optimization method ('mean_variance', 'risk_parity', 'black_litterman', 'equal_weight')
            expected_returns: Rendements attendus personnalisés
            views: Vues du gestionnaire pour Black-Litterman
            
        Returns:
            Dict avec les poids optimaux
        """
        symbols = list(self.assets.keys())
        
        if len(symbols) == 0:
            return {}
        
        # Handle equal weight allocation
        if method == 'equal_weight':
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}
        
        # Use the optimization method from init if not specified
        if method is None:
            method = self.optimization_method
        
        # Calculer les rendements attendus si non fournis
        if expected_returns is None:
            expected_returns = self.calculate_expected_returns()
        
        # Calculer la matrice de covariance
        covariance_matrix = self.calculate_covariance_matrix()
        
        if covariance_matrix.size == 0:
            # Fallback: poids égaux
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}
        
        # Préparer les arrays pour l'optimisation
        returns_array = np.array([expected_returns.get(symbol, 0.0) for symbol in symbols])
        
        # Contraintes basées sur les assets
        constraints = {
            'min_weight': np.array([self.assets[symbol].weight_constraint_min for symbol in symbols]),
            'max_weight': np.array([min(self.assets[symbol].weight_constraint_max, self.max_position_size) 
                                  for symbol in symbols])
        }
        
        # Optimisation selon la méthode choisie
        if method == 'mean_variance':
            optimal_weights = self.optimizer.mean_variance_optimization(
                returns_array, covariance_matrix, constraints)
                
        elif method == 'risk_parity':
            optimal_weights = self.optimizer.risk_parity_optimization(covariance_matrix)
            
        elif method == 'black_litterman':
            # Utiliser des caps de marché simulées (ou égales si pas de données)
            market_caps = np.ones(len(symbols))
            optimal_weights = self.optimizer.black_litterman_optimization(
                market_caps, covariance_matrix, views)
        
        else:
            # Défaut: poids égaux
            optimal_weights = np.ones(len(symbols)) / len(symbols)
        
        # Convertir en dictionnaire
        optimal_allocation = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
        
        # Appliquer les contraintes minimales
        for symbol in optimal_allocation:
            if optimal_allocation[symbol] < self.min_position_size:
                optimal_allocation[symbol] = 0.0
        
        # Renormaliser après application des contraintes
        total_weight = sum(optimal_allocation.values())
        if total_weight > 0:
            optimal_allocation = {symbol: weight / total_weight 
                                for symbol, weight in optimal_allocation.items()}
        
        return optimal_allocation
    
    def should_rebalance(self, current_date: datetime) -> bool:
        """
        Détermine si un rééquilibrage est nécessaire
        
        Args:
            current_date: Date actuelle
            
        Returns:
            True si un rééquilibrage est nécessaire
        """
        if self.last_rebalance_date is None:
            return True
        
        if self.rebalancing_method == RebalancingMethod.CALENDAR:
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance >= self.rebalancing_frequency_days
        
        elif self.rebalancing_method == RebalancingMethod.THRESHOLD:
            # Calculer la dérive par rapport aux poids cibles
            # Recalculer les métriques pour avoir les poids actuels à jour
            metrics = self.calculate_portfolio_metrics()
            current_weights = metrics.get('current_weights', {})

            max_drift = 0.0
            for symbol, target_weight in self.target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                # target_weight = self.target_weights[symbol] # Déjà obtenu
                if target_weight > 0: # Éviter division par zéro si le poids cible est 0
                    drift = abs(current_weight - target_weight) / target_weight
                    max_drift = max(max_drift, drift)
                elif current_weight > 0: # Si le poids cible est 0 mais qu'on a une position, c'est une dérive de 100%
                    max_drift = max(max_drift, 1.0)

            return max_drift > self.drift_threshold
        
        elif self.rebalancing_method == RebalancingMethod.VOLATILITY:
            # Rééquilibrer en cas de forte volatilité
            portfolio_returns = self.calculate_portfolio_returns()
            if len(portfolio_returns) >= self.volatility_lookback:
                recent_vol = portfolio_returns.tail(self.volatility_lookback).std() * np.sqrt(252)
                return recent_vol > 0.3  # 30% de volatilité annualisée
        
        return False
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """Calcule les rendements du portefeuille"""
        if not self.portfolio_history:
            return pd.Series(dtype=float) # Spécifier dtype pour éviter warning si vide
        
        values = [entry['total_value'] for entry in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def execute_rebalancing(self, 
                          target_allocation: Dict[str, float],
                          current_prices: Dict[str, float],
                          timestamp: datetime) -> List[Dict]:
        """
        Exécute le rééquilibrage du portefeuille pour atteindre l'allocation cible.
        Gère les actifs spot et futures.
        """
        orders = []
        portfolio_metrics = self.calculate_portfolio_metrics()
        # Utiliser l'équité totale du portefeuille pour calculer les valeurs cibles
        total_portfolio_equity = portfolio_metrics.get('total_value', self.current_capital)

        for symbol, target_weight in target_allocation.items():
            asset = self.assets.get(symbol)
            if not asset or not asset.is_active:
                continue

            current_price = current_prices.get(symbol)
            if current_price is None or current_price <= 0:
                self.logger.warning(f"Prix invalide ou manquant pour {symbol} lors du rééquilibrage. Actif ignoré.")
                continue

            # Valeur cible de l'exposition pour cet actif
            target_notional_value = total_portfolio_equity * target_weight
            
            # Quantité cible (nombre de contrats pour futures, nombre d'actions pour spot)
            # Pour les futures, la quantité est (Valeur Notionnelle Cible) / (Prix Actuel * Multiplicateur Contrat)
            target_quantity = target_notional_value / (current_price * asset.contract_multiplier)

            current_position = self.positions.get(symbol)
            current_quantity = current_position.quantity if current_position else 0.0
            
            # Pour les positions futures, la quantité peut être négative (short).
            # La logique de diff doit correctement gérer cela.
            # Si target_weight est 0, target_quantity sera 0, donc on ferme la position.
            # Si target_weight > 0 et current_quantity est short, on doit acheter pour couvrir et ensuite pour ouvrir long.
            # Si target_weight < 0 (non géré par l'optimiseur actuel, mais pour la robustesse), on initie un short.
            # L'optimiseur actuel ne génère que des poids >= 0.
            
            quantity_diff = target_quantity - current_quantity

            if abs(quantity_diff) * asset.contract_multiplier * current_price < asset.min_trade_size * current_price * asset.contract_multiplier and abs(quantity_diff) < asset.min_trade_size : # Ignorer les trades trop petits en valeur et en quantité
                 if abs(quantity_diff) > 1e-9: # Log si la différence n'est pas due à une imprécision flottante
                    self.logger.info(f"Différence de quantité pour {symbol} ({quantity_diff:.4f}) trop petite pour trader.")
                 continue


            order_type = 'buy' if quantity_diff > 0 else 'sell'
            order_quantity_abs = abs(quantity_diff) # Quantité absolue à trader

            # Calculer la valeur de l'ordre et les coûts de transaction
            order_notional_value = order_quantity_abs * current_price * asset.contract_multiplier
            transaction_cost = order_notional_value * asset.commission_rate # Utiliser commission_rate de l'Asset
            self.transaction_costs_paid += transaction_cost
            
            # Simuler l'impact sur le capital disponible (pour les actifs spot ou la marge des futures)
            # Pour les futures, l'achat/vente de contrats modifie la marge utilisée.
            # Pour les spots, cela modifie directement le capital.
            
            # Création de l'ordre
            order = {
                'symbol': symbol,
                'type': order_type,
                'quantity': order_quantity_abs, # Quantité de l'ordre (toujours positive)
                'price': current_price,
                'notional_value': order_notional_value,
                'transaction_cost': transaction_cost,
                'timestamp': timestamp,
                'reason': 'rebalancing',
                'is_future': asset.is_future,
                'contract_multiplier': asset.contract_multiplier
            }
            orders.append(order)

            # Mettre à jour la position (ou la créer si elle n'existe pas)
            if current_position is None:
                current_position = Position(
                    symbol=symbol,
                    quantity=0, # Sera mis à jour ci-dessous
                    entry_price=current_price, # Le prix d'entrée sera ajusté si moyennage
                    current_price=current_price,
                    timestamp=timestamp,
                    position_type=None, # Sera défini ci-dessous
                    leverage_used=asset.leverage, # Levier par défaut de l'asset
                    initial_margin_used=0.0, # Sera calculé
                    contract_multiplier=asset.contract_multiplier,
                    notional_value=0.0, # Sera recalculé
                    unrealized_pnl=0.0,
                    position_value=0.0, # Sera la valeur notionnelle
                    weight=0.0, # Sera recalculé
                    unrealized_pnl_pct=0.0
                )
                self.positions[symbol] = current_position
            
            new_quantity_signed = current_position.quantity + quantity_diff # Nouvelle quantité signée

            if asset.is_future:
                # Mettre à jour le type de position et la marge
                current_position.position_type = 'LONG' if new_quantity_signed > 0 else ('SHORT' if new_quantity_signed < 0 else None)
                
                # Marge initiale pour la NOUVELLE position totale
                # (Valeur Notionnelle Totale de la Position) / Levier
                new_notional_value_total = abs(new_quantity_signed) * current_price * asset.contract_multiplier
                new_initial_margin = new_notional_value_total / asset.leverage if asset.leverage > 0 else new_notional_value_total

                # Ajustement du capital disponible (self.current_capital)
                # Augmentation de la marge utilisée = new_initial_margin - marge_actuelle_de_cette_position
                # Si current_position.initial_margin_used est None, on le traite comme 0
                current_margin_for_pos = current_position.initial_margin_used if current_position.initial_margin_used is not None else 0.0
                margin_change = new_initial_margin - current_margin_for_pos
                self.current_capital -= margin_change # Réduire le capital si la marge augmente, l'augmenter si elle diminue
                
                current_position.initial_margin_used = new_initial_margin
                current_position.leverage_used = asset.leverage

            else: # Actif Spot
                # Pour les actifs spot, l'achat/vente affecte directement le capital
                self.current_capital -= quantity_diff * current_price # quantity_diff est positif pour achat (diminue capital), négatif pour vente (augmente capital)
            
            self.current_capital -= transaction_cost # Toujours déduire les coûts de transaction

            # Mettre à jour la quantité et le prix d'entrée de la position
            if abs(new_quantity_signed) < 1e-9: # Si la position est (presque) fermée
                # Retirer la position du dictionnaire si elle est fermée
                # Récupérer la marge si c'était un future
                if asset.is_future and current_position.initial_margin_used is not None:
                     self.current_capital += current_position.initial_margin_used # Rendre la marge
                del self.positions[symbol]
                if symbol in self.current_weights: del self.current_weights[symbol]
                if symbol in self.target_weights: self.target_weights[symbol] = 0 # S'assurer que le poids cible est 0
            else:
                if order_type == 'buy': # Augmentation de position ou ouverture long ou couverture short
                    # Si on augmente une position existante ou on ouvre une nouvelle position
                    if current_position.quantity == 0 or (current_position.quantity > 0 and quantity_diff > 0) or (current_position.quantity < 0 and quantity_diff > abs(current_position.quantity)): # Ouverture ou augmentation Long, ou flip de Short vers Long
                        current_position.entry_price = ( (current_position.quantity * current_position.entry_price) + (quantity_diff * current_price) ) / new_quantity_signed if new_quantity_signed !=0 else current_price
                    # Si on réduit une position short (achat pour couvrir)
                    # Le prix d'entrée ne change pas lors d'une réduction de position short. Le P&L est réalisé.
                    # La logique de P&L réalisé n'est pas explicitement gérée ici, mais dans le simulateur.
                # Si order_type == 'sell': Réduction de position long ou ouverture/augmentation short
                # Le prix d'entrée ne change pas lors d'une vente partielle d'une position longue.
                # Si on ouvre une nouvelle position short ou on augmente une position short existante:
                elif order_type == 'sell' and (current_position.quantity == 0 or (current_position.quantity < 0 and quantity_diff < 0) or (current_position.quantity > 0 and abs(quantity_diff) > current_position.quantity)): # Ouverture ou augmentation Short, ou flip de Long vers Short
                     current_position.entry_price = ( (abs(current_position.quantity) * current_position.entry_price) + (abs(quantity_diff) * current_price) ) / abs(new_quantity_signed) if new_quantity_signed !=0 else current_price


                current_position.quantity = new_quantity_signed
                current_position.timestamp = timestamp
                current_position.current_price = current_price # Mettre à jour le prix actuel
                # Recalculer les valeurs de la position après la transaction
                current_position.__post_init__() # Pour recalculer notional_value, unrealized_pnl, etc.
                # Assurer que position_value est la valeur notionnelle pour les futures
                if asset.is_future:
                    current_position.position_value = current_position.notional_value


        # Mettre à jour les poids cibles et la date de rééquilibrage
        self.target_weights = target_allocation.copy()
        self.last_rebalance_date = timestamp
        
        # Enregistrer l'historique de rééquilibrage
        self.rebalancing_history.append({
            'timestamp': timestamp,
            'target_allocation': target_allocation.copy(),
            'orders': orders,
            'total_transaction_costs': sum(order['transaction_cost'] for order in orders)
        })
        
        self.logger.info(f"Rééquilibrage exécuté: {len(orders)} ordres, "
                        f"coût total: {sum(order['transaction_cost'] for order in orders):.2f}")
        
        # Recalculer les métriques du portefeuille après toutes les transactions
        self.calculate_portfolio_metrics()
        return orders
    
    def update_portfolio(self, 
                        current_prices: Dict[str, float],
                        timestamp: datetime,
                        signals: Dict[str, float] = None) -> Dict:
        """
        Met à jour le portefeuille avec les prix actuels et rééquilibre si nécessaire.
        
        Args:
            current_prices: Prix actuels des actifs
            timestamp: Timestamp de la mise à jour
            signals: Signaux de trading optionnels (pour Black-Litterman par exemple)
            
        Returns:
            Dict avec les métriques du portefeuille après mise à jour et éventuel rééquilibrage.
        """
        # 1. Mettre à jour les données de prix historiques pour chaque actif
        for symbol, price in current_prices.items():
            if symbol in self.price_data:
                new_row_data = {'timestamp': timestamp, 'close': price}
                # Pour la simplicité, on peut omettre open, high, low, volume si non essentiels ici
                # ou les mettre égaux à 'close' et un volume par défaut.
                for col in ['open', 'high', 'low']:
                    if col not in self.price_data[symbol].columns:
                         self.price_data[symbol][col] = price # Ajoute la colonne si elle manque
                    new_row_data[col] = price
                if 'volume' not in self.price_data[symbol].columns:
                    self.price_data[symbol]['volume'] = 0
                new_row_data['volume'] = self.price_data[symbol]['volume'].iloc[-1] if len(self.price_data[symbol]) > 0 else 0 # Placeholder

                new_row_df = pd.DataFrame([new_row_data])
                # S'assurer que les colonnes correspondent avant concaténation
                if not all(col in self.price_data[symbol].columns for col in new_row_df.columns):
                    # Gérer le cas où les colonnes ne correspondent pas parfaitement (rare si bien initialisé)
                    # Pour l'instant, on suppose qu'elles correspondent ou que concat gère avec NaN
                    pass
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_row_df], ignore_index=True)

        # 2. Mettre à jour le prix actuel de toutes les positions existantes
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                # Recalculer P&L, valeur notionnelle, etc. en utilisant __post_init__ ou une méthode dédiée
                # Cela est important car __post_init__ utilise self.current_price
                asset = self.assets.get(symbol)
                if asset: # S'assurer que l'asset existe pour récupérer contract_multiplier
                    position.contract_multiplier = asset.contract_multiplier
                position.__post_init__() # Recalcule notional_value, unrealized_pnl, etc.
                if asset and asset.is_future: # Assurer la convention pour position_value
                    position.position_value = position.notional_value


        # 3. Calculer les métriques du portefeuille AVANT rééquilibrage (pour la décision de rééquilibrer)
        metrics_before_rebalancing = self.calculate_portfolio_metrics()
        
        # 4. Vérifier si un rééquilibrage est nécessaire
        rebalancing_executed_flag = False
        rebalancing_orders_executed = []
        if self.should_rebalance(timestamp):
            self.logger.info(f"Rééquilibrage nécessaire à {timestamp}.")
            # Intégrer les signaux dans l'optimisation si disponibles (pour Black-Litterman)
            # L'optimiseur a besoin des indices des actifs, pas des symboles directement pour les vues.
            # Créer un mapping symbol -> index pour les vues.
            symbol_to_idx = {s: i for i, s in enumerate(self.assets.keys())}
            bl_views = None
            if signals and self.optimization_method == 'black_litterman':
                bl_views = {symbol_to_idx[s]: v for s, v in signals.items() if s in symbol_to_idx}
            
            # Optimiser l'allocation
            optimal_allocation = self.optimize_portfolio(views=bl_views) # Utilise la méthode stockée dans self.optimization_method
            
            # Exécuter le rééquilibrage
            rebalancing_orders_executed = self.execute_rebalancing(optimal_allocation, current_prices, timestamp)
            rebalancing_executed_flag = True
            
            # Les métriques seront recalculées après le rééquilibrage (fait à la fin de execute_rebalancing)
            # et de nouveau ci-dessous pour le snapshot.
        
        # 5. Calculer/Recalculer les métriques finales du portefeuille après toute action
        final_metrics = self.calculate_portfolio_metrics()
        if rebalancing_executed_flag:
            final_metrics['rebalancing_executed'] = True
            final_metrics['rebalancing_orders'] = rebalancing_orders_executed
        else:
            final_metrics['rebalancing_executed'] = False
        
        # 6. Ajouter à l'historique du portefeuille
        portfolio_snapshot = {
            'timestamp': timestamp,
            'total_value': final_metrics['total_value'], # C'est l'équité
            'total_unrealized_pnl': final_metrics['total_unrealized_pnl'],
            'total_unrealized_pnl_pct': final_metrics['total_unrealized_pnl_pct'],
            'num_positions': final_metrics['num_positions'],
            'current_weights': final_metrics['current_weights'],
            'target_weights': self.target_weights.copy(), # Les poids cibles après optimisation
            'transaction_costs_paid': self.transaction_costs_paid, # Cumulatif
            'total_notional_value': final_metrics.get('total_notional_value', 0.0),
            'total_used_margin': final_metrics.get('total_used_margin', 0.0),
            'available_capital': final_metrics.get('available_capital', self.current_capital)
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        return final_metrics
    
    def get_portfolio_summary(self) -> Dict:
        """
        Génère un résumé complet du portefeuille
        
        Returns:
            Dict avec le résumé du portefeuille
        """
        if not self.portfolio_history:
            return {
                'initial_capital': self.initial_capital,
                'current_value': self.current_capital, # Devrait être l'équité actuelle
                'total_return': 0.0,
                'num_rebalancings': 0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility_annualized': 0.0
            }
        
        latest_snapshot = self.portfolio_history[-1]
        current_equity = latest_snapshot['total_value']
        total_return = (current_equity - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # Calculer les métriques de performance
        portfolio_returns = self.calculate_portfolio_returns() # Basé sur 'total_value' (équité) de l'historique
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': latest_snapshot['num_positions'],
            'num_rebalancings': len(self.rebalancing_history),
            'total_transaction_costs': self.transaction_costs_paid,
            'current_weights': latest_snapshot['current_weights'],
            'target_weights': latest_snapshot['target_weights'],
            'total_notional_value': latest_snapshot.get('total_notional_value', 0.0),
            'total_used_margin': latest_snapshot.get('total_used_margin', 0.0),
            'available_capital': latest_snapshot.get('available_capital', self.current_capital)
        }
        
        if len(portfolio_returns) > 1:
            volatility_ann = portfolio_returns.std() * np.sqrt(252)
            sharpe = (portfolio_returns.mean() * 252) / volatility_ann if volatility_ann > 0 else 0.0
            summary.update({
                'volatility_annualized': volatility_ann,
                'sharpe_ratio': sharpe,
                'max_drawdown': self.calculate_max_drawdown(), # Basé sur l'équité
                'avg_daily_return': portfolio_returns.mean(),
                'win_rate': (portfolio_returns > 0).mean() if len(portfolio_returns) > 0 else 0.0
            })
        else:
            summary.update({
                'volatility_annualized': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_daily_return': 0.0,
                'win_rate': 0.0
            })

        return summary
    
    def calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum sur la base de l'équité du portefeuille"""
        if not self.portfolio_history:
            return 0.0
        
        equity_values = [entry['total_value'] for entry in self.portfolio_history]
        if not equity_values:
            return 0.0
            
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0 # Éviter division par zéro si peak est 0
            max_dd = max(max_dd, dd)
        
        return max_dd

    def get_assets(self) -> Dict[str, Asset]:
        """Return dictionary of all assets in the portfolio"""
        return self.assets.copy()
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics based on historical equity returns"""
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return {
                'portfolio_volatility_annualized': 0.0,
                'portfolio_var_95_daily': 0.0,
                'portfolio_var_99_daily': 0.0,
                # Beta et corrélation nécessiteraient des données de marché de référence
            }
        
        # Calculate portfolio returns based on 'total_value' (equity)
        returns = self.calculate_portfolio_returns()
        
        if len(returns) < 2: # Encore une vérification après calcul des retours
             return {
                'portfolio_volatility_annualized': 0.0,
                'portfolio_var_95_daily': 0.0,
                'portfolio_var_99_daily': 0.0,
            }
        
        # Calculate volatility (annualized)
        volatility_annualized = returns.std() * np.sqrt(252)  # Assume daily returns
        
        # Calculate Value at Risk (VaR) - daily
        var_95_daily = returns.quantile(0.05)
        var_99_daily = returns.quantile(0.01)
        
        return {
            'portfolio_volatility_annualized': volatility_annualized,
            'portfolio_var_95_daily': var_95_daily,
            'portfolio_var_99_daily': var_99_daily,
        }
    
    def calculate_portfolio_performance(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """
        Calculate portfolio performance metrics over a specified period.
        If start_date or end_date is None, uses the full history.
        """
        if not self.portfolio_history:
            return {
                'total_return_period': 0.0,
                'sharpe_ratio_annualized_period': 0.0,
                'max_drawdown_period': 0.0,
                'volatility_annualized_period': 0.0,
            }

        # Filtrer l'historique du portefeuille pour la période donnée
        history_df = pd.DataFrame(self.portfolio_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        if start_date:
            history_df = history_df[history_df['timestamp'] >= start_date]
        if end_date:
            history_df = history_df[history_df['timestamp'] <= end_date]

        if history_df.empty or len(history_df) < 2:
            return {
                'total_return_period': 0.0,
                'sharpe_ratio_annualized_period': 0.0,
                'max_drawdown_period': 0.0,
                'volatility_annualized_period': 0.0,
            }

        period_returns = history_df['total_value'].pct_change().dropna()

        if period_returns.empty or len(period_returns) < 1: # Besoin d'au moins un retour pour certaines métriques
            return {
                'total_return_period': 0.0, # (Valeur finale / Valeur initiale) - 1
                'sharpe_ratio_annualized_period': 0.0,
                'max_drawdown_period': 0.0,
                'volatility_annualized_period': 0.0,
            }
        
        total_return_period = (history_df['total_value'].iloc[-1] / history_df['total_value'].iloc[0]) - 1 if len(history_df['total_value']) > 0 else 0.0
        
        volatility_period_annualized = 0.0
        sharpe_ratio_period_annualized = 0.0
        if len(period_returns) > 1: # Nécessite au moins 2 retours pour std dev
            volatility_period_annualized = period_returns.std() * np.sqrt(252)  # Annualized
            mean_return_period_annualized = period_returns.mean() * 252  # Annualized
            sharpe_ratio_period_annualized = mean_return_period_annualized / volatility_period_annualized if volatility_period_annualized > 0 else 0.0
        
        # Max drawdown pour la période
        period_equity_values = history_df['total_value'].tolist()
        peak_period = period_equity_values[0]
        max_dd_period = 0.0
        for value_p in period_equity_values:
            if value_p > peak_period:
                peak_period = value_p
            dd_p = (peak_period - value_p) / peak_period if peak_period > 0 else 0.0
            max_dd_period = max(max_dd_period, dd_p)
        
        return {
            'total_return_period': total_return_period,
            'sharpe_ratio_annualized_period': sharpe_ratio_period_annualized,
            'max_drawdown_period': max_dd_period,
            'volatility_annualized_period': volatility_period_annualized,
        }
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        if not self.price_data:
            return pd.DataFrame()
        
        # Combine all price data
        # On s'assure que les données de prix sont bien des DataFrames avec une colonne 'close'
        all_closes = {}
        for symbol, data_df in self.price_data.items():
            if isinstance(data_df, pd.DataFrame) and 'close' in data_df.columns:
                # S'assurer que l'index est un datetime pour un alignement correct
                if not isinstance(data_df.index, pd.DatetimeIndex) and 'timestamp' in data_df.columns:
                    data_df_indexed = data_df.set_index(pd.to_datetime(data_df['timestamp']))
                    all_closes[symbol] = data_df_indexed['close']
                elif isinstance(data_df.index, pd.DatetimeIndex):
                     all_closes[symbol] = data_df['close']
                else:
                    self.logger.warning(f"Données de prix pour {symbol} mal formatées pour calcul de corrélation.")
                    continue # Passer au suivant si les données ne sont pas bonnes

        if not all_closes:
            return pd.DataFrame()

        price_df = pd.DataFrame(all_closes).dropna(how='all') # Drop rows where all symbols are NaN
        
        # Calculate correlation matrix
        returns_df = price_df.pct_change().dropna(how='all') # Drop rows of NaNs from pct_change
        
        # Gérer le cas où il y a moins de 2 lignes de rendements après dropna
        if len(returns_df) < 2:
            return pd.DataFrame(index=price_df.columns, columns=price_df.columns) # Retourner une matrice vide avec les bons labels

        return returns_df.corr()
    
    def check_rebalancing_conditions(self, current_date: datetime) -> bool: # Ajout de current_date
        """Check if portfolio needs rebalancing based on current_date"""
        if not self.positions and not self.target_weights: # Si pas de positions et pas de cibles, pas besoin
            return False

        if not self.target_weights: # Si pas de poids cibles, pas de rééquilibrage vers une cible
            self.logger.info("Aucun poids cible défini, pas de rééquilibrage basé sur la dérive.")
            # On pourrait toujours rééquilibrer sur base calendaire si souhaité, même sans cible spécifique (ex: pour ré-optimiser)
            # Pour l'instant, on considère que sans target_weights, le rebalancement par dérive n'a pas de sens.

        # Vérification calendaire en premier si c'est la méthode
        if self.rebalancing_method == RebalancingMethod.CALENDAR:
            if self.last_rebalance_date is None: return True # Premier rééquilibrage
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            if days_since_rebalance >= self.rebalancing_frequency_days:
                return True
        
        # Vérification par seuil de dérive (peut être combiné avec calendaire ou être la seule méthode)
        # Si la méthode est THRESHOLD, ou si CALENDAR a déjà été vérifié et n'a pas déclenché.
        if self.rebalancing_method == RebalancingMethod.THRESHOLD or self.rebalancing_method == RebalancingMethod.CALENDAR:
            # Recalculer les métriques pour avoir les poids actuels à jour
            metrics = self.calculate_portfolio_metrics()
            current_weights = metrics.get('current_weights', {})
            
            if not self.target_weights and not current_weights : # Si pas de cibles et pas de poids actuels, rien à comparer
                return False
            if not self.target_weights and current_weights: # Si on a des positions mais pas de cible, on pourrait vouloir liquider (hors scope ici) ou optimiser
                 self.logger.info("Positions actuelles sans poids cibles. Rééquilibrage pour optimisation pourrait être nécessaire.")
                 return True # Déclencher pour permettre une ré-optimisation

            max_drift = 0.0
            # Comparer les poids actuels aux poids cibles
            all_symbols_to_check = set(current_weights.keys()) | set(self.target_weights.keys())

            for symbol in all_symbols_to_check:
                current_w = current_weights.get(symbol, 0.0)
                target_w = self.target_weights.get(symbol, 0.0)
                
                if target_w > 1e-9: # Si le poids cible est significatif
                    drift = abs(current_w - target_w) / target_w
                elif current_w > 1e-9: # Si le poids cible est nul (ou très petit) mais qu'on a une position
                    drift = 1.0 # Dérive maximale, on devrait vendre
                else: # Poids actuel et cible sont tous deux (presque) nuls
                    drift = 0.0
                max_drift = max(max_drift, drift)
            
            if max_drift > self.drift_threshold:
                return True
        
        # Vérification par volatilité (si c'est la méthode principale ou en plus)
        if self.rebalancing_method == RebalancingMethod.VOLATILITY:
            portfolio_returns = self.calculate_portfolio_returns()
            if len(portfolio_returns) >= self.volatility_lookback:
                recent_vol = portfolio_returns.tail(self.volatility_lookback).std() * np.sqrt(252)
                if recent_vol > 0.3:  # Seuil de volatilité arbitraire
                    return True
        
        return False # Si aucune condition n'est remplie
    
    def rebalance_portfolio(self, current_date: datetime, current_prices: Dict[str, float]): # Ajout current_date et current_prices
        """
        Rebalance portfolio to target weights if conditions are met.
        This method orchestrates the rebalancing.
        """
        if self.should_rebalance(current_date):
            self.logger.info(f"Rééquilibrage déclenché à {current_date}.")
            
            # L'optimiseur a besoin des indices des actifs, pas des symboles directement pour les vues.
            symbol_to_idx = {s: i for i, s in enumerate(self.assets.keys())}
            bl_views = None
            # Si on a des signaux et que la méthode est Black-Litterman, on les prépare
            # Pour l'instant, les signaux ne sont pas passés à cette méthode, donc views sera None.
            # Il faudrait une source de signaux pour BL.
            
            optimal_allocation = self.optimize_portfolio(views=bl_views) # Utilise la méthode stockée dans self.optimization_method
            
            if not optimal_allocation:
                self.logger.warning("L'optimisation n'a pas retourné d'allocation. Rééquilibrage annulé.")
                return []

            rebalancing_orders = self.execute_rebalancing(optimal_allocation, current_prices, current_date)
            
            # Mettre à jour le snapshot du portefeuille après rééquilibrage
            final_metrics = self.calculate_portfolio_metrics() # Recalcule avec les nouvelles positions/capital
            portfolio_snapshot = {
                'timestamp': current_date,
                'total_value': final_metrics['total_value'],
                'total_unrealized_pnl': final_metrics['total_unrealized_pnl'],
                'total_unrealized_pnl_pct': final_metrics['total_unrealized_pnl_pct'],
                'num_positions': final_metrics['num_positions'],
                'current_weights': final_metrics['current_weights'],
                'target_weights': self.target_weights.copy(),
                'transaction_costs_paid': self.transaction_costs_paid,
                'total_notional_value': final_metrics.get('total_notional_value', 0.0),
                'total_used_margin': final_metrics.get('total_used_margin', 0.0),
                'available_capital': final_metrics.get('available_capital', self.current_capital),
                'rebalancing_executed': True,
                'rebalancing_orders': rebalancing_orders
            }
            if self.portfolio_history and self.portfolio_history[-1]['timestamp'] == current_date:
                self.portfolio_history[-1] = portfolio_snapshot # Mettre à jour le snapshot existant pour ce timestamp
            else:
                 self.portfolio_history.append(portfolio_snapshot)
            
            return rebalancing_orders
        else:
            self.logger.info(f"Aucun rééquilibrage nécessaire à {current_date}.")
            # S'assurer qu'un snapshot est quand même pris si update_portfolio n'est pas appelé séparément
            # Normalement, update_portfolio gère cela.
            return []
