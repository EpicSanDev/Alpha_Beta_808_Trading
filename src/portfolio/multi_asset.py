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
    commission_rate: float
    is_active: bool = True
    leverage: float = 1.0
    weight_constraint_min: float = 0.0
    weight_constraint_max: float = 1.0

@dataclass
class Position:
    """Représente une position dans le portefeuille"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime
    position_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

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
    
    def add_asset(self, symbol: str = None, price_data: pd.DataFrame = None, returns_data: pd.Series = None, asset: Asset = None):
        """
        Ajoute un actif au portefeuille
        
        Args:
            symbol: Symbol of the asset (if creating asset from parameters)
            price_data: Price data for the asset (if creating asset from parameters)  
            returns_data: Returns data for the asset (if creating asset from parameters)
            asset: Asset object (if providing complete asset)
        """
        if asset is not None:
            # Use provided Asset object
            self.assets[asset.symbol] = asset
            self.target_weights[asset.symbol] = 0.0
            self.current_weights[asset.symbol] = 0.0
            self.logger.info(f"Actif ajouté: {asset.symbol} ({asset.asset_class.value})")
        elif symbol is not None:
            # Create Asset from parameters
            new_asset = Asset(
                symbol=symbol,
                name=symbol,
                asset_class=AssetClass.CRYPTO,  # Default to crypto
                exchange="binance",
                base_currency=symbol[:-4] if len(symbol) > 4 else symbol,
                quote_currency=symbol[-4:] if len(symbol) > 4 else "USDT",
                min_trade_size=0.001,
                tick_size=0.01,
                commission_rate=0.001,
                is_active=True
            )
            self.assets[symbol] = new_asset
            self.target_weights[symbol] = 0.0
            self.current_weights[symbol] = 0.0
            
            # Store price and returns data if provided
            if price_data is not None:
                self.update_price_data(symbol, price_data)
            
            self.logger.info(f"Actif ajouté: {symbol} (crypto)")
        else:
            raise ValueError("Must provide either 'asset' object or 'symbol' parameter")
    
    def update_price_data(self, symbol: str, price_data: pd.DataFrame):
        """Met à jour les données de prix pour un actif"""
        self.price_data[symbol] = price_data.copy()
        
        # Mettre à jour le prix actuel des positions
        if symbol in self.positions:
            current_price = price_data['close'].iloc[-1]
            position = self.positions[symbol]
            position.current_price = current_price
            position.position_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calcule les métriques du portefeuille"""
        if not self.positions:
            return {
                'total_value': self.current_capital,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'num_positions': 0
            }
        
        total_value = sum(pos.position_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calculer les poids actuels
        for symbol, position in self.positions.items():
            self.current_weights[symbol] = position.position_value / total_value if total_value > 0 else 0.0
            position.weight = self.current_weights[symbol]
        
        return {
            'total_value': total_value,
            'total_pnl': total_unrealized_pnl,
            'total_pnl_pct': total_unrealized_pnl / self.initial_capital if self.initial_capital > 0 else 0.0,
            'num_positions': len(self.positions),
            'current_weights': self.current_weights.copy(),
            'target_weights': self.target_weights.copy()
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
            max_drift = 0.0
            for symbol in self.target_weights:
                current_weight = self.current_weights.get(symbol, 0.0)
                target_weight = self.target_weights[symbol]
                if target_weight > 0:
                    drift = abs(current_weight - target_weight) / target_weight
                    max_drift = max(max_drift, drift)
            
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
            return pd.Series()
        
        values = [entry['total_value'] for entry in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def execute_rebalancing(self, 
                          target_allocation: Dict[str, float],
                          current_prices: Dict[str, float],
                          timestamp: datetime) -> List[Dict]:
        """
        Exécute le rééquilibrage du portefeuille
        
        Args:
            target_allocation: Allocation cible
            current_prices: Prix actuels
            timestamp: Timestamp de l'exécution
            
        Returns:
            Liste des ordres de rééquilibrage
        """
        orders = []
        
        # Calculer la valeur totale actuelle du portefeuille
        total_value = sum(pos.position_value for pos in self.positions.values()) + self.current_capital
        
        # Calculer les nouvelles quantités cibles
        target_quantities = {}
        for symbol, target_weight in target_allocation.items():
            if symbol in current_prices:
                target_value = total_value * target_weight
                target_quantity = target_value / current_prices[symbol]
                target_quantities[symbol] = target_quantity
        
        # Générer les ordres de rééquilibrage
        for symbol in self.assets:
            current_quantity = self.positions.get(symbol, Position(
                symbol=symbol, quantity=0, entry_price=0, current_price=current_prices.get(symbol, 0),
                timestamp=timestamp, position_value=0, weight=0, unrealized_pnl=0, unrealized_pnl_pct=0
            )).quantity
            
            target_quantity = target_quantities.get(symbol, 0.0)
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) > self.assets[symbol].min_trade_size:
                order_type = 'buy' if quantity_diff > 0 else 'sell'
                order_quantity = abs(quantity_diff)
                order_value = order_quantity * current_prices.get(symbol, 0)
                
                # Frais de transaction
                transaction_cost = order_value * self.transaction_cost
                self.transaction_costs_paid += transaction_cost
                
                order = {
                    'symbol': symbol,
                    'type': order_type,
                    'quantity': order_quantity,
                    'price': current_prices.get(symbol, 0),
                    'value': order_value,
                    'transaction_cost': transaction_cost,
                    'timestamp': timestamp,
                    'reason': 'rebalancing'
                }
                
                orders.append(order)
                
                # Mettre à jour la position
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=0,
                        entry_price=current_prices.get(symbol, 0),
                        current_price=current_prices.get(symbol, 0),
                        timestamp=timestamp,
                        position_value=0,
                        weight=0,
                        unrealized_pnl=0,
                        unrealized_pnl_pct=0
                    )
                
                position = self.positions[symbol]
                
                if order_type == 'buy':
                    new_quantity = position.quantity + order_quantity
                    new_entry_price = ((position.quantity * position.entry_price + 
                                      order_quantity * current_prices.get(symbol, 0)) / 
                                     new_quantity) if new_quantity > 0 else current_prices.get(symbol, 0)
                    position.quantity = new_quantity
                    position.entry_price = new_entry_price
                else:  # sell
                    position.quantity = max(0, position.quantity - order_quantity)
                
                # Mettre à jour les valeurs de position
                position.current_price = current_prices.get(symbol, 0)
                position.position_value = position.quantity * position.current_price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                position.unrealized_pnl_pct = ((position.current_price - position.entry_price) / 
                                             position.entry_price) if position.entry_price > 0 else 0
        
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
        
        return orders
    
    def update_portfolio(self, 
                        current_prices: Dict[str, float],
                        timestamp: datetime,
                        signals: Dict[str, float] = None) -> Dict:
        """
        Met à jour le portefeuille avec les prix actuels
        
        Args:
            current_prices: Prix actuels des actifs
            timestamp: Timestamp de la mise à jour
            signals: Signaux de trading optionnels
            
        Returns:
            Dict avec les métriques du portefeuille
        """
        # Mettre à jour les prix des positions
        for symbol, price in current_prices.items():
            if symbol in self.price_data:
                # Ajouter le nouveau prix aux données historiques
                new_row = pd.DataFrame({
                    'timestamp': [timestamp],
                    'close': [price],
                    'high': [price],  # Simplification
                    'low': [price],
                    'volume': [1000]  # Valeur par défaut
                })
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_row], ignore_index=True)
        
        # Calculer les métriques actuelles
        metrics = self.calculate_portfolio_metrics()
        
        # Vérifier si un rééquilibrage est nécessaire
        if self.should_rebalance(timestamp):
            # Intégrer les signaux dans l'optimisation si disponibles
            views = None
            if signals:
                views = {i: signal for i, (symbol, signal) in enumerate(signals.items()) 
                        if symbol in self.assets}
            
            # Optimiser l'allocation
            optimal_allocation = self.optimize_portfolio(views=views)
            
            # Exécuter le rééquilibrage
            rebalancing_orders = self.execute_rebalancing(optimal_allocation, current_prices, timestamp)
            
            # Recalculer les métriques après rééquilibrage
            metrics = self.calculate_portfolio_metrics()
            metrics['rebalancing_executed'] = True
            metrics['rebalancing_orders'] = rebalancing_orders
        else:
            metrics['rebalancing_executed'] = False
        
        # Ajouter à l'historique
        portfolio_snapshot = {
            'timestamp': timestamp,
            'total_value': metrics['total_value'],
            'total_pnl': metrics['total_pnl'],
            'total_pnl_pct': metrics['total_pnl_pct'],
            'num_positions': metrics['num_positions'],
            'current_weights': metrics['current_weights'],
            'target_weights': metrics['target_weights'],
            'transaction_costs_paid': self.transaction_costs_paid
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        return metrics
    
    def get_portfolio_summary(self) -> Dict:
        """
        Génère un résumé complet du portefeuille
        
        Returns:
            Dict avec le résumé du portefeuille
        """
        if not self.portfolio_history:
            return {
                'initial_capital': self.initial_capital,
                'current_value': self.current_capital,
                'total_return': 0.0,
                'num_rebalancings': 0
            }
        
        latest = self.portfolio_history[-1]
        total_return = (latest['total_value'] - self.initial_capital) / self.initial_capital
        
        # Calculer les métriques de performance
        portfolio_returns = self.calculate_portfolio_returns()
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': latest['total_value'],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': latest['num_positions'],
            'num_rebalancings': len(self.rebalancing_history),
            'total_transaction_costs': self.transaction_costs_paid,
            'current_weights': latest['current_weights'],
            'target_weights': latest['target_weights']
        }
        
        if len(portfolio_returns) > 1:
            summary.update({
                'volatility_annualized': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown(),
                'avg_return': portfolio_returns.mean(),
                'win_rate': (portfolio_returns > 0).mean()
            })
        
        return summary
    
    def calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum"""
        if not self.portfolio_history:
            return 0.0
        
        values = [entry['total_value'] for entry in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

    def get_assets(self) -> Dict[str, Asset]:
        """Return dictionary of all assets in the portfolio"""
        return self.assets.copy()
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics"""
        if not self.positions:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var_95': 0.0,
                'portfolio_var_99': 0.0,
                'beta': 0.0,
                'correlation_with_market': 0.0
            }
        
        # Calculate portfolio returns
        returns = self.calculate_portfolio_returns()
        
        if len(returns) < 2:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var_95': 0.0,
                'portfolio_var_99': 0.0,
                'beta': 0.0,
                'correlation_with_market': 0.0
            }
        
        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(252)  # Assume daily returns
        
        # Calculate Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        return {
            'portfolio_volatility': volatility,
            'portfolio_var_95': var_95,
            'portfolio_var_99': var_99,
            'beta': 1.0,  # Placeholder - would need market data to calculate properly
            'correlation_with_market': 0.0  # Placeholder
        }
    
    def calculate_portfolio_performance(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate portfolio performance metrics over a period"""
        if not self.portfolio_history:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'alpha': 0.0,
                'beta': 1.0
            }
        
        # Get portfolio returns
        returns = self.calculate_portfolio_returns()
        
        if len(returns) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'alpha': 0.0,
                'beta': 1.0
            }
        
        # Calculate metrics
        total_return = (returns + 1).prod() - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming risk-free rate = 0)
        mean_return = returns.mean() * 252  # Annualized
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        # Max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'alpha': 0.0,  # Placeholder - would need benchmark to calculate
            'beta': 1.0   # Placeholder - would need benchmark to calculate
        }
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        if not self.price_data:
            return pd.DataFrame()
        
        # Combine all price data
        price_df = pd.DataFrame()
        for symbol, data in self.price_data.items():
            if hasattr(data, 'values'):
                price_df[symbol] = data.values
            else:
                price_df[symbol] = data
        
        # Calculate correlation matrix
        returns_df = price_df.pct_change().dropna()
        return returns_df.corr()
    
    def check_rebalancing_conditions(self) -> bool:
        """Check if portfolio needs rebalancing"""
        if not self.positions:
            return False
        
        # Check if enough time has passed since last rebalance
        if self.last_rebalance_date:
            days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
            if days_since_rebalance < self.rebalancing_frequency_days:
                return False
        
        # Check drift from target weights
        current_weights = self.current_weights
        target_weights = self.target_weights
        
        max_drift = 0.0
        for symbol in current_weights:
            if symbol in target_weights:
                drift = abs(current_weights[symbol] - target_weights[symbol])
                max_drift = max(max_drift, drift)
        
        return max_drift > self.drift_threshold
    
    def rebalance_portfolio(self):
        """Rebalance portfolio to target weights"""
        if not self.positions or not self.target_weights:
            return
        
        # Calculate new target allocations
        total_value = sum(pos.position_value for pos in self.positions.values())
        
        for symbol, target_weight in self.target_weights.items():
            target_value = total_value * target_weight
            
            if symbol in self.positions:
                current_value = self.positions[symbol].position_value
                difference = target_value - current_value
                
                if abs(difference) > total_value * 0.01:  # 1% threshold
                    # Record rebalancing action
                    self.rebalancing_history.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'buy' if difference > 0 else 'sell',
                        'amount': abs(difference),
                        'reason': 'rebalancing'
                    })
        
        self.last_rebalance_date = datetime.now()
        self.logger.info("Portfolio rebalanced successfully")
