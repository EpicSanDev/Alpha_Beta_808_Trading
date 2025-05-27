import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import datetime as dt

# Ajout du répertoire src au sys.path pour permettre les imports directs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Étape 1: Acquisition et Prétraitement des Données
from acquisition.connectors import generate_random_market_data, load_binance_klines # Ajout de load_binance_klines
from acquisition.preprocessing import handle_missing_values, normalize_min_max
from acquisition.preprocessing_utils import handle_missing_values_column

# Étape 2: Feature Engineering
from feature_engineering.technical_features import calculate_sma, calculate_ema, calculate_rsi

# Étape 3: Modélisation Prédictive
from modeling.models import prepare_data_for_model, train_model, load_model_and_predict

# Étape 4: Génération de Signaux et Allocation
from signal_generation.signal_generator import generate_base_signals_from_predictions, allocate_capital_simple

# Étape 5: Gestion des Risques
from risk_management.risk_controls import check_position_limit

# Étape 6: Exécution (Simulation)
from execution.simulator import BacktestSimulator

def run_backtest():
    """
    Orchestre un backtest simple en utilisant les modules du projet.
    """
    print("Démarrage du backtest simple...")
    load_dotenv() # Charger les variables d'environnement depuis .env

    # --- Étape 1: Acquisition et Prétraitement des Données ---
    print("\nÉtape 1: Acquisition et Prétraitement des Données")

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("ERREUR: Les clés API Binance (BINANCE_API_KEY, BINANCE_API_SECRET) ne sont pas configurées dans les variables d'environnement.")
        print("Veuillez créer un fichier .env à la racine du projet avec vos clés ou les définir autrement.")
        # Fallback sur les données aléatoires si les clés ne sont pas trouvées pour permettre au script de tourner
        print("Utilisation des données aléatoires générées par défaut.")
        market_data_df = generate_random_market_data(num_rows=200, start_price=100.0, volatility=0.01, freq='D')
        market_data_df['symbol'] = "ALPHA808_RANDOM"
    else:
        symbol_to_load = 'BTCUSDC'
        intervals_to_load = ['1d', '4h', '30m']
        # Calculer la date de début pour les 3 dernières années
        start_date = (datetime.now(dt.timezone.utc) - timedelta(days=3*365)).strftime("%Y-%m-%d")
        
        all_market_data = load_binance_klines(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol_to_load,
            intervals=intervals_to_load,
            start_date_str=start_date
        )

        if not all_market_data or '1d' not in all_market_data: # On se base sur '1d' pour la suite du MVP
            print(f"ERREUR: Impossible de charger les données Binance pour {symbol_to_load} sur l'intervalle '1d'.")
            print("Vérifiez votre connexion, vos clés API, et la disponibilité de la paire/intervalle.")
            print("Utilisation des données aléatoires générées par défaut.")
            market_data_df = generate_random_market_data(num_rows=200, start_price=100.0, volatility=0.01, freq='D')
            market_data_df['symbol'] = "ALPHA808_RANDOM"
        else:
            # Pour ce MVP, nous utilisons l'intervalle '1d' pour la suite du pipeline.
            # Les autres intervalles (4h, 30m) sont chargés mais pas encore utilisés ici.
            market_data_df = all_market_data['1d'].copy() # Utiliser une copie pour éviter les SettingWithCopyWarning
            print(f"Données de marché pour {symbol_to_load} (intervalle '1d') chargées depuis Binance:\n{market_data_df.head()}")
            # La colonne 'symbol' est déjà ajoutée par load_binance_klines

    if market_data_df.empty:
        print("Aucune donnée de marché à traiter. Arrêt du backtest.")
        return

    # Simuler quelques valeurs manquantes pour le test si on utilise les données aléatoires
    if 'ALPHA808_RANDOM' in market_data_df['symbol'].unique():
         market_data_df.loc[market_data_df.sample(frac=0.05).index, 'close'] = np.nan

    # Gestion des valeurs manquantes (s'applique aux données Binance ou aléatoires)
    # S'assurer que la colonne 'close' existe avant de tenter de traiter les NaN
    if 'close' in market_data_df.columns:
        market_data_df = handle_missing_values_column(market_data_df, column='close', strategy='ffill')
        print(f"\nDonnées après gestion des valeurs manquantes (close):\n{market_data_df.head()}")
        market_data_df = normalize_min_max(market_data_df, column='close')
        print(f"\nDonnées après normalisation (close):\n{market_data_df.head()}")
    else:
        print("AVERTISSEMENT: Colonne 'close' non trouvée, le prétraitement sur 'close' est sauté.")


    # --- Étape 2: Feature Engineering ---
    print("\nÉtape 2: Feature Engineering")
    # S'assurer que la colonne 'close' existe avant de calculer les features
    if 'close' in market_data_df.columns and not market_data_df['close'].isnull().all():
        # Features de base
        market_data_df = calculate_sma(market_data_df, column='close', windows=[10, 20])
        market_data_df = calculate_ema(market_data_df, column='close', windows=[10, 20])
        market_data_df = calculate_rsi(market_data_df, column='close', window=14)
        
        # Import des nouvelles fonctions de features
        from src.feature_engineering.technical_features import (
            calculate_macd, calculate_bollinger_bands, 
            calculate_price_momentum, calculate_volume_features
        )
        
        # Features avancées
        market_data_df = calculate_macd(market_data_df, column='close')
        market_data_df = calculate_bollinger_bands(market_data_df, column='close')
        market_data_df = calculate_price_momentum(market_data_df, column='close', windows=[5, 10])
        
        # Features de volume si disponible
        if 'volume' in market_data_df.columns:
            market_data_df = calculate_volume_features(market_data_df, volume_col='volume', price_col='close', windows=[10, 20])
        
        market_data_df.dropna(inplace=True)
        market_data_df.reset_index(drop=True, inplace=True) 
        print(f"Données après ajout des features techniques:\n{market_data_df.head()}")
    else:
        print("AVERTISSEMENT: Colonne 'close' non trouvée ou toutes les valeurs sont NaN. Feature engineering sauté.")
        # Si le feature engineering est sauté, les colonnes de features n'existeront pas.
        # Il faut s'assurer que le reste du script peut gérer cela ou s'arrêter.
        print("Arrêt du backtest car les features n'ont pas pu être générées.")
        return

    if market_data_df.empty:
        print("Aucune donnée après feature engineering. Arrêt du backtest.")
        return
        
    # --- Étape 3: Modélisation Prédictive ---
    print("\nÉtape 3: Modélisation Prédictive")
    
    # Utiliser tous les features techniques disponibles (20 features au lieu de 5)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
    all_feature_cols = [col for col in market_data_df.columns if col not in exclude_cols]
    print(f"Features techniques disponibles: {len(all_feature_cols)} features")
    print(f"Features: {all_feature_cols}")
    
    # Utiliser la nouvelle fonction prepare_data_for_model avec tous les features
    X, y = prepare_data_for_model(
        market_data_df, 
        target_shift_days=1, 
        feature_columns=all_feature_cols,  # Utiliser tous les features au lieu du défaut
        price_change_threshold=0.01  # Seuil de 1% pour plus de sensibilité
    )
    
    if len(X) == 0:
        print("ERREUR: Aucune donnée disponible pour l'entraînement après préparation.")
        return
    
    print(f"Dimensions de X: {X.shape}, Dimensions de y: {y.shape}")
    
    # TODO: VAL-001 - Validation Out-of-Sample Rigoureuse
    # La séparation actuelle est un simple split chronologique.
    # Pour une validation rigoureuse, envisager une séparation stricte d'un ensemble de test final
    # qui ne sera JAMAIS utilisé pendant l'entraînement ou l'optimisation des hyperparamètres.
    # Ce test final devrait simuler le déploiement en conditions réelles.
    # Les données X_predict ici servent de "walk-forward" ou de test continu,
    # mais un VRAI ensemble "out-of-time" final est crucial.

    # Split en train/predict pour la simulation
    split_ratio = 0.8 # Pourrait être ajusté pour garder plus de données pour un test final séparé.
    split_index = int(len(X) * split_ratio)
    if split_index == 0 and len(X) > 0 : # Cas où il y a peu de données, assurer au moins 1 pour train
        split_index = 1
    if split_index >= len(X): # Cas où toutes les données iraient en train
        if len(X) > 1:
            split_index = len(X) - 1
        else: # Pas assez de données pour split
            print("ERREUR: Pas assez de données pour séparer en train/predict. Arrêt.")
            return

    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_predict = X.iloc[split_index:] # Ceci est l'ensemble de "test" pour le backtest actuel.
    timestamps_predict = market_data_df.iloc[split_index:]['timestamp'].reset_index(drop=True)
    
    if X_train.empty or y_train.empty or X_predict.empty:
        print("ERREUR: Ensembles d'entraînement ou de prédiction vides après le split. Arrêt.")
        print(f"Len X: {len(X)}, split_index: {split_index}")
        return

    # Entraîner le modèle avec les nouvelles améliorations et tous les features
    model_save_path = 'models_store/logistic_regression_mvp.joblib'
    if not os.path.exists("models_store"):
        os.makedirs("models_store")
    
    metrics = train_model(
        X_train, y_train, 
        model_type='logistic_regression', 
        model_path=model_save_path,
        scale_features=True  # Activer la standardisation
    )
    
    print(f"Modèle 'logistic_regression' entraîné et sauvegardé dans {model_save_path}")
    
    # Générer les prédictions avec le nouveau modèle
    predictions = load_model_and_predict(X_predict, model_path=model_save_path, return_probabilities=True)
    print(f"Prédictions générées (premiers 5): {predictions[:5]}")
    print(f"Type de prédictions: {type(predictions)}, Range: min={predictions.min():.3f}, max={predictions.max():.3f}")
    
    # Afficher la distribution des prédictions
    prob_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("Distribution des prédictions générées:")
    for low, high in prob_ranges:
        count = np.sum((predictions >= low) & (predictions < high))
        print(f"  [{low:.1f}-{high:.1f}): {count} ({count/len(predictions)*100:.1f}%)")

    # --- Étape 4: Génération de Signaux et Allocation ---
    print("\nÉtape 4: Génération de Signaux et Allocation")
    
    # Maintenant on a directement des probabilités, plus besoin de conversion
    predictions_prob = predictions
    
    print(f"Prédictions probabilistes (premiers 5): {predictions_prob[:5]}")
    print(f"Range final: min={predictions_prob.min():.3f}, max={predictions_prob.max():.3f}")
    
    # Générer des signaux avec des seuils adaptés au range réel des prédictions
    signals = pd.Series(['HOLD'] * len(predictions_prob))
    
    # Seuils basés sur l'analyse du range réel des prédictions (0.236-0.650)
    # Utiliser des percentiles pour des seuils adaptatifs
    upper_threshold = np.percentile(predictions_prob, 80)  # Top 20% -> BUY
    lower_threshold = np.percentile(predictions_prob, 20)  # Bottom 20% -> SELL
    
    print(f"Seuils adaptatifs: BUY >= {upper_threshold:.3f}, SELL <= {lower_threshold:.3f}")
    
    signals[predictions_prob >= upper_threshold] = 'BUY'
    signals[predictions_prob <= lower_threshold] = 'SELL'
    
    print(f"Signaux générés avec seuils adaptatifs (premiers 5): {signals[:5]}")
    
    # Analyse des signaux pour optimiser l'allocation
    signal_counts = signals.value_counts()
    print(f"Distribution des signaux: {signal_counts.to_dict()}")

    total_capital = 100000
    # Optimiser la taille des positions pour éviter les problèmes de liquidité
    buy_signals_count = (signals == 'BUY').sum()
    sell_signals_count = (signals == 'SELL').sum()
    hold_signals_count = (signals == 'HOLD').sum()
    
    print(f"Répartition finale des signaux - BUY: {buy_signals_count}, SELL: {sell_signals_count}, HOLD: {hold_signals_count}")
    
    if buy_signals_count > 0:
        # Allocation plus conservative: maximum 70% du capital, réparti intelligemment
        total_buyable_capital = total_capital * 0.70
        max_allocation_per_trade = min(0.05, total_buyable_capital / buy_signals_count)  # Max 5% par trade
        risk_per_trade = max(0.01, max_allocation_per_trade)  # Minimum 1%
        
        print(f"Capital disponible pour BUY: ${total_buyable_capital:,.0f}")
        print(f"Allocation par trade: {risk_per_trade:.1%} (${risk_per_trade * total_capital:,.0f})")
    else:
        risk_per_trade = 0.01
    
    positions_to_allocate = allocate_capital_simple(signals, total_capital, risk_per_trade=risk_per_trade) 
    print(f"Positions à allouer optimisées (premiers 5): {positions_to_allocate[:5]}")
    print(f"Risk per trade ajusté: {risk_per_trade:.3f} ({risk_per_trade*100:.1f}%)") 
    print(f"Positions à allouer optimisées (premiers 5): {positions_to_allocate[:5]}")
    print(f"Risk per trade ajusté: {risk_per_trade:.3f} ({risk_per_trade*100:.1f}%)")

    # --- Étape 5: Gestion des Risques (Exemple) ---
    print("\nÉtape 5: Gestion des Risques (Exemple)")
    example_new_position_size = 5000
    example_current_exposure = 20000
    example_max_exposure_limit = 50000
    can_trade = check_position_limit(
        new_position_size=example_new_position_size,
        current_total_exposure=example_current_exposure,
        max_exposure_limit=example_max_exposure_limit
    )
    print(f"Exemple de vérification de limite de position: Peut trader ? {can_trade}")

    # --- Étape 6: Exécution (Simulation) ---
    print("\nÉtape 6: Exécution (Simulation)")
    
    if len(timestamps_predict) != len(signals):
        print(f"ERREUR: Incohérence de longueur entre timestamps_predict ({len(timestamps_predict)}) et signals ({len(signals)}).")
        print("Arrêt de la simulation.")
        return

    signals_df = pd.DataFrame({
        'timestamp': timestamps_predict.values,  # Use .values to get the underlying array
        'signal': signals, 
        'position_to_allocate': positions_to_allocate
    })
    signals_df.loc[signals_df['signal'] == 'HOLD', 'position_to_allocate'] = 0

    # Utiliser les données de marché correspondant aux mêmes timestamps que les prédictions
    # Récupérer les colonnes OHLC originales depuis market_data_df
    original_data_subset = market_data_df.iloc[split_index:].copy()
    
    # Créer market_data_for_simulation avec les bonnes colonnes
    market_data_for_simulation = pd.DataFrame({
        'timestamp': timestamps_predict.values,
        'open': original_data_subset['open'].values,
        'high': original_data_subset['high'].values, 
        'low': original_data_subset['low'].values,
        'close': original_data_subset['close'].values
    })
    
    # Assurer que les index sont réinitialisés pour un alignement correct avant le set_index
    market_data_for_simulation.reset_index(drop=True, inplace=True)
    signals_df.reset_index(drop=True, inplace=True)

    # S'assurer que les longueurs correspondent avant de tenter de définir l'index
    min_len = min(len(market_data_for_simulation), len(signals_df))
    market_data_for_simulation = market_data_for_simulation.head(min_len)
    signals_df = signals_df.head(min_len)

    if 'timestamp' not in market_data_for_simulation.columns or 'timestamp' not in signals_df.columns:
        print("ERREUR: Colonne 'timestamp' manquante dans market_data_for_simulation ou signals_df avant set_index.")
        return

    market_data_for_simulation = market_data_for_simulation.set_index('timestamp')
    signals_df = signals_df.set_index('timestamp')
    
    # Vérification finale de l'alignement des index après set_index
    if not market_data_for_simulation.index.equals(signals_df.index):
        # Utiliser l'intersection des index pour assurer l'alignement
        common_index = market_data_for_simulation.index.intersection(signals_df.index)
        market_data_for_simulation = market_data_for_simulation.loc[common_index]
        signals_df = signals_df.loc[common_index]
        if common_index.empty:
            print("ERREUR: Aucun index commun entre les données de marché et les signaux. Arrêt.")
            return

    simulator = BacktestSimulator(initial_capital=total_capital, market_data=market_data_for_simulation)
    simulator.run_simulation(signals_df)

    print("\n--- Résultats de la Simulation ---")
    portfolio_history = simulator.get_portfolio_history()
    trade_history = simulator.get_trades_history()

    print("\nHistorique de la valeur du portefeuille (dernières 5 entrées):")
    if not portfolio_history.empty:
        print(portfolio_history.tail())
    else:
        print("L'historique du portefeuille est vide.")

    print("\nHistorique des trades (dernières 5 entrées):")
    if not trade_history.empty:
        print(trade_history.tail())
    else:
        print("Aucun trade n'a été exécuté.")

    # --- Analyse de Performance ---
    print("\n--- Analyse de Performance ---")
    
    if not portfolio_history.empty:
        portfolio_df = pd.DataFrame(portfolio_history)
        
        # Calculs de performance
        initial_value = total_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculs de base
        days_traded = len(portfolio_df) - 1
        years_traded = days_traded / 365.25 if days_traded > 0 else 1
        annualized_return = (final_value / initial_value) ** (1/years_traded) - 1 if years_traded > 0 else 0
        
        # Buy and Hold de référence
        if not market_data_for_simulation.empty:
            initial_price = market_data_for_simulation['close'].iloc[0]
            final_price = market_data_for_simulation['close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            shares_buy_hold = initial_value / initial_price
            buy_hold_final_value = shares_buy_hold * final_price
        else:
            buy_hold_return = 0
            buy_hold_final_value = initial_value
        
        # Métriques de volatilité et drawdown
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        volatility = portfolio_df['daily_return'].std() * np.sqrt(365.25) if len(portfolio_df) > 1 else 0
        
        # Maximum Drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Sharpe Ratio (simplifié, sans taux sans risque)
        sharpe_ratio = (portfolio_df['daily_return'].mean() * 365.25) / (portfolio_df['daily_return'].std() * np.sqrt(365.25)) if portfolio_df['daily_return'].std() > 0 else 0
        
        # Statistiques des trades
        if not trade_history.empty:
            trade_df = pd.DataFrame(trade_history)
            total_trades = len(trade_df)
            buy_trades = len(trade_df[trade_df['type'] == 'BUY'])
            sell_trades = len(trade_df[trade_df['type'] == 'SELL'])
        else:
            total_trades = buy_trades = sell_trades = 0
        
        # Alpha vs Buy & Hold
        alpha = total_return - buy_hold_return
        
        # Affichage du rapport de performance
        print("\n" + "="*70)
        print("RAPPORT DE PERFORMANCE ALPHABETA808TRADING")
        print("="*70)
        print(f"Capital Initial:      ${initial_value:>12,.2f}")
        print(f"Valeur Finale:        ${final_value:>12,.2f}")
        print(f"Rendement Total:      {total_return*100:>12.2f}%")
        print(f"Rendement Annualisé:  {annualized_return*100:>12.2f}%")
        print(f"Volatilité:           {volatility*100:>12.2f}%")
        print(f"Sharpe Ratio:         {sharpe_ratio:>12.3f}")
        print(f"Maximum Drawdown:     {max_drawdown*100:>12.2f}%")
        print("-"*70)
        print("COMPARAISON vs BUY & HOLD:")
        print(f"Alpha:                {alpha*100:>12.2f}%")
        print(f"Buy & Hold Return:    {buy_hold_return*100:>12.2f}%")
        print(f"B&H Valeur finale:    ${buy_hold_final_value:>12,.2f}")
        print("-"*70)
        print(f"Jours de trading:     {days_traded:>12}")
        print(f"Nombre de trades:     {total_trades:>12}")
        print(f"Trades BUY:           {buy_trades:>12}")
        print(f"Trades SELL:          {sell_trades:>12}")
        
        # Position finale
        if not portfolio_df.empty:
            cash_final = portfolio_df['cash'].iloc[-1]
            shares_final = portfolio_df['current_shares'].iloc[-1]
            print("-"*70)
            print("POSITION FINALE:")
            print(f"  Cash:               ${cash_final:>12,.2f}")
            print(f"  Actions détenues:   {shares_final:>12.4f}")
        print("="*70)
            
        # Sauvegarder les résultats
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les données
        portfolio_df.to_csv(f"results/portfolio_history_{timestamp}.csv", index=False)
        if not trade_history.empty:
            pd.DataFrame(trade_history).to_csv(f"results/trades_history_{timestamp}.csv", index=False)
        signals_df.to_csv(f"results/signals_{timestamp}.csv")
        
        print(f"\nRésultats sauvegardés dans results/ avec timestamp {timestamp}")
        
        # --- Intégration de la Visualisation ---
        print("\n--- Génération des Visualisations ---")
        try:
            # Créer un DataFrame pour la visualisation
            backtest_results = {
                'portfolio_history': portfolio_df,
                'trade_history': pd.DataFrame(trade_history) if trade_history else pd.DataFrame(),
                'market_data': market_data_for_simulation,
                'signals': signals_df.reset_index(),
                'performance_metrics': {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'alpha': alpha,
                    'buy_hold_return': buy_hold_return
                }
            }
            
            # Importer et utiliser le script de visualisation
            try:
                sys.path.append('.')
                import visualize_results
                visualize_results.create_backtest_visualization()
                print(f"Graphiques de visualisation générés avec succès")
            except ImportError as e:
                print(f"Module de visualisation non disponible: {e}")
            except Exception as e:
                print(f"Erreur lors de la génération des graphiques: {e}")
                
        except Exception as e:
            print(f"Erreur lors de l'intégration de la visualisation: {e}")
        
    else:
        print("Impossible de calculer les métriques de performance - historique vide.")

    print("\nBacktest simple terminé.")

if __name__ == "__main__":
    run_backtest()
