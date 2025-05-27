# Plan d'Action : Intégration des Données Futures Bitget et Amélioration des Modèles

**Objectif Général** : Intégrer les données de contrats à terme (futures) de Bitget pour enrichir l'ingénierie de caractéristiques, améliorer la précision des modèles de trading, et adapter l'architecture du système pour supporter ces nouvelles données et stratégies.

## Phases du Projet

### Phase 1 : Acquisition et Préparation des Données de Futures (Priorité Haute)

1.  **Extension du `BitgetConnector` ([`src/acquisition/connectors.py:79`](src/acquisition/connectors.py:79))** :
    *   Modifier `BitgetConnector` pour récupérer toutes les données de futures disponibles via l'API Bitget :
        *   Open Interest (via `get_open_interest`)
        *   Funding Rates (historiques via `get_historical_funding_rates`, actuel via `get_current_funding_rate`)
        *   Mark Price (via `get_mark_price`), qui servira pour le calcul du Basis.
        *   Volume spécifique aux contrats à terme (déjà inclus dans les klines OHLCV via `base_volume`).
    *   Gérer la pagination et les limites de taux pour ces nouveaux endpoints (fait).
    *   Assurer la robustesse et la gestion des erreurs (fait via `_make_request` et gestion dans les méthodes).
    *   Ajout d'une méthode `fetch_and_store_all_metrics` dans `BitgetConnector` pour récupérer et stocker toutes les nouvelles métriques pour un symbole.

2.  **Intégration au Pipeline de Données** :
    *   Le `BitgetConnector` étendu inclut désormais la logique pour insérer les données dans la base de données via des sessions SQLAlchemy.
    *   Schéma de la base de données SQL (SQLite via SQLAlchemy, dans [`src/core/database.py`](src/core/database.py:1)) :
        *   Nouvelle table `open_interest` : `id`, `symbol` (TEXT), `timestamp` (DATETIME), `value` (FLOAT). Contrainte `UNIQUE` sur (`symbol`, `timestamp`).
        *   Nouvelle table `funding_rates` : `id`, `symbol` (TEXT), `timestamp` (DATETIME), `value` (FLOAT). Contrainte `UNIQUE` sur (`symbol`, `timestamp`).
        *   Nouvelle table `mark_prices` : `id`, `symbol` (TEXT), `timestamp` (DATETIME), `value` (FLOAT). Contrainte `UNIQUE` sur (`symbol`, `timestamp`).
    *   Un script [`src/acquisition/data_collector.py`](src/acquisition/data_collector.py:1) a été créé pour appeler `fetch_and_store_all_metrics` pour une liste de symboles, peuplant ainsi la base de données.

3.  **Prétraitement des Données de Futures** :
    *   Les fonctions suivantes ont été ajoutées/mises à jour dans [`src/acquisition/preprocessing.py`](src/acquisition/preprocessing.py:1) :
        *   `load_data_from_db`: Charge les données depuis les nouvelles tables SQL pour un symbole et une période donnés.
        *   `resample_and_align_data`: Rééchantillonne les données de métriques (Open Interest, Funding Rate, Mark Price) à la fréquence des klines, les aligne temporellement et gère les valeurs manquantes par interpolation (ex: `ffill`).
        *   `calculate_basis`: Calcule le "Basis" (différence `mark_price` - `close` des klines).
        *   `preprocess_new_metrics_for_symbol`: Fonction principale qui orchestre le chargement des klines et des nouvelles métriques depuis la BDD, leur alignement, le calcul du basis, et la fusion en un seul DataFrame par symbole.
    *   La gestion des valeurs manquantes initiales est effectuée par `ffill` (puis `bfill` pour les trous au début) lors de l'alignement.
    *   Les noms de colonnes (`open_interest`, `funding_rate`, `mark_price`, `basis`) et les types de données (timestamp en UTC, float pour les valeurs) sont normalisés.

### Phase 2 : Ingénierie de Caractéristiques (Feature Engineering) pour les Futures (Priorité Haute)

1.  **Identification et Calcul de Nouvelles Caractéristiques** :
    *   Dans [`src/feature_engineering/technical_features.py`](src/feature_engineering/technical_features.py) ou un nouveau module `src/feature_engineering/futures_features.py` :
        *   **Open Interest Features** :
            *   Open Interest brut.
            *   Variation de l'Open Interest (sur N périodes).
            *   Ratio Open Interest / Volume.
            *   Moyennes mobiles de l'Open Interest.
        *   **Funding Rate Features** :
            *   Funding Rate brut.
            *   Moyennes mobiles du Funding Rate.
            *   Indicateurs basés sur les changements de signe ou l'amplitude du Funding Rate (sentiment).
        *   **Basis Features** :
            *   Basis brut.
            *   Moyennes mobiles du Basis.
            *   Indicateurs de convergence/divergence du Basis par rapport au prix spot.
        *   **Volume des Futures Features** :
            *   Volume spécifique des contrats à terme (si différent du volume OHLCV).
            *   Indicateurs de volume avancés (similaires à ceux existants mais appliqués au volume des futures).
        *   **Caractéristiques Combinées** :
            *   Indicateurs de momentum combinant prix et open interest.
            *   Indicateurs de sentiment basés sur les funding rates et l'évolution du basis.
            *   Volatilité implicite (si calculable à partir des données disponibles ou d'options sur futures).

2.  **Intégration au Pipeline de Features Existant** :
    *   Assurer que les nouvelles fonctions de calcul de caractéristiques peuvent être appelées de manière transparente par le pipeline existant (notamment dans `_engineer_features` de [`src/backtesting/comprehensive_backtest.py:288`](src/backtesting/comprehensive_backtest.py:288)).
    *   Gérer l'alignement temporel des nouvelles caractéristiques avec les caractéristiques existantes.

### Phase 3 : Adaptation et Développement des Modèles (Priorité Moyenne à Haute)

1.  **Impact sur les Modèles Existants** ([`src/modeling/models.py`](src/modeling/models.py), [`src/modeling/xgboost_model.py`](src/modeling/xgboost_model.py)) :
    *   Analyser comment les nouvelles caractéristiques de futures peuvent être intégrées comme inputs aux modèles existants (XGBoost, LSTM, CNN).
    *   Ré-entraîner les modèles existants avec les jeux de données enrichis et évaluer l'amélioration de la performance.
    *   Porter une attention particulière à la normalisation et au scaling des nouvelles features.

2.  **Exploration de Nouveaux Types de Modèles** :
    *   Rechercher et prototyper des modèles spécifiquement conçus pour exploiter les dynamiques des marchés de futures et les régimes de marché :
        *   **Modèles basés sur les régimes de marché (Market Regime Models)** : Identifier différents états du marché (par exemple, haute/basse volatilité, tendance/range pour les futures) et entraîner des modèles spécifiques pour chaque régime ou utiliser le régime comme feature.
        *   **Modèles prenant en compte la structure à terme (Term Structure Models)** : Si des données sur plusieurs échéances de contrats sont disponibles, des modèles analysant la courbe des futures pourraient être envisagés.
        *   **Modèles d'Attention pour les Séries Temporelles** : Étendre les LSTM/CNN avec des mécanismes d'attention pour mieux capturer les dépendances temporelles et l'importance relative des différentes features (y compris celles des futures).
    *   Intégrer ces nouveaux modèles dans la structure de [`src/modeling/models.py`](src/modeling/models.py).

### Phase 4 : Stratégie d'Évaluation Améliorée (Priorité Moyenne)

1.  **Nouvelles Métriques d'Évaluation** ([`src/validation/metrics.py`](src/validation/metrics.py)) :
    *   Développer et intégrer des métriques spécifiques aux stratégies basées sur les futures :
        *   **Profit per Contract** : Bénéfice moyen réalisé par contrat tradé.
        *   **Skewness des Rendements des Futures** : Pour évaluer le risque de "fat tails".
        *   **Métriques liées au Levier** (si la simulation le permet) : Rendement sur capital engagé (Return on Margin), impact du levier sur la volatilité du portefeuille.
        *   **Fréquence et Coût du Roulement des Contrats (Rollover)** : Si la stratégie implique de tenir des positions sur plusieurs échéances.
    *   Intégrer ces métriques dans le `BacktestAnalyzer` ([`src/core/performance_analyzer.py`](src/core/performance_analyzer.py)) et les rapports de backtesting.

2.  **Scénarios de Test Spécifiques aux Futures** :
    *   Définir des scénarios de backtesting qui simulent des conditions de marché spécifiques aux futures (par exemple, périodes de forte volatilité, événements de liquidation en cascade, changements de funding rates importants).

### Phase 5 : Modifications Architecturales (Priorité Moyenne à Haute)

1.  **Pipeline de Données** : (Déjà couvert en Phase 1, point 2)
    *   Assurer que le pipeline de données de l'acquisition au feature engineering peut gérer le volume et la complexité des nouvelles données.

2.  **Backtesting** ([`src/backtesting/comprehensive_backtest.py`](src/backtesting/comprehensive_backtest.py)) :
    *   Modifier `_prepare_market_data` ([`src/backtesting/comprehensive_backtest.py:193`](src/backtesting/comprehensive_backtest.py:193)) pour permettre la sélection de `BitgetConnector` et la récupération des données de futures.
    *   Adapter `_engineer_features` ([`src/backtesting/comprehensive_backtest.py:288`](src/backtesting/comprehensive_backtest.py:288)) pour inclure les nouvelles caractéristiques de futures.
    *   Améliorer `BacktestSimulator` ([`src/execution/simulator.py`](src/execution/simulator.py)) pour une **simulation détaillée des opérations sur futures** :
        *   Gestion du levier (configurable).
        *   Calcul et application des frais de financement (funding fees).
        *   Simulation des appels de marge et des liquidations (basé sur le levier et la maintenance margin).
        *   Gestion du roulement des contrats (si applicable).
    *   Mettre à jour `_run_trading_simulation` ([`src/backtesting/comprehensive_backtest.py:450`](src/backtesting/comprehensive_backtest.py:450)) pour utiliser le simulateur amélioré.

3.  **Génération de Signaux** ([`src/signal_generation/signal_generator.py`](src/signal_generation/signal_generator.py)) :
    *   Explorer des logiques de génération de signaux plus avancées qui tirent parti des données de futures :
        *   Signaux basés sur le sentiment (funding rates, open interest).
        *   Signaux de breakout de volatilité basés sur le basis ou l'open interest.
        *   Stratégies de couverture utilisant les futures.
    *   Adapter `allocate_capital_simple` ([`src/signal_generation/signal_generator.py:49`](src/signal_generation/signal_generator.py:49)) ou créer une nouvelle fonction pour une allocation de capital tenant compte du levier et de la gestion de marge pour les futures.

4.  **Gestion des Risques** ([`src/risk_management/`](src/risk_management/)) :
    *   Développer un nouveau module `src/risk_management/margin_management.py` pour :
        *   Calculer les exigences de marge initiales et de maintenance.
        *   Surveiller le niveau de marge du portefeuille.
        *   Implémenter une logique pour gérer les appels de marge (par exemple, réduire la position, ajouter des fonds - ce dernier étant plus pour du live trading mais bon à simuler).
    *   Intégrer ce module dans le `BacktestSimulator` et potentiellement dans le système de trading live.
    *   Adapter les `dynamic_stops` ([`src/risk_management/dynamic_stops.py`](src/risk_management/dynamic_stops.py)) pour qu'ils puissent utiliser la volatilité des futures ou d'autres indicateurs pertinents.

### Phase 6 : Implémentation et Tests (Continu)

1.  **Développement Itératif** : Implémenter les changements par petites étapes, avec des tests unitaires et d'intégration à chaque étape.
2.  **Tests de Non-Régression** : S'assurer que les modifications n'impactent pas négativement les fonctionnalités existantes pour les marchés spot.
3.  **Documentation** : Mettre à jour la documentation (schémas de base de données, nouvelles features, fonctionnement des modèles, architecture du simulateur).

## Diagramme Mermaid du Flux Général Envisagé

```mermaid
graph TD
    A[API Bitget Futures] --> B(BitgetConnector Ext)
    B --> C{Stockage Données Futures (SQL)}
    C --> D[Prétraitement Futures]
    D --> E[Feature Engineering Futures]
    subgraph Pipeline Existant Modifié
        F[Données Spot (Existant)] --> G[Prétraitement Spot (Existant)]
        G --> H[Feature Engineering Spot (Existant)]
        E --> I{Combinaison Features}
        H --> I
        I --> J[Modélisation Avancée (Adaptée/Nouveaux Modèles)]
        J --> K[Génération Signaux Futures]
        K --> L[Backtesting Futures Détaillé]
        L --> M[Analyse Performance & Métriques Futures]
    end
    N[Gestion de Marge] --> L
    O[Gestion des Risques Futures] --> L
```

## Priorisation des Actions Clés

1.  **Très Haute Priorité** :
    *   Phase 1 : Acquisition complète des données de futures (Open Interest, Funding Rates, etc.) et stockage SQL.
    *   Phase 5 (Backtesting) : Adaptation du `BacktestSimulator` pour une simulation détaillée des futures (levier, frais de financement). Sans cela, l'évaluation des stratégies futures sera irréaliste.
2.  **Haute Priorité** :
    *   Phase 2 : Développement des fonctions de calcul pour les caractéristiques clés des futures (Open Interest, Funding Rates).
    *   Phase 3 : Adaptation initiale des modèles existants (LSTM/CNN) pour ingérer ces nouvelles caractéristiques.
    *   Phase 5 (Gestion des Risques) : Développement du module de gestion de marge.
3.  **Moyenne Priorité** :
    *   Phase 3 : Exploration de nouveaux types de modèles spécifiques aux futures.
    *   Phase 4 : Développement de métriques d'évaluation spécifiques aux futures.
    *   Phase 5 (Génération de Signaux) : Développement de logiques de signaux et d'allocation de capital avancées pour les futures.
4.  **Basse Priorité (mais important à long terme)** :
    *   Phase 2 : Développement de caractéristiques combinées plus complexes.
    *   Phase 4 : Scénarios de test de stress spécifiques aux futures.

Ce plan d'action est une proposition initiale et pourra être affiné. Il met l'accent sur la construction d'une fondation solide pour l'acquisition et la simulation des données de futures avant de se concentrer sur des optimisations de modèles plus poussées.