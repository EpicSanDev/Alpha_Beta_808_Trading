import os

class EnvironmentConfig:
    """
    Configuration de base pour les environnements.
    Les valeurs par défaut peuvent être surchargées par des variables d'environnement.
    """
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.broker_api_key = os.getenv(f"{environment_name.upper()}_BROKER_API_KEY", "default_api_key")
        self.broker_api_secret = os.getenv(f"{environment_name.upper()}_BROKER_API_SECRET", "default_api_secret")
        self.broker_url = os.getenv(f"{environment_name.upper()}_BROKER_URL", "default_broker_url") # Généralement pour l'exécution

        # Configuration spécifique à la source de données
        self.data_source_name = os.getenv(f"{environment_name.upper()}_DATA_SOURCE_NAME", "binance").lower() # ex: binance, bitget, csv

        # Clés pour Binance (si data_source_name == 'binance' ou par défaut pour broker_api_key)
        self.binance_api_key = os.getenv(f"{environment_name.upper()}_BINANCE_API_KEY", self.broker_api_key)
        self.binance_api_secret = os.getenv(f"{environment_name.upper()}_BINANCE_API_SECRET", self.broker_api_secret)
        self.binance_testnet = os.getenv(f"{environment_name.upper()}_BINANCE_TESTNET", "False").lower() == "true"


        # Clés pour Bitget (si data_source_name == 'bitget')
        self.bitget_api_key = os.getenv(f"{environment_name.upper()}_BITGET_API_KEY", "default_bitget_api_key")
        self.bitget_api_secret = os.getenv(f"{environment_name.upper()}_BITGET_API_SECRET", "default_bitget_api_secret")
        self.bitget_passphrase = os.getenv(f"{environment_name.upper()}_BITGET_PASSPHRASE", "default_bitget_passphrase")
        # Bitget n'a pas de mode testnet distinct pour l'API publique de klines de la même manière que Binance,
        # mais on pourrait avoir une URL de base différente pour un sandbox si disponible.
        # Pour l'instant, on utilise l'URL de production.

        self.db_connection_string = os.getenv(f"{environment_name.upper()}_DB_CONNECTION_STRING", "default_db_string")
        self.log_level = os.getenv(f"{environment_name.upper()}_LOG_LEVEL", "INFO")
        self.is_simulated_execution = True # Par défaut, exécution simulée

    def __repr__(self):
        return (f"<EnvironmentConfig(name='{self.environment_name}', "
                f"data_source='{self.data_source_name}', "
                f"broker_url='{self.broker_url}', "
                f"simulated={self.is_simulated_execution})>")

class DevelopmentConfig(EnvironmentConfig):
    def __init__(self):
        super().__init__("development")
        self.log_level = os.getenv("DEVELOPMENT_LOG_LEVEL", "DEBUG")
        # Spécificités pour le développement, ex: broker de test, base de données locale
        self.broker_url = os.getenv("DEVELOPMENT_BROKER_URL", "http://localhost:8080/dev_broker")
        self.is_simulated_execution = True 

class StagingConfig(EnvironmentConfig):
    def __init__(self):
        super().__init__("staging")
        # L'environnement de staging devrait se rapprocher de la production
        # mais potentiellement avec un compte de courtier papier ou des données limitées.
        self.broker_url = os.getenv("STAGING_BROKER_URL", "https_broker_paper_trading_url")
        self.is_simulated_execution = True # Ou False si on teste avec un compte papier réel

class ProductionConfig(EnvironmentConfig):
    def __init__(self):
        super().__init__("production")
        self.log_level = os.getenv("PRODUCTION_LOG_LEVEL", "WARNING")
        self.broker_url = os.getenv("PRODUCTION_BROKER_URL", "https_broker_live_trading_url")
        self.is_simulated_execution = False # Exécution réelle en production

class ShadowConfig(EnvironmentConfig):
    def __init__(self):
        super().__init__("shadow")
        # L'environnement shadow utilise les mêmes données que la production
        # mais les ordres sont simulés pour évaluer un nouveau modèle/stratégie.
        self.broker_url = os.getenv("SHADOW_BROKER_URL", os.getenv("PRODUCTION_BROKER_URL", "https_broker_live_trading_url")) # Peut pointer vers le même broker que prod
        self.is_simulated_execution = True # Les ordres sont simulés, pas exécutés réellement.
        self.log_level = os.getenv("SHADOW_LOG_LEVEL", "INFO")


def get_config(env_name: str = None) -> EnvironmentConfig:
    """
    Récupère la configuration pour l'environnement spécifié.
    L'environnement peut être spécifié par la variable d'environnement APP_ENV.
    """
    if env_name is None:
        env_name = os.getenv("APP_ENV", "development").lower()

    if env_name == "production":
        return ProductionConfig()
    elif env_name == "staging":
        return StagingConfig()
    elif env_name == "shadow":
        return ShadowConfig()
    elif env_name == "development":
        return DevelopmentConfig()
    else:
        print(f"Avertissement: Environnement '{env_name}' non reconnu. Utilisation de la configuration de développement par défaut.")
        return DevelopmentConfig()

# Exemple d'utilisation (à mettre dans main.py ou un point d'entrée)
# if __name__ == "__main__":
#     # Pour tester, vous pouvez définir APP_ENV dans votre terminal avant de lancer
#     # export APP_ENV=staging
#     config = get_config()
#     print(f"Configuration chargée pour l'environnement: {config.environment_name}")
#     print(f"  URL du Broker: {config.broker_url}")
#     print(f"  Exécution simulée: {config.is_simulated_execution}")
#     print(f"  Niveau de log: {config.log_level}")

#     # Exemple pour charger une configuration spécifique
#     prod_config = get_config("production")
#     print(f"\nConfiguration de Production (exemple):")
#     print(f"  URL du Broker: {prod_config.broker_url}")
#     print(f"  Exécution simulée: {prod_config.is_simulated_execution}")

"""
Commentaires sur la gestion des environnements (pour main.py ou documentation):

La gestion des environnements (développement, staging, production, shadow) est cruciale
pour un déploiement robuste et sécurisé d'une stratégie de trading algorithmique.

Ce fichier `config/deployment_config.py` propose une structure pour gérer ces configurations.
L'idée principale est d'utiliser des classes de configuration distinctes pour chaque environnement,
héritant potentiellement d'une classe de base.

Les configurations spécifiques (clés API, URLs de broker, chaînes de connexion DB)
sont idéalement chargées à partir de variables d'environnement pour éviter de les
coder en dur dans le code source (sécurité et flexibilité).

Dans `main.py` ou le point d'entrée de l'application, une fonction comme `get_config()`
serait appelée pour charger la configuration appropriée en fonction d'une variable
d'environnement (par exemple, `APP_ENV`).

Exemple de flux dans `main.py`:
1. Lire la variable d'environnement `APP_ENV`.
2. Appeler `config = get_config(os.getenv('APP_ENV'))`.
3. Utiliser `config.broker_url`, `config.is_simulated_execution`, etc., dans toute l'application.

Environnements spécifiques:
- Développement: Utilisé pour le codage et les tests unitaires/intégration locaux.
  Devrait utiliser des brokers de test/simulation, des bases de données locales/de test.
  Exécution simulée par défaut.

- Staging: Un environnement qui reflète aussi fidèlement que possible la production.
  Utilisé pour les tests d'intégration finaux, les tests de performance, et la validation
  avant le déploiement en production. Peut utiliser un compte de courtier "papier" (paper trading)
  pour simuler des transactions avec des conditions de marché réelles mais sans argent réel.
  L'exécution peut être simulée ou réelle (sur compte papier).

- Production: L'environnement live où la stratégie trade avec de l'argent réel.
  Doit être hautement sécurisé, monitoré, et robuste.
  Exécution réelle.

- Shadow: Cet environnement est une copie de la production en termes de flux de données
  et de logique de stratégie, mais les ordres générés ne sont pas envoyés au courtier
  (ou sont envoyés à un simulateur). L'objectif est d'évaluer la performance d'une
  nouvelle version de modèle ou d'une nouvelle stratégie en conditions réelles, sans
  risquer de capital. C'est une étape clé avant un déploiement canary ou un remplacement
  complet du modèle en production.
  Exécution simulée.

La distinction entre `is_simulated_execution` (un booléen dans la config) et le type
d'environnement (ex: "shadow") permet une granularité fine. Par exemple, même en
production, on pourrait vouloir forcer une exécution simulée pour des tests spécifiques
(bien que ce soit rare et à manipuler avec précaution).
"""