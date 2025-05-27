import sys
import os
from typing import List

# Ajout pour permettre l'importation depuis le répertoire parent src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors import BitgetConnector
from core.database import create_db_and_tables, SessionLocal

# Configuration des symboles à suivre
# TODO: Déplacer cela vers un fichier de configuration ou des arguments CLI
SYMBOLS_TO_TRACK: List[str] = [
    "BTCUSDT_UMCBL",
    "ETHUSDT_UMCBL",
    # Ajoutez d'autres symboles de contrats à terme ici
]

# Nombre de pages à récupérer pour l'historique des funding rates
# Chaque page contient généralement 100 enregistrements (Bitget).
# 5 pages = ~500 enregistrements (environ 500 * 8 heures = 4000 heures = ~166 jours)
FUNDING_RATE_HISTORY_PAGES = 20 # Augmenté pour un historique plus long

def run_collection(symbols: List[str]):
    """
    Exécute la collecte de données pour les symboles spécifiés.
    """
    print("Initialisation du collecteur de données...")
    
    # S'assurer que la base de données et les tables existent
    print("Création/vérification des tables de la base de données...")
    create_db_and_tables()
    print("Tables de la base de données prêtes.")

    # Initialiser le connecteur Bitget (pas besoin de clés API pour les données publiques)
    connector = BitgetConnector()

    print(f"Début de la collecte pour les symboles: {', '.join(symbols)}")

    for symbol in symbols:
        print(f"\nTraitement du symbole: {symbol}")
        try:
            connector.fetch_and_store_all_metrics(
                symbol=symbol,
                fetch_historical_funding=True,
                funding_pages_to_fetch=FUNDING_RATE_HISTORY_PAGES
            )
            print(f"Collecte terminée avec succès pour {symbol}.")
        except Exception as e:
            print(f"Erreur lors de la collecte pour le symbole {symbol}: {e}")
            # Continuer avec le symbole suivant même en cas d'erreur
            continue
    
    print("\nCollecte de toutes les données terminée.")

if __name__ == "__main__":
    run_collection(SYMBOLS_TO_TRACK)