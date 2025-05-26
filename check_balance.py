import os
from dotenv import load_dotenv

# Configuration du chemin pour les imports de modules locaux
import sys
# Ajoute le répertoire parent (AlphaBeta808Trading) au sys.path
# pour que les imports comme src.acquisition.connectors fonctionnent.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.acquisition.connectors import get_binance_balance, BINANCE_AVAILABLE

def main():
    """
    Fonction principale pour charger les clés API, récupérer et afficher les soldes de stablecoins.
    """
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("Erreur : Les clés API Binance (BINANCE_API_KEY, BINANCE_API_SECRET) "
              "ne sont pas configurées dans le fichier .env.")
        return

    if not BINANCE_AVAILABLE:
        print("Erreur : La librairie python-binance n'est pas installée. "
              "Veuillez l'installer (pip install python-binance).")
        return

    print("Récupération des soldes de stablecoins depuis Binance testnet...")
    
    # Vous pouvez spécifier une liste de stablecoins si vous le souhaitez, par exemple :
    # stablecoins_to_check = ['USDC', 'USDT']
    # Sinon, la fonction get_binance_balance utilisera sa liste par défaut.
    stablecoin_balances = get_binance_balance(api_key, api_secret, testnet=True) 

    if stablecoin_balances:
        print("\nSoldes des stablecoins trouvés :")
        total_stablecoin_value = 0.0
        for asset, balance in stablecoin_balances.items():
            print(f"- {asset}: {balance}")
            total_stablecoin_value += balance # Supposant une parité 1:1 avec une monnaie de référence (ex: USD)
        
        print(f"\nValeur totale approximative des stablecoins : {total_stablecoin_value:.2f}")
    else:
        print("Aucun solde de stablecoin n'a été récupéré ou les soldes sont nuls pour les actifs par défaut.")
        print("Actifs par défaut vérifiés : ['USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX']")

if __name__ == "__main__":
    main()
