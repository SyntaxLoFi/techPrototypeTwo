#!/usr/bin/env python3
"""
Setup script for Polymarket-Lyra Arbitrage Scanner
Creates necessary directories and environment
"""
import os
import shutil
import subprocess
import sys


def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'results',
        'visualizations',
        'data',
        'data/funding_rates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("✓ Created .env file from .env.example")
            print("⚠️  Please edit .env file with your configuration")
        else:
            print("❌ .env.example not found")
    else:
        print("✓ .env file already exists")


def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies...")
    
    # Create requirements.txt if it doesn't exist
    requirements = """
# Core dependencies
websockets>=11.0
aiohttp>=3.8.0
requests>=2.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Additional dependencies
python-dotenv>=1.0.0
colorama>=0.4.6
tabulate>=0.9.0
certifi>=2023.0.0

# Optional for enhanced features
ccxt>=4.0.0  # For additional exchange support
redis>=4.5.0  # For caching (optional)
""".strip()
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Install packages
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("✓ Dependencies installed")


def create_test_script():
    """Create a simple test script"""
    test_script = """#!/usr/bin/env python3
'''
Quick test script to verify setup
'''
import asyncio
from polymarket_fetcher import PolymarketFetcher
from perps_data_collector import PerpsDataCollector
from config_manager import ARBITRAGE_ENABLED_CURRENCIES

def test_polymarket():
    print("Testing Polymarket connection...")
    fetcher = PolymarketFetcher()
    events = fetcher.fetch_crypto_events()
    print(f"✓ Found {len(events)} crypto events")
    
    contracts = fetcher.parse_contracts(events)
    print(f"✓ Parsed {len(contracts)} contracts")
    
    return contracts

def test_lyra():
    print("\\nTesting Lyra connection...")
    collector = PerpsDataCollector()
    
    for currency in ['BTC', 'ETH']:
        data = collector.fetch_perp_info(currency, fetch_funding_history=False)
        if data:
            print(f"✓ {currency} perp: ${data['mark_price']:,.2f}")

def main():
    print("🧪 Running setup tests...")
    print("="*50)
    
    # Test Polymarket
    contracts = test_polymarket()
    
    # Test Lyra
    test_lyra()
    
    print("\\n✅ All tests passed! Ready to run main scanner.")
    print("\\nRun: python main_scanner.py")

if __name__ == "__main__":
    main()
"""
    
    with open('test_setup.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_setup.py', 0o755)
    print("✓ Created test_setup.py")


def main():
    """Main setup process"""
    print("🚀 Setting up Polymarket-Lyra Arbitrage Scanner")
    print("="*50)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Create .env file
    print("\nSetting up configuration...")
    create_env_file()
    
    # Install dependencies
    install_dependencies()
    
    # Create test script
    print("\nCreating test script...")
    create_test_script()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your settings")
    print("2. Run: python test_setup.py (to verify setup)")
    print("3. Run: python main_scanner.py (to start scanning)")


if __name__ == "__main__":
    main()