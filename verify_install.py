"""
Bitcoinator - Installation Verification Script

Checks if all required packages are installed and working correctly.
Run this after installation to verify everything is set up properly.

Usage:
    python verify_install.py
"""

import sys
import subprocess
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_check(name: str, status: bool, message: str = ""):
    """Print a check result."""
    symbol = "✅" if status else "❌"
    status_text = "OK" if status else "FAILED"
    print(f"{symbol} {name:30s} [{status_text}]")
    if message and not status:
        print(f"   → {message}")


def check_package(package_name: str, import_name: str = None):
    """Check if a package is installed."""
    import_name = import_name or package_name
    
    try:
        __import__(import_name)
        return True, ""
    except ImportError as e:
        return False, str(e)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 11)
    
    if version >= required:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor} (need 3.11+)"


def check_data_files():
    """Check if data files exist."""
    data_file = Path("archive/btcusd_1-min_data.csv")
    
    if data_file.exists():
        # Check file size
        size_mb = data_file.stat().st_size / (1024 * 1024)
        return True, f"Found ({size_mb:.1f} MB)"
    else:
        return False, "Data file not found"


def check_config():
    """Check if config file exists."""
    config_file = Path("config.yaml")
    
    if config_file.exists():
        return True, "config.yaml found"
    else:
        return False, "config.yaml not found"


def check_directory_structure():
    """Check if required directories exist."""
    required_dirs = [
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/backtesting",
        "src/validation",
        "src/utils",
        "dashboard",
        "dashboard/pages",
        "models",
        "logs",
        "tests"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All directories present"


def check_model_files():
    """Check if model files exist (optional)."""
    model_files = [
        "models/xgboost_infinite.json",
        "models/xgboost_best.json",
        "models/model_best.json"
    ]
    
    existing = [f for f in model_files if Path(f).exists()]
    
    if existing:
        return True, f"Found: {Path(existing[0]).name}"
    else:
        return None, "No trained models (run training first)"


def test_imports():
    """Test importing key modules."""
    modules = [
        ("src.data.loader", "Data loading"),
        ("src.features.technical", "Feature engineering"),
        ("src.evaluation.metrics", "Metrics"),
        ("src.backtesting.backtester", "Backtesting"),
        ("src.validation.walk_forward", "Walk-forward validation"),
        ("src.training.optimizer", "Hyperparameter optimization"),
        ("src.utils.config", "Configuration"),
        ("src.utils.logger", "Logging"),
        ("src.utils.mlflow_tracker", "MLflow tracking"),
    ]
    
    results = []
    for module, description in modules:
        try:
            __import__(module)
            results.append((description, True, ""))
        except Exception as e:
            results.append((description, False, str(e)))
    
    return results


def get_package_version(package_name: str) -> str:
    """Get installed version of a package."""
    try:
        import importlib
        module = importlib.import_module(package_name)
        return getattr(module, '__version__', 'unknown')
    except:
        return 'not installed'


def main():
    """Run all verification checks."""
    print_header("BITCOINATOR - Installation Verification")
    
    all_passed = True
    
    # Python version
    print_header("1. Python Version")
    status, msg = check_python_version()
    print_check("Python Version", status, msg)
    if msg:
        print(f"   → {msg}")
    if not status:
        all_passed = False
    
    # Core packages
    print_header("2. Core Packages")
    core_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("yaml", "pyyaml"),
        ("plotly", "plotly"),
        ("streamlit", "streamlit"),
    ]
    
    for import_name, display_name in core_packages:
        status, error = check_package(import_name, import_name)
        print_check(display_name, status, error)
        if not status:
            all_passed = False
    
    # ML packages
    print_header("3. Machine Learning Packages")
    ml_packages = [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ]
    
    for import_name, display_name in ml_packages:
        status, error = check_package(import_name, import_name)
        print_check(display_name, status, error)
        if not status:
            print(f"   → Install with: pip install {import_name}")
    
    # Optimization & Tracking
    print_header("4. Optimization & Tracking")
    opt_packages = [
        ("optuna", "Optuna"),
        ("mlflow", "MLflow"),
    ]
    
    for import_name, display_name in opt_packages:
        status, error = check_package(import_name, import_name)
        print_check(display_name, status, error)
        if not status:
            print(f"   → Optional: pip install {import_name}")
    
    # Testing packages
    print_header("5. Testing Packages")
    test_packages = [
        ("pytest", "pytest"),
    ]
    
    for import_name, display_name in test_packages:
        status, error = check_package(import_name, import_name)
        print_check(display_name, status, error)
        if not status:
            print(f"   → Optional: pip install {import_name}")
    
    # Data and Config
    print_header("6. Data & Configuration")
    status, msg = check_data_files()
    print_check("Data File", status, msg)
    if not status:
        print(f"   → Place data in: archive/btcusd_1-min_data.csv")
        all_passed = False
    
    status, msg = check_config()
    print_check("Configuration", status, msg)
    if not status:
        all_passed = False
    
    # Directory structure
    print_header("7. Directory Structure")
    status, msg = check_directory_structure()
    print_check("Directories", status, msg)
    if not status:
        all_passed = False
        print(f"   → {msg}")
    
    # Module imports
    print_header("8. Module Imports")
    import_results = test_imports()
    for description, status, error in import_results:
        print_check(description, status, error)
        if not status:
            all_passed = False
    
    # Model files (optional)
    print_header("9. Trained Models (Optional)")
    status, msg = check_model_files()
    if status is not None:
        print_check("Model Files", status, msg)
    else:
        print(f"⚠️  {msg}")
        print(f"   → Run: python train_optimized.py --model xgboost")
    
    # Summary
    print_header("Summary")
    
    if all_passed:
        print("✅ All required checks passed!")
        print("\n🎉 Bitcoinator is ready to use!")
        print("\nNext steps:")
        print("  1. Train a model: python train_optimized.py --model xgboost")
        print("  2. Run dashboard: streamlit run dashboard/app.py")
        print("  3. View docs: cat README.md")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
