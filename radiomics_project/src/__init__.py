__version__ = "2.0.0"
__author__ = "Medical AI Team"


def __getattr__(name):
    _lazy_imports = {
        "MRIPreprocessor": ".data_preprocessing",
        "PyRadiomicsExtractor": ".feature_extraction",
        "DeltaFeatureCalculator": ".delta_features",
        "ICCAnalyzer": ".icc_analysis",
        "CascadeFeatureSelector": ".feature_selection",
        "MultiModelTrainer": ".model_training",
        "ComprehensiveEvaluator": ".evaluation",
        "load_config": ".utils",
        "setup_logger": ".utils",
    }
    if name in _lazy_imports:
        import importlib
        mod = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MRIPreprocessor",
    "PyRadiomicsExtractor",
    "DeltaFeatureCalculator",
    "ICCAnalyzer",
    "CascadeFeatureSelector",
    "MultiModelTrainer",
    "ComprehensiveEvaluator",
    "load_config",
    "setup_logger",
]
