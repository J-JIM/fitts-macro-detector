"""fittsguard: Fitts' law based macro / bot detection."""
from .age_model import AgeMotorModel
from .detector import LayeredDetector, MacroDetector, Verdict, load_detector
from .features import extract_features
from .fitts import fit_fitts, index_of_difficulty, throughput
from .schema import Session, Trial

__all__ = [
    "Session", "Trial", "fit_fitts", "index_of_difficulty", "throughput", "extract_features",
    "LayeredDetector", "MacroDetector", "Verdict", "load_detector", "AgeMotorModel",
]
__version__ = "0.1.0"
