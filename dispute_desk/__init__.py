from dispute_desk.client import DisputeDeskEnv
from dispute_desk.models import CaseAction, CaseObservation, EnvironmentStateModel

DisputeDeskAction = CaseAction
DisputeDeskObservation = CaseObservation
DisputeDeskState = EnvironmentStateModel

__all__ = [
    "DisputeDeskAction",
    "DisputeDeskEnv",
    "DisputeDeskObservation",
    "DisputeDeskState",
    "CaseAction",
    "CaseObservation",
    "EnvironmentStateModel",
]
