"""
Ao et al. 2024 Basal Ganglia Model
A Spiking Neural Network Action Decision Method Inspired by Basal Ganglia
"""

from .basal_ganglia import BasalGangliaActor
from .critic import CriticNetwork
from .plasticity import CorticostriatalPlasticity

__all__ = [
    'BasalGangliaActor',
    'CriticNetwork',
    'CorticostriatalPlasticity'
]
