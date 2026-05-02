# stn_gpe/__init__.py
"""
basal_ganglia model package
Contains BG network model, train loop
for studying decision-making under different conditions
"""

from .BGNetwork import BGNetwork
from .train import train


# Define what should be accessible when importing `stn_gpe`
__all__ = [
    "BGNetwork",
    "train"
]