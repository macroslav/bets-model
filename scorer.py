import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Protocol

from models import BaseModel


class BaseScorer(Protocol):
    """

        Implementation of base scorer class

    """


class MoneyScorer:
    """ """
