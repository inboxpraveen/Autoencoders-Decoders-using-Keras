"""Shared test setup: import from the project root and plot without a display."""

import os
import sys

import matplotlib

matplotlib.use('Agg')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
