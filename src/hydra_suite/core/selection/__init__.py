"""Embedding-based candidate selection shared by the labeling kits.

Pure numpy (scikit-learn for clustering, hnswlib when installed): no Qt, no
torch, so it imports cheaply and is usable from scripts.  ClassKit's active
learning batch and PoseKit's Smart Select are both built from these pieces.
"""
