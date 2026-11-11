#!/usr/bin/env python3
"""Deterministic local smoke run for the ICDE05 certifiers."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.graph import Graph, Query
from src.certifier import Counts, Certifier
from src.batched_certifier import BatchedCertifier
def main():
    graph=Graph(3,[(0,1),(1,2),(0,2)])
    queries=[Query(0,1),Query(1,2),Query(0,2)]
    counts=Counts(np.array([128,128,128]),np.array([96,83,79]))
    cert=Certifier(graph,queries,1.0,0.05)
    batched=BatchedCertifier(graph,queries,1.0,0.05)
    result={"certifier":cert.evaluate(counts).tolist(),
            "batched":batched.evaluate(counts).tolist(),
            "scores":graph.oracle(counts.S-counts.N//2,queries[0],False)[0]}
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
