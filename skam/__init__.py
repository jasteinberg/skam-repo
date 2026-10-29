"""Structured Knowledge Associative Memory.

Implementation of J. Steinberg & H. Sompolinsky, "Associative memory of
structured knowledge", Sci. Rep. 12, 21808 (2022), written from the paper's
equations and Methods.

  vsa       HRR binding, Dictionary, structures and sequences   Eqs. 1-6, 10-11, 15-18
  memory    learning rules and retrieval dynamics               Eqs. 23-28, SI Eq. 66
  measures  order parameters, SNR, p(m) shape, m_min            Eqs. 20, 29, 32, SI 60-62
  theory    analytic predictions                                Eqs. 9, 19, 22, 30-31
  ks        sequences of attractors                             SI Eqs. 85-89
  backend   numpy (default) or optional torch for the hot loops
  trials    parallel, resumable, shardable trial execution
"""
