"""Domain bindings for ANVIL.

A domain supplies the three domain-specific planes ANVIL needs — the
mutable scaffold, the golden set, and the scorer/predict path — while
reusing the domain-agnostic loop / optimizer / frontier / git-gate. The
built-in NeoVolt customer-support domain lives directly under
``anvil.*``; additional domains (e.g. ``savesage``) live here.
"""
