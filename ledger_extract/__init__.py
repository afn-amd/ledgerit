"""ledger_extract: extract normalized transactions (with DR/CR type) from
Indian bank statement PDFs using Camelot's `stream` flavor.

Pipeline: PDF --(camelot stream)--> raw tables --(detect bank)--> profile
--> row parser (date / description / reference / running balance) --> type
derived from the running-balance delta (cross-checked against explicit
Debit/Credit columns) --> normalized CSV.

Canonical output columns:
    Date, Description, Reference, Debit, Credit, Balance, Type
"""

from .common import CANONICAL_COLUMNS

__all__ = ["CANONICAL_COLUMNS"]
