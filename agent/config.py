from __future__ import annotations

import re


DEFAULT_MODEL = "gemini-3.1-flash-lite"

MAX_REPAIRS = 1

MAX_EXCERPTS = 14

EXCERPT_CHARS = 1400

DISCLAIMER = (
    "*Automated, preliminary draft built only from the company's latest Form 10-K. "
    "It is not investment advice or a credit rating. Verify every figure against the filing before use.*"
)

USAGE = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}

DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}

INSTANT_TAGS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}

LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit", "op_income": "Operating income",
    "net_income": "Net income", "interest_expense": "Interest expense", "d_and_a": "Depreciation and amortization",
    "cfo": "Operating cash flow", "capex": "Capital expenditures", "total_assets": "Total assets",
    "total_liabilities": "Total liabilities", "equity": "Shareholders' equity",
    "current_assets": "Current assets", "current_liabilities": "Current liabilities", "cash": "Cash and equivalents",
}

RETRIEVAL_PLAN = [  # (section, query, top_k)
    ("Business", "company business overview segments products services customers markets", 2),
    ("Risk Factors", "competition competitors pricing pressure market share", 1),
    ("Risk Factors", "indebtedness debt covenants refinancing credit facility liquidity", 2),
    ("Risk Factors", "legal proceedings litigation regulation compliance investigations fines", 1),
    ("Risk Factors", "supplier concentration customer dependence supply chain disruption", 1),
    ("Risk Factors", "macroeconomic conditions inflation interest rates currency exchange", 1),
    ("MD&A", "liquidity and capital resources cash flows debt repayment dividends repurchases", 2),
    ("MD&A", "results of operations revenue increased decreased primarily due to", 2),
    ("MD&A", "going concern substantial doubt impairment restructuring material weakness", 1),
]

SYSTEM_PROMPT = """You are a credit analyst assistant. You draft a preliminary credit memo using ONLY the data provided.

Rules:
1. Every sentence that states a fact must end with one or more citations like [F3] or [S2]. Use [F#] for the facts table and [S#] for filing excerpts. Never cite an ID that is not provided.
2. Copy numbers EXACTLY as written in the facts table (for example $391.0B, 44.1%, 1.07x). Do not round, convert, or recompute anything. Do not do arithmetic.
3. Do not use outside knowledge. If something is not in the data, write 'Not available in the provided data.'
4. Write about the company's credit profile: ability to service debt, liquidity, earnings quality, and key risks. Do not give an investment recommendation or a formal rating.
5. Be concise and specific. No filler. About 450 to 650 words."""

MEMO_TEMPLATE = """Write the memo in Markdown with exactly these sections:
## 1. Business overview
## 2. Financial summary
## 3. Credit strengths
## 4. Key risks
## 5. Red flags and watch items
## 6. Preliminary credit view
Section 6 gives a qualitative view (Stronger / Adequate / Weaker) in 2 to 3 cited sentences, labelled preliminary."""

CITE_RE = re.compile(r"\[([FS]\d+(?:\s*[,;]\s*[FS]\d+)*)\]")

NUM_RE = re.compile(r"(?<![\w.])(-?\$?-?\d[\d,]*(?:\.\d+)?)(\s?(?:%|x\b|billion\b|million\b|thousand\b|[BMK]\b))?", re.I)
