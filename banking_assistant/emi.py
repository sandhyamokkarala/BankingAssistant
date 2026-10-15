"""EMI calculation independent of LLM and API dependencies."""

import math


def calculate_emi(principal: float, annual_interest_rate: float, tenure_months: int) -> float:
    """Return the monthly installment for a fixed-rate loan."""
    if not math.isfinite(principal) or principal <= 0:
        raise ValueError("Principal must be a finite amount greater than zero.")
    if not math.isfinite(annual_interest_rate) or annual_interest_rate < 0:
        raise ValueError("Annual interest rate must be a finite, non-negative percentage.")
    if tenure_months <= 0:
        raise ValueError("Tenure must be a positive number of months.")

    monthly_rate = annual_interest_rate / 100 / 12
    if monthly_rate == 0:
        return principal / tenure_months
    factor = (1 + monthly_rate) ** tenure_months
    return principal * monthly_rate * factor / (factor - 1)
