import pytest

from banking_assistant.emi import calculate_emi


def test_calculates_zero_interest_emi() -> None:
    assert calculate_emi(120_000, 0, 12) == pytest.approx(10_000)


def test_calculates_fixed_rate_emi() -> None:
    assert calculate_emi(1_000_000, 9, 60) == pytest.approx(20_758.36, abs=0.01)


@pytest.mark.parametrize(
    ("principal", "rate", "months"),
    [(0, 9, 60), (-1, 9, 60), (10_000, -1, 12), (10_000, 9, 0)],
)
def test_rejects_invalid_loan_inputs(principal: float, rate: float, months: int) -> None:
    with pytest.raises(ValueError):
        calculate_emi(principal, rate, months)
