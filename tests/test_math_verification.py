"""Math == code: the loss/health formulas match an independent numpy reimplementation."""

from scripts.verify_math import main as verify_main


def test_math_matches_code():
    assert verify_main() == 0, "a formula in src/ drifted from its mathematical definition"


if __name__ == "__main__":
    test_math_matches_code()
    print("OK")
