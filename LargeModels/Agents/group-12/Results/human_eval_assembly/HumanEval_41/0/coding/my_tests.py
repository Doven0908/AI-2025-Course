# Disable ruff linter for template files
# ruff: noqa: F821



METADATA = {}


def check(candidate):
    assert candidate(2) == 4
    assert candidate(3) == 9
    assert candidate(4) == 16
    assert candidate(8) == 64
    assert candidate(10) == 100




def run_tests(candidate):
    check(candidate)
    # We can search for this string in the output
    print("ALL TESTS PASSED !#!#\nTERMINATE")
