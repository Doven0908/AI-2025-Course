# Disable ruff linter for template files
# ruff: noqa: F821



METADATA = {}


def check(candidate):
    assert candidate([1, 2, 3]) == 3
    assert candidate([5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10]) == 124



def run_tests(candidate):
    check(candidate)
    # We can search for this string in the output
    print("ALL TESTS PASSED !#!#\nTERMINATE")
