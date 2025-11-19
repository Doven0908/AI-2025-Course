# Disable ruff linter for template files
# ruff: noqa: F821



METADATA = {}


def check(candidate):
    assert candidate([-1, -2, 4, 5, 6]) == [4, 5, 6]
    assert candidate([5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 3, 9, 123, 1]
    assert candidate([-1, -2]) == []
    assert candidate([]) == []




def run_tests(candidate):
    check(candidate)
    # We can search for this string in the output
    print("ALL TESTS PASSED !#!#\nTERMINATE")
