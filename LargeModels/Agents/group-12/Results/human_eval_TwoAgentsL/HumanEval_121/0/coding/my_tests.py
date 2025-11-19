# Disable ruff linter for template files
# ruff: noqa: F821 E722

def check(candidate):

    # Check some simple cases
    assert candidate([5, 8, 7, 1])    == 12
    assert candidate([3, 3, 3, 3, 3]) == 9
    assert candidate([30, 13, 24, 321]) == 0
    assert candidate([5, 9]) == 5
    assert candidate([2, 4, 8]) == 0
    assert candidate([30, 13, 23, 32]) == 23
    assert candidate([3, 13, 2, 9]) == 3

    # Check some edge cases that are easy to work out by hand.




def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("ALL TESTS PASSED !#!#\nTERMINATE")
    except:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")
