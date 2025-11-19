# Disable ruff linter for template files
# ruff: noqa: F821 E722

def check(candidate):

    # Check some simple cases
    assert candidate("AsDf") == "aSdF"
    assert candidate("1234") == "4321"
    assert candidate("ab") == "AB"
    assert candidate("#a@C") == "#A@c"
    assert candidate("#AsdfW^45") == "#aSDFw^45"
    assert candidate("#6@2") == "2@6#"

    # Check some edge cases that are easy to work out by hand.
    assert candidate("#$a^D") == "#$A^d"
    assert candidate("#ccc") == "#CCC"

    # Don't remove this line:



def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("ALL TESTS PASSED !#!#\nTERMINATE")
    except:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")
