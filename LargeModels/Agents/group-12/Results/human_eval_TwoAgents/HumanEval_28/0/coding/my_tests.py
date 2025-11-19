# Disable ruff linter for template files
# ruff: noqa: F821 E722



METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate([]) == ''
    assert candidate(['x', 'y', 'z']) == 'xyz'
    assert candidate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'



def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("ALL TESTS PASSED !#!#\nTERMINATE")
    except:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")
