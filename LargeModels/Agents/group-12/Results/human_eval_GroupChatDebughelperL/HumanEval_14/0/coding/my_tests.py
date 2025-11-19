# Disable ruff linter for template files
# ruff: noqa: F821



METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate('') == []
    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']
    assert candidate('WWW') == ['W', 'WW', 'WWW']



def run_tests(candidate):
    check(candidate)
    # We can search for this string in the output
    print("ALL TESTS PASSED !#!#\nTERMINATE")
