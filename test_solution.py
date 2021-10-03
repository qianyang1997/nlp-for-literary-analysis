import pytest
from solution import find_function_names, find_dates


text_good = """
def add(x, y):
    return x, y

text = "There is a distinction between birthday
 and birthdate: The former, other than February 29,
 occurs each year (e.g., January 15), while the
 latter is the exact date a person was born
 (e.g., January 15, 2001).
"""


def test_find_function_names_good():
    assert find_function_names(text_good) == ['add']


def test_find_function_names_bad():
    assert find_function_names("") == []


def test_find_dates_good():
    assert find_dates(text_good) == [('January', 15, 2001)]


def test_find_dates_bad():
    with pytest.raises(TypeError):
        find_dates(1)
