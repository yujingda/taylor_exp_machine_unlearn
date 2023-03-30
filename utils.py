def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def list_difference(lst1, lst2):
    return list(set(lst1) - set(lst2))
def test_intersection():
    assert intersection([1, 2, 3], [2, 3, 4]) == [2, 3]
    assert intersection([1, 2, 3], [4, 5, 6]) == []
    assert intersection([1, 2, 3], [1, 2, 3]) == [1, 2, 3]
    assert intersection([], []) == []

test_intersection()