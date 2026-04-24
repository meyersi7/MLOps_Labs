import pytest

def answer_to_life_universe_everything():
    return 42


def test_answer():
    assert 42 == answer_to_life_universe_everything()

#def test_answer_fail():
#    assert 47 == answer_to_life_universe_everything()

#def validate_answer(answer):
#    if 42 != answer:
#        raise ValueError(f"Answer {answer} is wrong.")

#def test_validate_answer():
#    with pytest.raises(ValueError):
#        validate_answer(88)

#@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
#def test_eval(test_input, expected):
#    assert eval(test_input) == expected

