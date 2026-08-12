from tools.emi_calculator import calculate_emi

def test_emi_positive():
    assert calculate_emi(800000,9,60)>0
