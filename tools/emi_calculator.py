def calculate_emi(principal, annual_rate, months):
    if principal<=0 or months<=0: raise ValueError("Invalid inputs")
    r=annual_rate/12/100
    if r==0: return round(principal/months,2)
    return round(principal*r*(1+r)**months/((1+r)**months-1),2)
