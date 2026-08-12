# Preprocessing

Baseline pipeline:
1. Remove duplicates
2. Handle infinite/missing values
3. Create age/employment features
4. Create credit-to-income, annuity-to-income and goods-to-income ratios
5. Create income-per-family-member
6. Create mean external-source score
7. Count missing values
8. Encode categorical variables
9. Standardize model inputs
10. Stratified train/validation/test split

Next: aggregate bureau, previous_application, installments, POS_CASH and credit_card tables into customer-level historical features.
