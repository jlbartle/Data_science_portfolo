
CREATE VIEW Meal_Contents as
SELECT MealDT, F.Name, Amount, F.Serving_Info, Calorie, Carb, Fat, Protein, Fiber  from MEALS
JOIN FOOD F on MEALS.FoodID = F.FoodID

CREATE VIEW Credit_History as
SELECT TransactionDT, CREDIT.Description, EC.Description, Amount
from CREDIT
join EXPENSE_CATEGORY EC on CREDIT.CategoryID = EC.CategoryID

CREATE VIEW Credit_Balance as
SELECT ROUND(SUM(Amount), 2) as Balance
FROM CREDIT

CREATE VIEW Checking_Balance as
SELECT ROUND(SUM(Amount), 2) AS Balance
FROM CHECKING


Create procedure Sort_Credit_Misc
AS
    UPDATE CREDIT SET CategoryID = 14 WHERE Description LIKE 'PAYPAL*'; -- SORT INTO AMENITYS
    UPDATE CREDIT SET CategoryID = 13 WHERE Description LIKE '*UNIQLO*' OR Description LIKE '*SHOES*' OR
                                            Description LIKE '*MASSDROP*' OR Description LIKE '*CLOTHES*'; --SORT INTO CLOTHES
GO;