------------------------------------------------------------------------------

--Table creates for Samsung Health data
------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.PULSE;
GO

create table PULSE
(
	PulseID int identity
		constraint PULSE_pk
			primary key nonclustered,
	PulseDT datetime,
	Pulse int
)
go

-----------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.CAL_BURNED;
GO

create table CAL_BURNED
(
	Cal_BurnedID int identity
		constraint CAL_BURNED_pk
			primary key nonclustered,
	Cal_BurnedDT datetime,
	Rest_Cal int,
	Active_Cal int
)
go

-----------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.STEPS;
GO

create table STEPS
(
	StepsID int identity
		constraint STEPS_pk
			primary key nonclustered,
	StepsDT datetime,
	Steps_Count int,
	Speed float,
	Cal_Burned float
)
go

-----------------------------------------------------------------------------------------------------

alter table MEALS
    drop constraint MEALS_FOOD_FoodID_fk
go

DROP TABLE IF EXISTS dbo.FOOD;
GO

create table FOOD
(
	FoodID int identity not null
		constraint FOOD_pk
			primary key nonclustered,
	Name varchar(100) not null,
	Serving_Info varchar(100),
	Calorie float default 0,
	Carb float default 0,
	Fat float default 0,
	Protein float default 0,
	Fiber float default 0,
	Cholesterol float default 0,
	VA float default 0,
	Calcium float default 0,
	VC float default 0,
	Sat_Fat float default 0,
	MonoSat_Fat float default 0,
	Potassium float default 0,
	Sodium float default 0,
	Sugars float default 0,
	Iron float default 0
)
go

create unique index FOOD_FOODID_uindex
	on FOOD (FoodID)
go

-----------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.MEALS;
GO

create table MEALS
(
	MealID int identity,
	FoodID int not null
		constraint MEALS_FOOD_FoodID_fk  --fk linking meals to foods
			references FOOD,
	MealDT datetime,
	Amount float
)
go

create unique index MEALS_MealID_uindex
	on MEALS (MealID)
go

alter table MEALS
	add constraint MEALS_pk
		primary key nonclustered (MealID)
go


------------------------------------------------------------------------------

-- Table creates for finacial data

------------------------------------------------------------------------------

alter table CREDIT
    drop constraint CREDIT_CATEGORY_CategoryID_fk
go

alter table CHECKING
    drop constraint CHECKING_CAT_FK
go

DROP TABLE IF EXISTS dbo.EXPENSE_CATEGORY;
GO

create table EXPENSE_CATEGORY
(
	CategoryID int identity
		constraint EXPENSE_CATEGORY_pk
			primary key nonclustered,
	Description varchar(100)
)
go

create unique index EXPENSE_CATEGORY_CATEGORYID_uindex
	on EXPENSE_CATEGORY (CategoryID)
go


--Inserts for categorizations. Expanding on the shopping category and creating a new category payments for tracking card payments
INSERT into EXPENSE_CATEGORY(Description) VALUES('Misc');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Payments');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Automotive');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Bills & Utilities');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Education');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Entertainment');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Food & Drink');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Gas');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Gifts & Donations');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Groceries');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Health & Wellness');  --medical
INSERT into EXPENSE_CATEGORY(Description) VALUES('Home');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Personal');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Amenities');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Clothing');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Travel');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Rent');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Wages');
INSERT into EXPENSE_CATEGORY(Description) VALUES('ATM');
INSERT into EXPENSE_CATEGORY(Description) VALUES('Professional Services');


-----------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.CREDIT;
GO

create table CREDIT
(
	ExpenseID int identity
		constraint CREDIT_pk
			primary key nonclustered,
	TransactionDT datetime,
	Description varchar(100),
	CategoryID int not null default 0
		constraint CREDIT_CATEGORY_CategoryID_fk  -- fk linking credit to its categorys
			references EXPENSE_CATEGORY,
	Amount float default 0
)
go

-----------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS dbo.CHECKING;
GO

create table CHECKING
(
	CheckingID int identity
		constraint CHECKING_pk
			primary key nonclustered,
	TransactionDT datetime,
	Description varchar(100),
	Amount float default 0
)
go

ALTER table CHECKING
    add CategoryID int not null default 1;


ALTER table CHECKING
    add CONSTRAINT CHECKING_CAT_FK  --add fk for checking to use the category table
    FOREIGN KEY (CategoryID)
            references EXPENSE_CATEGORY(CategoryID);

-----------------------------------------------------------------------------------------------------

-- VIEWS

-----------------------------------------------------------------------------------------------------
DROP VIEW IF EXISTS dbo.Meal_Contents;
DROP VIEW IF EXISTS dbo.Credit_Balance;
DROP VIEW IF EXISTS dbo.Credit_History;
DROP VIEW IF EXISTS dbo.Checking_Balance;
go;

CREATE VIEW Meal_Contents as
SELECT MealDT, F.Name, Amount, F.Serving_Info, Calorie, Carb, Fat, Protein, Fiber  from MEALS --view for looking at meals with their food description
JOIN FOOD F on MEALS.FoodID = F.FoodID
go;
CREATE VIEW Credit_History as
SELECT TransactionDT, CREDIT.Description as Description, EC.Description as Category, Amount --view for looking at credit history joined with categorys
from CREDIT
join EXPENSE_CATEGORY EC on CREDIT.CategoryID = EC.CategoryID
go;
CREATE VIEW Credit_Balance as
SELECT ROUND(SUM(Amount), 2) as Balance
FROM CREDIT
go;
CREATE VIEW Checking_Balance as
SELECT ROUND(SUM(Amount), 2) AS Balance
FROM CHECKING
go;

-----------------------------------------------------------------------------------------------------

-- Procedures

-----------------------------------------------------------------------------------------------------

DROP procedure IF EXISTS Sort_Credit_Misc;
DROP procedure IF EXISTS Sort_Checking_Misc;
go;


Create procedure Sort_Credit_Misc --adds my custom categorys to credit
as
    begin
    UPDATE CREDIT SET CategoryID = 14 WHERE Description LIKE 'PAYPAL%'; -- SORT INTO AMENITYS
    UPDATE CREDIT SET CategoryID = 13 WHERE Description LIKE '%UNIQLO%' OR Description LIKE '%SHOES%' OR
                                            Description LIKE '%MASSDROP%' OR Description LIKE '%CLOTHES%'; --SORT INTO CLOTHES
                                            end
go;


Create procedure Sort_Checking_Misc --adds my custom categorys to checking
as
    begin
    UPDATE CHECKING SET CategoryID = 22 WHERE Description LIKE '%ATM%' OR Description LIKE '%Withdrawal%'; -- SORT INTO AMENITYS
    UPDATE CHECKING SET CategoryID = 20 WHERE Description LIKE '%Premier%' or Amount = -850;
    UPDATE CHECKING SET CategoryID = 21 WHERE Description LIKE '%DIR DEP%' OR Description LIKE '%PAYROLL%';
    UPDATE CHECKING SET CategoryID = 2 WHERE Description LIKE '%Payment%';
    end
go;

create procedure Clean_Food -- removes food entries with duplicate names
as
    begin
        delete from FOOD where FoodID not in(select min(FoodID) as id from food group by Name)
    end


-----------------------------------------------------------------------------------------------------

-- INSERTS

-----------------------------------------------------------------------------------------------------
