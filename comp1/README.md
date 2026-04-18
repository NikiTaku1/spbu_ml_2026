# Model attempts

## 1st attempt
1st attempt: lgbm with minimal feature engineering, best kaggle score: 0.01425

## 2nd attempt
2nd attempt: tuned parameters, best kaggle score: 0.01339

## 3rd attempt
3rd attempt: changed lgbm to xgb: best kaggle score: 0.01272

## 4th attempt
4th attempt: ran pipeline_test.py (comparison of xgb with multiple non boosting models), best models: xgb, extra trees

## 5th attempt
5th attempt: ran further comparisons between xgb and et, with the latter showing better validation mape score by ~0.1%

## 6th attempt
6th attempt: extra trees, feature enginnering: change -999 to nan, impute nan with median, best kaggle score: 0.01060

## 7th attempt
7th attempt: improved feature enginnering: mapped rooms_4, removed browser and flash player related features (don't fit topic of dataset), removed strange floor values, changed date variable to date format, extracted year and month as separate variables; added more estimators to model, best kaggle score: 0.00954

---

# Results

Best kaggle public score: 0.00954
Best kaggle private score: 0.02241 (actually 0.02240)

Tried but didn't improve score: 1nn, knn, knn with group separation by cat values unique combinations, catboost

---

# Pipeline explanation

1. load libraries  
2. define mape metric  
3. load data  
4. log transform target  
5. separate ids and split target  
6. drop useless columns  
7. map rooms_4  
8. convert agreement_date to datetime, split out year and month as separate features  
9. manually define categorical columns  
10. replace -999 with nan  
11. impute missing values  
12. filter out invalid floor values (in here: -9, 221)  
13. label encoder on categoricals  
14. Convert values to sklearn format  
15. CV with KFold (5 folds)  
16. tuning hyperparemeters with GridSearchCV  
17. custom mape scorer for GridSearchCV  
18. run cv and extract best models and parameters (final pipeline only has one model and one set of parameters to reduce compute time)  
19. retrain model on full dataset  
20. Predict test and create submission file  
21. save model parameters as txt  
