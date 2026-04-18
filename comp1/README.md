1st attempt: lgbm with minimal feature engineering, best kaggle score: 0.01425

2nd attempt: tuned parameters, best kaggle score: 0.01339

3rd attempt: changed lgbm to xgb: best kaggle score: 0.01272

4th attempt: ran pipeline_test.py (comparison of xgb with multiple non boosting models), best models: xgb, extra trees

5th attempt: ran further comparisons between xgb and et, with the latter showing better validation mape score by ~0.1%

6th attempt: extra trees, feature enginnering: change -999 to nan, impute nan with median, best kaggle score: 0.01060

7th attempt: improved feature enginnering: mapped rooms_4, removed browser and flash player related features (don't fit topic of dataset), removed strange floor values, changed date variable to date format, extracted year and month as separate variables; added more estimators to model, best kaggle score: 0.00954

Best kaggle private score: 0.02241 (actually 0.02240)

Tried but didn't improve score:
-1nn
-knn
-knn with group separation by cat values unique combinations
-catboost
