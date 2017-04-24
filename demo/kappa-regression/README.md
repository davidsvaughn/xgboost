Guide for Kappa Regression
=====

This demonstrates how to use XGBoost Kappa Regression for the [Kaggle Prudential Life Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment) 

1. Download the data from [here](https://www.kaggle.com/c/prudential-life-insurance-assessment/data)
```bash
cd ../..
make
```

2. Put training.csv test.csv on folder './data' (you can create a symbolic link)

3. Run ./run.sh

Speed
=====
speedtest.py compares xgboost's speed on this dataset with sklearn.GBM


Using R module
=====
* Alternatively, you can run using R, higgs-train.R and higgs-pred.R. 

