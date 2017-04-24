QWiKBoost: XGBoost + Quadratic Weighted Kappa optimization
--------
* XGBoost augmented for optimization of [Quadratic Weighted Kappa](https://www.kaggle.com/c/asap-aes/details/evaluation) objective function
* *ONLY* software that supports **direct**, **analytic** optimization of QWK
* code based on [this](https://www.dropbox.com/s/oj85rcradm6m56b/kappa.pdf) paper (currently under review)
* forked from XGBoost commit: [0dc68b1aefe4a09775a2b86103643f9bce6979d2](https://github.com/dmlc/xgboost/tree/0dc68b1aefe4a09775a2b86103643f9bce6979d2)
* Core module [build and installation](https://github.com/dmlc/xgboost/blob/0dc68b1aefe4a09775a2b86103643f9bce6979d2/doc/build.md) instructions (and python interface [installation](https://github.com/dmlc/xgboost/tree/0dc68b1aefe4a09775a2b86103643f9bce6979d2/python-package))
* see Kappa Regression [starter script](demo/kappa-regression)

Installation Quick Reference
--------
1. To build core module, run ```bash build.sh``` (you can also type make) from root directory
2. Install python interface with ``cd python-package; python setup.py install`` from root directory (make sure [ml_metrics](https://github.com/benhamner/Metrics) module installed with ```pip install ml_metrics```)
3. Proceed to Kappa Regression [starter script](demo/kappa-regression)