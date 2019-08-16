from sklearn import linear_model as lm
import numpy as np
import sklearn
import scipy
import numpy
import sys
import platform
print(platform.platform())
print("Python", sys.version)
print("NumPy", numpy.__version__)
print("SciPy", scipy.__version__)
print("Scikit-Learn", sklearn.__version__)

X = np.load('lasso_error_X_1.npy')
target = np.load('lasso_error_target_1.npy')

reg = lm.LassoLarsCV(cv=3)
reg.fit(X, target)
