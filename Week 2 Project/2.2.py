import numpy as np
import pandas as pd
import math
from scipy.stats import skew,kurtosis,t
from scipy.optimize import minimize
import statsmodels.api as sm
import seaborn as sns

class OLE:
    def __init__(self, filename):
        #Read csv data
        self.data = pd.read_csv(filename)

    def get_XY(self):
        self.X = self.data.iloc[:,0].to_numpy()
        self.Y = self.data.iloc[:,1].to_numpy()

    def get_model(self):
        self.x = np.column_stack((np.ones(len(self.X)), self.X))
        self.x_t = np.transpose(self.x)
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x_t,self.x)),self.x_t), self.Y)
        self.error = self.Y - np.matmul(self.x, self.beta)
        
        #Using OLS
        self.model = sm.OLS(self.Y, self.X).fit()

    def prn_summary(self):
        print(self.model.summary())

        #Calculate skewness and kurtosis of the error vector
        print("Skewness of error: {}".format(skew(self.error)))
        print("kurtosis of error: {}".format(kurtosis(self.error)))

        # Plot the distribution of error
        sns.displot(self.error, kde=True)

    def mle_norm(self, a_1):
        #Fit the MLE using the assumption of a T distribution of the errors
        s = a_1[0]
        b = np.transpose(a_1[1:])
        xm = self.Y - np.matmul(self.x, b)
        s2 = s*s
        mlell = -len(self.Y)/2 * math.log(s2*2*math.pi)-np.matmul(np.transpose(xm),xm)/(2*s2)
        return -mlell

    def mlell_t(self, a_2):
        s = a_2[0]
        nu = a_2[1]
        b = a_2[2:]
        xm = self.Y - self.x @ b
        mlell = t.logpdf(xm,nu,0,s).sum()
        return -mlell

    def mle_Optimization(self):
        mlell_normal = minimize(self.mle_norm, [1,1,2])
        print("Normal betas:",mlell_normal.x[1:])
        print("Normal s:{}".format(mlell_normal.x[0]))
        mlell_result = -self.mle_norm(mlell_normal.x)
        print("Normal ll:{}".format(mlell_result))
        aic_normal = 2*(1+1+1)- 2*mlell_result
        print("Normal AIC:{}".format(aic_normal))
        bic_normal = 3*math.log(len(self.Y))-2*mlell_result
        print("Normal BIC:{}".format(bic_normal))

        mlell_t_res = minimize(self.mlell_t, [1,3,1,2])
        return mlell_t_res

    def prn_optimization(self, mlell_t_res):
        print("T betas:", mlell_t_res.x[2:])
        print("T s:{}".format(mlell_t_res.x[0]))
        print("T df:{}".format(mlell_t_res.x[1]))
        mlell_t_max = - self.mlell_t(mlell_t_res.x)
        print("T ll:{}".format(mlell_t_max))
        aic_t = 2*4- 2*mlell_t_max
        print("T AIC:{}".format(aic_t))
        bic_t = 4*math.log(len(self.Y))-2*mlell_t_max
        print("T BIC:{}".format(bic_t))

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    ole = OLE("problem2.csv")
    ole.get_XY()
    ole.get_model()
    ole.prn_summary()
    mlell_t_res = ole.mle_Optimization()
    ole.prn_optimization(mlell_t_res)

    
