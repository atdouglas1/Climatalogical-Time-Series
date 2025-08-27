# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 13:33:09 2025

@author: atdou
"""

import numpy
import pandas
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader, Subset 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import cartopy 
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 

class unit_scaler(object):
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x
    def fit_transform(self, x):
        return x  
    def invert(self, y):
        return y
    
    
class ab_scaler(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.min = None
        self.max = None
    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
    def transform(self, x):
        return self.a + (self.b - self.a)*(x - self.min)/(self.max - self.min)
    def fit_transform(self, x):
        self.min = x.min()
        self.max = x.max()
        return self.a + (self.b - self.a)*(x - self.min)/(self.max - self.min)
    def invert(self, y):
        return self.min + (self.max - self.min)*(y - self.a)/(self.b - self.a)


class ln_ab_scaler(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.f_min = None
        self.f_max = None
    def fit(self, x):
        mini = x.min()
        maxi = x.max()
        self.f_min = numpy.sign(mini)*numpy.log(1+numpy.abs(mini))
        self.f_max = numpy.sign(maxi)*numpy.log(1+numpy.abs(maxi))
    def transform(self, x):
        f = numpy.sign(x)*numpy.log(1+numpy.abs(x))
        return self.a + (self.b - self.a)*(f - self.f_min)/(self.f_max - self.f_min)
    def fit_transform(self, x):
        mini = x.min()
        maxi = x.max()
        self.f_min = numpy.sign(mini)*numpy.log(1+numpy.abs(mini))
        self.f_max = numpy.sign(maxi)*numpy.log(1+numpy.abs(maxi))
        f = numpy.sign(x)*numpy.log(1+numpy.abs(x))
        return self.a + (self.b - self.a)*(f - self.f_min)/(self.f_max - self.f_min)
    def invert(self, z):
        y = self.f_min + (self.f_max-self.f_min)/(self.b-self.a)*(z - self.a)
        return numpy.sign(y)*(numpy.e**y-1)


class symmetric_scaler(object):
    def __init__(self):
        self.min = None
        self.max = None
    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
    def transform(self, x):
        return 2*(x-self.min)/(self.max-self.min)-1
    def fit_transform(self, x):
        self.min = x.min()
        self.max = x.max()
        return 2*(x-self.min)/(self.max-self.min)-1    
    def invert(self, y):
        return (y+1)*(self.max-self.min)/2+self.min


class ln_scaler(object):
    def __init__(self):
        self.a = -1
        self.b = 1
        self.f_min = None
        self.f_max = None
    def fit(self, x):
        mini = x.min()
        maxi = x.max()
        self.f_min = numpy.sign(mini)*numpy.log(1+numpy.abs(mini))
        self.f_max = numpy.sign(maxi)*numpy.log(1+numpy.abs(maxi))
    def transform(self, x): 
        mini = x.min()
        maxi = x.max()
        self.f_min = numpy.sign(mini)*numpy.log(1+numpy.abs(mini))
        self.f_max = numpy.sign(maxi)*numpy.log(1+numpy.abs(maxi))
        f = numpy.sign(x)*numpy.log(1+numpy.abs(x))
        return self.a + (self.b - self.a)*(f - self.f_min)/(self.f_max - self.f_min)
    def invert(self, z):
        y = self.f_min + (self.f_max-self.f_min)/(self.b-self.a)*(z - self.a)
        return numpy.sign(y)*(numpy.e**y-1)


def se_rounder(series, d):
    series_new = series.copy()
    def f(x):
        if type(x) == list or type(x) == tuple:
            x_rounded = []
            for k in range(len(x)):
                x_rounded.append(round(x[k], d))
            return x_rounded
        elif type(x) == float:
            return round(x, d)
        else:
            return x
    for row in series_new.index: 
            series_new[row] = f(series_new[row])
    return series_new


def df_rounder(dframe, d):
    dframe_new = dframe.copy()
    def f(x):
        if type(x) == list or type(x) == tuple:
            x_rounded = []
            for k in range(len(x)):
                x_rounded.append(round(x[k], d))
            return x_rounded
        elif type(x) == float:
            return round(x, d)
        else:
            return x
    for col in dframe: 
            dframe_new[col] = dframe[col].apply(f) 
    return dframe_new


class my_ARMA(object):
    def __init__(self, phi, beta, f , D, theta, N):
        """ 
        phi is list of phis, theta is list of thetas, beta is constant, 
        D is variance of white noise, N is number of points
        f is an inhomogeneity that I could add.
        initial conditions are all zeros
        """
        self.phi = phi
        self.b = beta
        self.f = f                                  # an optional inhomogeneous function
        self.D = D 
        self.theta = theta
        self.AR = N*[0]                             # setting list of AR parts of model to 0 initially
        self.AM = N*[0]                             # setting list of AM parts of model to 0 initially  
        self.X = N*[0]
        self.N = N
        self.W = numpy.random.normal(0,D**0.5,N)    # list of white noises
        for j in range(len(phi),N):
            ar = beta + self.f(j)
            for k in range(len(phi)):
               ar += phi[k]*self.X[j-1-k]
            am = self.W[j]
            for k in range(len(theta)):
               am += theta[k]*self.W[j-1-k]         # If len(theta)>len(phi), then W[j-1-k] index will be negative.  whatevs?
            self.AM[j] = am
            self.AR[j] = ar
            self.X[j] = am+ar 
    def df_lags(self, k):
        df = {}
        for j in range(k+1):
            df[str(j)] = self.X[k+1-j-1:-1-j]
        df = pandas.DataFrame(df)
        return df
    def plot_X(self):
        pyplot.figure()
        T = range(self.N)
        R = self.X
        pyplot.plot(T,R)
        pyplot.xlim(0,self.N)
        pyplot.title("phi = " + str(self.phi) + ", beta = " + str(self.b) + ", D = " + str(self.D) + \
                     ", theta = " + str(self.theta))
        pyplot.xlabel("T")
        pyplot.ylabel("X")
#        pyplot.ylim(0,100)
        pyplot.grid()
        pyplot.show()
    def plot_lag(self, k):
        pyplot.figure()
        D = self.X[:self.N-k]
        R = self.X[k:self.N]
        pyplot.scatter(D,R)
#        pyplot.xlim(k,N)
        pyplot.title("X_n vs. X_n-"+ str(k))
        pyplot.xlabel("X_n-"+str(k))
        pyplot.ylabel("X_n")
#        pyplot.ylim(0,100)
        pyplot.grid()
        pyplot.show()


class create_series(object):
    def __init__(self):
        self.X = None
    def fit(self, data):
        self.X = pandas.Series(data)
    def fit_vec(self, *data):
        data = list(zip(*data))
        self.X = pandas.Series(data)
    def to_array(self):
        self.X = numpy.array(self.series.values.tolist())


class create_dataframe(object):
    def __init__(self, X_seq_length, y_seq_length):
        self.X = None
        self.y = None
        self.df = None
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
    def fit(self, data):
        indices, X, y = [], [], []
        for j in range(self.X_seq_length, len(data)-self.y_seq_length+1): 
            indices.append(j)
            X_temp = [i for i in data[j-self.X_seq_length:j]]
            y_temp = data[j:j+self.y_seq_length]
            X.append(X_temp)
            y.append(y_temp)
        self.X = pandas.DataFrame(X, columns=["X_(n-"+str(j)+")" for j in range(self.X_seq_length,0,-1)], index = indices)
        self.y = pandas.DataFrame(y, columns=["y"+str(j) for j in range(self.y_seq_length)], index = indices)
        self.df = pandas.concat([self.X, self.y], axis=1)
    def fit_vec(self, *data):
        data = list(zip(*data))
        indices, X, y = [], [], []
        for j in range(self.X_seq_length, len(data)-self.y_seq_length+1): 
            indices.append(j)
            X_temp = [i for i in data[j-self.X_seq_length:j]]
            y_temp = data[j:j+self.y_seq_length]
            X.append(X_temp)
            y.append(y_temp)
        self.X = pandas.DataFrame(X, columns=["X_(n-"+str(j)+")" for j in range(self.X_seq_length,0,-1)], index = indices)
        self.y = pandas.DataFrame(y, columns=["y"+str(j) for j in range(self.y_seq_length)], index = indices)
        self.df = pandas.concat([self.X, self.y], axis=1)
    def fit_datasets(self, data_sets):
        np_arrays = []
        val_sets = data_sets.copy()
        for key in val_sets:
            df = create_dataframe(self.X_seq_length, self.y_seq_length)
            df.fit(val_sets[key])
            np_array = numpy.array(df.df)
            np_arrays.append(np_array)
        np_arrays_stacked = numpy.vstack(np_arrays)
        df = pandas.DataFrame(np_arrays_stacked)
        self.X = df.iloc[:,0:self.X_seq_length]
        self.X.columns = ["X_(n-"+str(j)+")" for j in range(self.X_seq_length,0,-1)]
        self.y = df.iloc[:,self.X_seq_length:self.X_seq_length + self.y_seq_length]
        self.y.columns = ["y"+str(j) for j in range(self.y_seq_length)]
        self.df = pandas.concat([self.X, self.y], axis=1)
    def fit_vec_datasets(self, data_sets):
        np_arrays = []
        val_sets = data_sets.copy()
        for key in val_sets:
            if type(val_sets[key]) == pandas.DataFrame:
                val_sets[key] = val_sets[key].T.values.tolist()
            df = create_dataframe(self.X_seq_length, self.y_seq_length)
            df.fit_vec(*val_sets[key])
            np_array = numpy.array(df.df)
            np_arrays.append(np_array)
        np_arrays_stacked = numpy.vstack(np_arrays)
        df = pandas.DataFrame(np_arrays_stacked)
        self.X = df.iloc[:,0:self.X_seq_length]
        self.X.columns = ["X_(n-"+str(j)+")" for j in range(self.X_seq_length,0,-1)]
        self.y = df.iloc[:,self.X_seq_length:self.X_seq_length + self.y_seq_length]
        self.y.columns = ["y"+str(j) for j in range(self.y_seq_length)]
        self.df = pandas.concat([self.X, self.y], axis=1)
    def augment_X(self, poly_columns, degree):
        def first_word(string):                         # this is for extracting the first column name of a column string, e.g., "col1*col2*col3", etc.
            if '*' not in string:                       # it's a helper function for unique_powers function
                return string
            else: 
                return string[0:string.find('*')]      
        def powers_columns(X, columns, degree):         # function finds all unique combinations of column names and their powers involved when raise the 
                                                        # whole set of them to degree poewr.
            if degree == 1:
                return columns
            else:
                previous_powers_columns = powers_columns(X, columns, degree-1)
                new_powers = []
                for elem_c in columns:
                    for elem_t in previous_powers_columns:
                        if columns.index(elem_c) <= columns.index(first_word(elem_t)):
                            new_powers.append("{:s}*{:s}".format(elem_c, elem_t))
                            X["{:s}*{:s}".format(elem_c, elem_t)] = X[elem_c]*X[elem_t]
                return new_powers   
        powers_columns(self.X, poly_columns, degree)
    def expand_X(self, expand, hues):
        def f(t):
            l_copy = list(t)
            l = list(t)
            for c in expand:
                for h in hues:
                    l.append(l_copy[c]*l_copy[h]) 
            return tuple(l)
        self.X = self.X.applymap(f)
    def remove_y(self, component):
        self.y = self.y.applymap(lambda t: tuple(v for i, v in enumerate(t) if i != component))
    def insert_constant(self, p):
        self.X["beta"] = len(self.X)*[p]
        self.df = pandas.concat([self.X, self.y], axis=1)
    def insert_placeholders(self, n, p):
        for j in range(n):
            self.X["X"+str(j)] = len(self.X)*[p]
        self.df = pandas.concat([self.X, self.y], axis=1)
    def insert_placeholders_vec(self, n, p):
        for j, col in enumerate(self.y.columns):
            series = self.y[col].tolist()
            n = len(series)
            sequences_X = list(zip(*series))
            sequences_y = []
            for k in range(len(p)):
                if p[k] == "_":
                    sequences_X[k] = list(sequences_X[k])
                else:
                    sequences_y.append(list(sequences_X[k]))
                    sequences_X[k] = [p[k]]*n
            self.X["X"+str(j)] = list(zip(*sequences_X))
            try:
                self.y[col] = list(zip(*sequences_y))
            except:
                pass
        self.df = pandas.concat([self.X, self.y], axis=1)
    def rename_X(self, new_columns):
        self.X.columns = new_columns
    def to_array(self):
        self.X = numpy.array(self.X.values.tolist())
        self.y = numpy.array(self.y.values.tolist())


class tfold(object):
   def __init__(self):
       pass
   def split_rolling_stoch(self, test_indices, X_seq_length, y_seq_length):
      for k in test_indices:
         train_indices = list(range(k-X_seq_length, k))
         test_indices = list(range(k, k+y_seq_length))
         yield (train_indices, test_indices)
   def split_expanding_stoch(self, test_indices, start, y_seq_length):
      for k in test_indices:
         train_indices = list(range(start, k)) 
         test_indices = list(range(k, k+y_seq_length))
         yield (train_indices, test_indices)
   def split_rolling_nn(self, test_indices, window, y_seq_length):
      for k in test_indices:
         train_indices = list(range(k-y_seq_length-window+1, k-y_seq_length+1))
         test_indices = [k]
         yield (train_indices, test_indices)
   def split_expanding_nn(self, test_indices, start, y_seq_length):
      for k in test_indices:
         train_indices = list(range(start, k-y_seq_length+1))
         test_indices = [k] 
         yield (train_indices, test_indices)


def create_batches(sequences, batch_size):
    from math import ceil
    num_batches = ceil(len(sequences)/batch_size)
    batches = []
    for j in range(num_batches-1):
        batches.append(sequences[j*batch_size:(j+1)*batch_size])
    last_batch = sequences[(num_batches-1)*batch_size:-1]
    batches.append(last_batch)
    return batches


class rolling_average(object):
    def __init__(self, seq, T):
        self.seq = seq
        self.T = T
        self.seq_roll = None
    def fit_predict(self, x):
        L = len(self.seq)
        half_T = int(min(x, self.T/2, L-x))
        return numpy.array(self.seq[x-half_T:x+half_T]).mean()
    def plot(self, xlim = None, ylim = None, Domain = None):
        xlim = xlim
        pyplot.figure()
        D = range(len(self.seq)) if Domain is None else Domain
        R1 = self.seq
        R2 = [self.fit_predict(x) for x in range(len(self.seq))]
        self.seq_roll = R2
        pyplot.plot(D,R1)
        pyplot.plot(D,R2)
        pyplot.title("Window = " + str(self.T))
        pyplot.xlim(xlim)
#        pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        pyplot.ylim(ylim)
        pyplot.show()


class lmfit_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, params, model, residuals, minimizer):
        self.X_seq_length = None
        self.y_seq_length = None
        self.X_seq_start = None
        self.model = model
        self.params = params
        self.new_params = None
        self.residuals = residuals
        self.minimizer = minimizer
        self.cv_results = None
    def fit(self, X_train, y_train):
        mini = self.minimizer(self.residuals, self.params, fcn_args = (X_train.values, y_train.values))
        result = mini.minimize(method = "leastsq")
        self.new_params = result.params
    def predict(self, X_test):
        return self.model(self.new_params, X_test.values)
    def cross_val_roll(self, X, y, X_seq_length, y_seq_length, test_indices):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, y_seq_length)
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length)) 
           X_test = X.loc[new_indices]
           y_test = y.loc[new_indices].values.ravel() 
           y_pred = numpy.array(self.predict(X_test)).ravel()
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])      
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, X, y, X_seq_start, y_seq_length, test_indices):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, y_seq_length)
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length))
           X_test = X.loc[test_indices]  
           y_test = y.loc[new_indices].values.ravel() 
           y_pred = numpy.array(self.predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])      
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                          "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title)            
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class exps_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within Statsmodel's ARIMA models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, optimized = True, alpha = None, beta = None, gamma = None, \
                 trend_type = None, seasonal_type = None, seasonal_periods = None):
        self.optimized = optimized
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trend_type = trend_type
        self.seasonal_type = seasonal_type
        self.seasonal_periods = seasonal_periods
        self.X_seq_start = None
        self.X_seq_length = None
        self.y_seq_length = None
        self.model = None
        self.cv_results = None
    def fit_predict(self, X_test, y_seq_length):
        X_test = X_test.values
        self.model = ExponentialSmoothing(X_test, trend = self.trend_type, seasonal = self.seasonal_type, \
                                          seasonal_periods = self.seasonal_periods)
        if self.optimized == True:
            fitted_model = self.model.fit(optimized = True)
        else:
            fitted_model = self.model.fit(smoothing_level = self.alpha, smoothing_slope = self.beta, \
                                          smoothing_seasonal = self.gamma)       
        return fitted_model.forecast(steps = y_seq_length) 
    def cross_val_roll(self, Seq, X_seq_length, y_seq_length, test_indices): 
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, self.y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, Seq, X_seq_start, y_seq_length, test_indices): 
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title)     
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class ARIMA_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within Statsmodel's ARIMA models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, p, d, q, trend = "c"):
        self.p = p
        self.q = q 
        self.d = d
        self.trend = trend
        self.X_seq_start = None
        self.X_seq_length = None
        self.y_seq_length = None
        self.X_seq_start = None
        self.model = None
        self.cv_results = None
    def fit_predict(self, X_test, y_seq_length):
        X_test = X_test.values
        self.model = ARIMA(X_test, order = (self.p, self.d, self.q), trend = self.trend)
        return self.model.fit().forecast(steps = y_seq_length) 
    def cross_val_roll(self, Seq, X_seq_length, y_seq_length, test_indices): 
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, self.y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, Seq, X_seq_start, y_seq_length, test_indices): 
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title) 
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class SARIMA_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within Statsmodel's ARIMA models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, p, d, q, P, D, Q, s, trend = "c"):
        self.p = p
        self.q = q
        self.d = d
        self.P = P
        self.Q = Q
        self.D = D
        self.s = s
        self.trend = trend
        self.X_seq_start = None
        self.X_seq_length = None
        self.y_seq_length = None
        self.model = None
        self.cv_results = None
    def fit_predict(self, X_test, y_seq_length):
        X_test = X_test.values
        self.y_seq_length = y_seq_length
        self.model = SARIMAX(X_test, order = (self.p, self.d, self.q),\
                             seasonal_order = (self.P, self.D, self.Q, self.s), trend = self.trend) 
        return self.model.fit(disp = False).forecast(steps = self.y_seq_length)
    def cross_val_roll(self, Seq, X_seq_length, y_seq_length, test_indices): 
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, self.y_seq_length) 
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, Seq, X_seq_start, y_seq_length, test_indices): 
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, self.y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test = Seq[test_indices]
           y_pred = self.fit_predict(X_test, y_seq_length) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()        
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title)         
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class ARIMAX_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within Statsmodel's ARIMA models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, p, d, q, trend = "c"):
        self.p = p
        self.q = q
        self.d = d
        self.X_seq_start = None
        self.trend = trend
        self.X_seq_length = None
        self.y_seq_length = None
        self.model = None
        self.cv_results = None
    def fit_predict(self, X_test, y_seq_length, pred_col = 0):
        self.y_seq_length = y_seq_length
        X_seq_length = len(X_test) - y_seq_length
        X_fit = X_test.iloc[:X_seq_length]
        X_fit_df = pandas.DataFrame(zip(*X_fit.tolist())).transpose()
        X_fit_endog = X_fit_df[pred_col].values
        X_fit_exog = X_fit_df.drop(pred_col, axis = 1).values
        X_test = X_test.iloc[X_seq_length:]
        X_test_df = pandas.DataFrame(zip(*X_test.tolist())).transpose()
        X_test_exog = X_test_df.drop(pred_col, axis = 1).values
        self.model = ARIMA(X_fit_endog, order = (self.p, self.d, self.q), exog = X_fit_exog, trend = self.trend)
        results = self.model.fit()
        forecast_object = results.get_forecast(steps = y_seq_length, exog = X_test_exog)
        return forecast_object.predicted_mean
    def cross_val_roll(self, Seq, X_seq_length, y_seq_length, test_indices, pred_col = 0): 
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, self.y_seq_length) 
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test_series = Seq[test_indices]
           y_test_df = pandas.DataFrame(zip(*y_test_series.tolist())).transpose()
           y_test = y_test_df[pred_col].values
           y_pred = self.fit_predict(X_test, y_seq_length, pred_col = pred_col) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, Seq, X_seq_start, y_seq_length, test_indices, pred_col = 0): 
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, self.y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test_series = Seq[test_indices]
           y_test_df = pandas.DataFrame(zip(*y_test_series.tolist())).transpose()
           y_test = y_test_df[pred_col].values
           y_pred = self.fit_predict(X_test, y_seq_length, pred_col = pred_col) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class SARIMAX_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within Statsmodel's ARIMA models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, p, d, q, P, D, Q, s, trend = "c"):
        self.p = p
        self.q = q
        self.d = d
        self.P = P
        self.Q = Q
        self.D = D
        self.s = s
        self.X_seq_start = None
        self.X_seq_length = None
        self.y_seq_length = None
        self.model = None
        self.cv_results = None
        self.trend = trend
    def fit_predict(self, X_test, y_seq_length, pred_col = 0):
        self.y_seq_length = y_seq_length
        X_seq_length = len(X_test) - y_seq_length
        X_fit = X_test.iloc[:X_seq_length]
        X_fit_df = pandas.DataFrame(zip(*X_fit.tolist())).transpose()
        X_fit_endog = X_fit_df[pred_col].values
        X_fit_exog = X_fit_df.drop(pred_col, axis = 1).values
        X_test = X_test.iloc[X_seq_length:]
        X_test_df = pandas.DataFrame(zip(*X_test.tolist())).transpose()
        X_test_exog = X_test_df.drop(pred_col, axis = 1).values
        self.model = SARIMAX(X_fit_endog, order = (self.p, self.d, self.q),  \
                           seasonal_order = (self.P, self.D, self.Q, self.s), exog = X_fit_exog, trend = self.trend)
        results = self.model.fit(disp = False)
        forecast_object = results.get_forecast(steps = y_seq_length, exog = X_test_exog)
        return forecast_object.predicted_mean    
    def cross_val_roll(self, Seq, X_seq_length, y_seq_length, test_indices, pred_col = 0): 
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_stoch(test_indices, self.X_seq_length, self.y_seq_length) 
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test_series = Seq[test_indices]
           y_test_df = pandas.DataFrame(zip(*y_test_series.tolist())).transpose()
           y_test = y_test_df[pred_col].values
           y_pred = self.fit_predict(X_test, y_seq_length, pred_col = pred_col) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, Seq, X_seq_start, y_seq_length, test_indices, pred_col = 0): 
        self.X_seq_start = X_seq_start
        self.y_seq_length = y_seq_length
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_stoch(test_indices, X_seq_start, self.y_seq_length)
        for train_indices, test_indices in folds: 
           X_test = Seq[train_indices]
           y_test_series = Seq[test_indices]
           y_test_df = pandas.DataFrame(zip(*y_test_series.tolist())).transpose()
           y_test = y_test_df[pred_col].values
           y_pred = self.fit_predict(X_test, y_seq_length, pred_col = pred_col) 
           indices.extend(test_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in test_indices[:-1]]+[numpy.mean((y_test - y_pred)**2)])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("X_seq_length", str(self.X_seq_length)), "expanding": ("X_seq_start", str(self.X_seq_start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title(kind_dict[kind][0] + " = " + kind_dict[kind][1] + ", y_seq_length = " + str(self.y_seq_length))
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class RE_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, alpha = 0):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.alpha = alpha
        self.model = None
        self.window = None
        self.start = None
        self.coefs_ = None
        self.cv_results = None
    def fit(self, X_train, y_train):
        if self.alpha == 0:
            self.model = LinearRegression(fit_intercept = False)
            self.model.fit(X_train.values, y_train.values)
            self.interecept_ = self.model.intercept_
            self.coefs_ = self.model.coef_
            self.coefs_ = pandas.DataFrame(numpy.transpose(self.coefs_), index=X_train.columns, columns = y_train.columns)
        else: 
            self.model = Ridge(fit_intercept = False, alpha = self.alpha)
            self.model.fit(X_train.values, y_train.values)
            self.intercept_ = self.model.intercept_
            self.coefs_ = self.model.coef_
            self.coefs_ = pandas.DataFrame(numpy.transpose(self.coefs_), index=X_train.columns, columns = y_train.columns)
    def predict(self, X_test): 
        predictions = []
        X_test = X_test.values
        y_pred = self.model.predict(X_test).ravel().tolist()
        return numpy.array(y_pred) 
    def cross_val_roll(self, X, y, test_indices, window):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        params_ = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, 1)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           X_test = X.loc[test_indices] 
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length))
           y_test = y.loc[test_indices[0]].values.ravel() 
           y_pred = numpy.array(self.predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])
           params_.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [self.coefs_.values.round(3)])        
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse, "params": params_}).set_index("indices")
    def cross_val_exp(self, X, y, test_indices, start):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        params_ = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, 1)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           X_test = X.loc[test_indices] 
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length)) 
           y_test = y.loc[test_indices[0]].values.ravel() 
           y_pred = numpy.array(self.predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])
           params_.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [self.coefs_.values.round(3)])        
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse, "params": params_}).set_index("indices")
    def plot_targets_predictions(self, kind, title):
        kind_dict = {"rolling": ("window", str(self.window)), "expanding": ("start", str(self.start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = list(self.cv_results["targets"])
        R2 = list(self.cv_results["predictions"])
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle="--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + kind_dict[kind][0] + " = " + kind_dict[kind][1])
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class RE_Vector_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, alpha = 0):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.alpha = alpha
        self.model = None
        self.window = None
        self.start = None
        self.coefs_ = None
        self.coefs_ = []
        self.cv_results = None
        def ravel(x):
            array = []
            for row in x:
                new_row = []
                for elem in row:
                    new_row.extend(elem)
                array.append(new_row)
            return array
        def sep(x, pred_cols=[]):
            shape = len(x[0][0])
            array_pred = []
            array_other = []
            for row in x:
                pred_row = []
                other_row = []
                for elem in row:
                    pred_row.extend([elem[j] for j in pred_cols])
                    other_row.extend([elem[j] for j in (set(range(shape)) - set(pred_cols))])
                array_pred.append(pred_row)
                array_other.append(other_row)
            return array_pred, array_other           
        self.ravel = ravel
        self.sep = sep
    def fit(self, X_train, y_train, pred_cols = []):
        X_list = X_train.values.tolist()
        y_list = y_train.values.tolist()
        X_array = numpy.array(self.ravel(X_list))
        y_array = numpy.array(self.sep(y_list, pred_cols)[1])
        X_train = numpy.hstack([X_array, y_array])
        y_train = numpy.array(self.sep(y_list, pred_cols)[0])
        if self.alpha == 0: 
            self.model = LinearRegression(fit_intercept = False)
            self.model.fit(X_train, y_train)
            self.interecept_ = self.model.intercept_
            self.coefs_.append(self.model.coef_)
        else: 
            self.model = Ridge(fit_intercept = False, alpha = self.alpha)
            self.model.fit(X_train, y_train)
            self.interecept_ = self.model.intercept_
            self.coefs_.append(self.model.coef_)
    def predict(self, X_test, y_test, pred_cols = []): 
        predictions = []
        X_list = X_test.values.tolist()
        y_list = y_test.values.tolist()
        X_array = numpy.array(self.ravel(X_list))
        y_array = numpy.array(self.sep(y_list, pred_cols)[1])
        X_test = numpy.hstack([X_array, y_array])
        y_test = numpy.array(self.sep(y_list, pred_cols)[0])
        y_pred = self.model.predict(X_test).reshape(1,self.y_seq_length, len(pred_cols)).tolist()
        return numpy.array(y_pred) 
    def cross_val_roll(self, X, y, test_indices, window, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, pred_cols = pred_cols)  
           y_pred = self.predict(X_test, y_test, pred_cols = pred_cols).reshape(1,self.y_seq_length, len(pred_cols))
           y_list = y_test.values.tolist()
           y_test = numpy.array(self.sep(y_list, pred_cols = pred_cols)[0]).reshape(1,self.y_seq_length, len(pred_cols))
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist()[0])
           predictions.extend(y_pred.tolist()[0])
           residuals.extend((y_test - y_pred).tolist()[0])
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, pred_cols = pred_cols)  
           y_pred = self.predict(X_test, y_test, pred_cols = pred_cols).reshape(1,self.y_seq_length, len(pred_cols))
           y_list = y_test.values.tolist()
           y_test = numpy.array(self.sep(y_list, pred_cols = pred_cols)[0]).reshape(1,self.y_seq_length, len(pred_cols))
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist()[0])
           predictions.extend(y_pred.tolist()[0])
           residuals.extend((y_test - y_pred).tolist()[0])
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")  
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose() 
    def plot_targets_predictions(self, kind, labels = ["a", "b", "c"], title = None):
        kind_dict = {"rolling": ("window", self.window), "expanding": ("start", self.start)}
        D = list(self.cv_results.index)
        R1 = list(zip(*self.cv_results["targets"].tolist()))
        R2 = list(zip(*self.cv_results["predictions"].tolist()))
        for k in range(len(R1)):
            pyplot.figure()
            pyplot.plot(D,R1[k],label="target_" + labels[k])
            pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
            r = int(len(self.cv_results)/self.y_seq_length)
            delta = self.y_seq_length
            for n in range(r):
                pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
            pyplot.legend()
            pyplot.grid()
            if title == None:
                pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + str(kind_dict[kind][0]) + " = " + str(kind_dict[kind][1]))
            else:
                pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()


class RES_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, coefs, intercept):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.coefs_ = coefs
        self.intercept_ = intercept
        self.cv_results = None
    def fit_predict(self, X_test): 
        X_test = X_test.values
        self.model = LinearRegression()
        self.model.coef_ = self.coefs_
        self.model.intercept_ = self.intercept_
        y_pred = self.model.predict(X_test).ravel().tolist()
        return numpy.array(y_pred) 
    def cross_val(self, X, y, test_indices): 
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           new_indices = list(range(test_indices, test_indices + self.y_seq_length)) 
           y_test = y.loc[test_indices].values.ravel() 
           y_pred = self.fit_predict(X_test)
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])      
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, title = None):
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = list(self.cv_results["targets"])
        R2 = list(self.cv_results["predictions"])
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle="--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()
        

class RES_Vector_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, coefs, intercept):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.coefs_ = coefs
        self.intercept_ = intercept
        self.cv_results = None
        def ravel(x):
            array = []
            for row in x:
                new_row = []
                for elem in row:
                    new_row.extend(elem)
                array.append(new_row)
            return array
        def sep(x, pred_cols=[]):
            shape = len(x[0][0])
            array_pred = []
            array_other = []
            for row in x:
                pred_row = []
                other_row = []
                for elem in row:
                    pred_row.extend([elem[j] for j in pred_cols])
                    other_row.extend([elem[j] for j in (set(range(shape)) - set(pred_cols))])
                array_pred.append(pred_row)
                array_other.append(other_row)
            return array_pred, array_other           
        self.ravel = ravel
        self.sep = sep
    def fit_predict(self, X_test, y_test, pred_cols): 
        self.model = LinearRegression()
        self.model.coef_ = self.coefs_
        self.model.intercept_ = self.intercept_
        X_list = X_test.values.tolist()
        y_list = y_test.values.tolist()
        X_array = numpy.array(self.ravel(X_list))
        y_array = numpy.array(self.sep(y_list, pred_cols)[1])
        X_test = numpy.hstack([X_array, y_array])
        y_test = numpy.array(self.sep(y_list, pred_cols)[0])
        y_pred = self.model.predict(X_test).reshape(1,self.y_seq_length, len(pred_cols)).tolist()
        return numpy.array(y_pred)         
    def cross_val(self, X, y, test_indices, pred_cols): 
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           y_test = y.loc[[test_indices]]
           y_pred = self.fit_predict(X_test, y_test, pred_cols)
           y_list = y_test.values.tolist()
           y_test = numpy.array(self.sep(y_list, pred_cols = pred_cols)[0]).reshape(1,self.y_seq_length, len(pred_cols))
           indices.extend(list(range(test_indices, test_indices + self.y_seq_length)))
           targets.extend(y_test.tolist()[0])
           predictions.extend(y_pred.tolist()[0])
           residuals.extend((y_test - y_pred).tolist()[0])
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, labels = ["a", "b", "c"]):
       D = list(self.cv_results.index)
       R1 = list(zip(*self.cv_results["targets"].tolist()))
       R2 = list(zip(*self.cv_results["predictions"].tolist()))
       for k in range(len(R1)):
           pyplot.figure()
           pyplot.plot(D,R1[k],label="target_" + labels[k])
           pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
           r = int(len(self.cv_results)/self.y_seq_length)
           delta = self.y_seq_length
           for n in range(r):
               pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
           pyplot.legend()
           pyplot.grid()
           pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
       pyplot.show()
    def plot_targets_predictions_as_vectors(self, axes = None, title = None):
        if axes == None:
            fig = pyplot.figure() 
            ax = pyplot.axes()
        else:
            ax = axes
        T = list(zip(*self.cv_results["targets"].tolist()))
        P = list(zip(*self.cv_results["predictions"].tolist()))
        ax.plot(T[0], T[1], label="target_", color = "red", marker = ".",)
        ax.plot(P[0], P[1], label="prediction_", linestyle="--", color = "orange")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for n in range(r):
            ax.plot(P[0][n*delta:(n+1)*delta], P[1][n*delta:(n+1)*delta], color = "black", marker = ".")
        if title == None:
            ax.set_title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
        else:
            ax.set_title(title)
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"]):  
       pyplot.figure()
       R = list(zip(*self.cv_results["residuals"].tolist()))
       for k in range(len(R)):
           pyplot.figure()
           pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
           pyplot.ylabel("count")
           pyplot.xlabel("residual")
           pyplot.title("Residual Histogram")
           pyplot.legend()
           pyplot.show()


class DE_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, alpha = 0):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.alpha = alpha
        self.model = None
        self.window = None
        self.start = None
        self.coefs_ = pandas.DataFrame(index = ["X_(n-"+str(i)+")" for i in range(self.X_seq_length,0,-1)] + ["beta"])
        self.cv_results = None
        self.counter = 1
    def fit(self, X_train, y_train):
        if self.alpha == 0:
            self.model = LinearRegression(fit_intercept = False)
            self.model.fit(X_train.values, y_train.values)
            self.interecept_ = self.model.intercept_
            self.coefs_["run "+str(self.counter)] = self.model.coef_[0]
            self.counter +=1
        else:
            self.model = Ridge(fit_intercept = False, alpha = self.alpha)
            self.model.fit(X_train.values, y_train.values)
            self.intercept_ = self.model.intercept_
            self.coefs_["run "+str(self.counter)] = self.model.coef_[0]
            self.counter +=1
    def predict(self, X_test): 
        predictions = []
        X_test = X_test.values
        for k in range(self.y_seq_length): 
            y_pred = self.model.predict(X_test).ravel().tolist()
            predictions.extend(y_pred) 
            X_test = X_test.tolist()[0]
            X_test = [X_test[1:-1] + y_pred + [X_test[-1]]]
            X_test = numpy.array(X_test)
        return numpy.array(predictions) 
    def cross_val_roll(self, X, y, test_indices, window):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        params_ = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, 1)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           X_test = X.loc[test_indices] 
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length)) 
           y_test = y["y0"].loc[new_indices].values.ravel() 
           y_pred = numpy.array(self.predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])     
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def cross_val_exp(self, X, y, test_indices, start):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        params_ = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, 1)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices] 
           y_train = y.loc[train_indices] 
           self.fit(X_train, y_train)  
           X_test = X.loc[test_indices] 
           new_indices = list(range(test_indices[0], test_indices[0] + self.y_seq_length)) 
           y_test = y["y0"].loc[new_indices].values.ravel() 
           y_pred = numpy.array(self.predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])    
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("window", str(self.window)), "expanding": ("start", str(self.start))}
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = list(self.cv_results["targets"])
        R2 = list(self.cv_results["predictions"])
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle="--", color = "orange")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + kind_dict[kind][0] + " = " + kind_dict[kind][1])
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class DE_Vector_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, alpha = 0):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.alpha = alpha
        self.model = None
        self.window = None
        self.start = None
        self.coefs_ = None
        self.coefs_ = pandas.DataFrame()
        self.cv_results = None
        self.counter = 1
        def ravel(x):
            array = []
            for row in x:
                new_row = []
                for elem in row:
                    new_row.extend(elem)
                array.append(new_row)
            return array
        self.ravel = ravel
    def fit(self, X_train, y_train, pred_cols = []):
        X_list = X_train.values.tolist()
        y_list = y_train[["y0"]].values.tolist()
        X_array = numpy.array(self.ravel(X_list))
        y_array = numpy.array(self.ravel(y_list))
        X_train = numpy.hstack([X_array, numpy.delete(y_array, pred_cols, axis=1)]) 
        y_train = y_array[:, pred_cols]
        indices = ["X_(n-"+str(i)+","+str(j)+")" for i in range(self.X_seq_length,0,-1) for j in range(y_array.shape[1])]\
                  +["beta "+str(j) for j in range(y_array.shape[1])]+["y_(0,"+str(j)+")" for j in (set(range(y_array.shape[1])) - set(pred_cols))]
        self.coefs_.index = indices
        if self.alpha == 0: 
            self.model = LinearRegression(fit_intercept = False)
            self.model.fit(X_train, y_train)
            self.interecept_ = self.model.intercept_
#            for num, pred in enumerate(pred_cols):
#                self.coefs_["run " + str(self.counter) + " | " + str(pred)] = self.model.coef_[num]
#            self.counter += 1
        else: 
            self.model = Ridge(fit_intercept = False, alpha = self.alpha)
            self.model.fit(X_train, y_train)
            self.intercept_ = self.model.intercept_
 #           for num, pred in enumerate(pred_cols):
 #               self.coefs_["run " + str(self.counter) + " | " + str(pred)] = self.model.coef_[num]
 #           self.counter += 1
    def predict(self, X_test, y_test, pred_cols = []): 
        predictions = []
        X_list = X_test.values.tolist()
        y_list = y_test.values.tolist()
        for k in range(self.y_seq_length): 
            X_array = numpy.array(self.ravel(X_list))
            y_array = numpy.array(y_list[0][k]).reshape(1,-1)
            X_test = numpy.hstack([X_array, numpy.delete(y_array, pred_cols, axis=1)])
            y_pred = self.model.predict(X_test).reshape(1,-1).tolist()
            predictions.extend(y_pred) 
            y_pred = numpy.array(y_pred).ravel()
            X_new = y_array.ravel()
            for key, value in enumerate(pred_cols): 
                X_new[value] = y_pred[key]
            X_new = tuple(X_new)
            X_list = [X_list[0][1:-1] + [X_new] + [X_list[0][-1]]] 
        return numpy.array(predictions) 
    def cross_val_roll(self, X, y, test_indices, window, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, pred_cols = pred_cols)  
           y_pred = self.predict(X_test, y_test, pred_cols)
           y_test = numpy.array(y_test.values.tolist())[0][:,pred_cols]
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, pred_cols = pred_cols)  
           y_pred = self.predict(X_test, y_test, pred_cols)
           y_test = numpy.array(y_test.values.tolist())[0][:,pred_cols]
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose() 
    def plot_targets_predictions(self, kind, labels = ["a", "b", "c"], title = None):
        kind_dict = {"rolling": ("window", self.window), "expanding": ("start", self.start)}
        D = list(self.cv_results.index)
        R1 = list(zip(*self.cv_results["targets"].tolist()))
        R2 = list(zip(*self.cv_results["predictions"].tolist()))
        for k in range(len(R1)):
            pyplot.figure()
            pyplot.plot(D,R1[k],label="target_" + labels[k])
            pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
            r = int(len(self.cv_results)/self.y_seq_length)
            delta = self.y_seq_length
            for n in range(r):
                pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
            pyplot.legend()
            pyplot.grid()
            if title == None:
                pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + str(kind_dict[kind][0]) + " = " + str(kind_dict[kind][1]))
            else:
                pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()


class DES_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, coefs):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.coefs_ = coefs
        self.cv_results = None
    def fit_predict(self, X_test): 
        predictions = [] 
        X_test = X_test.values 
        coefs_ = numpy.array(self.coefs_)
        indices = ["X_(n-"+str(i)+")" for i in range(self.X_seq_length,0,-1)] + ["beta"]
        self.coefs_ = pandas.Series(coefs_, index = indices)
        for k in range(self.y_seq_length): 
            y_pred = numpy.dot(X_test, coefs_).tolist()
            predictions.extend(y_pred)  
            X_test = X_test.tolist()[0] 
            X_test = [X_test[1:-1] + y_pred + [X_test[-1]]]
            X_test = numpy.array(X_test) 
        return numpy.array(predictions) 
    def cross_val(self, X, y, test_indices): 
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           new_indices = list(range(test_indices, test_indices + self.y_seq_length)) 
           y_test = y["y0"].loc[new_indices].values.ravel() 
           y_pred = numpy.array(self.fit_predict(X_test))
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])      
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, title = None):
        pyplot.figure()
        D = list(self.cv_results.index)
        R1 = list(self.cv_results["targets"])
        R2 = list(self.cv_results["predictions"])
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle="--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class DES_Vector_Forecaster():
    """
    Description
    -----------
    """
    def __init__(self, X_seq_length, y_seq_length, coefs):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.model_coefs_ = numpy.array(coefs)
        self.coefs = None
        self.cv_results = None
        def ravel(x):
            array = []
            for row in x:
                new_row = []
                for elem in row:
                    new_row.extend(elem)
                array.append(new_row)
            return array
        self.ravel = ravel
    def fit_predict(self, X_test, y_test, pred_cols):
        predictions = []
        X_list = X_test.values.tolist()
        y_list = y_test.values.tolist()
        dim = len(y_list[0][0])
        indices = ["X_(n-"+str(i)+","+str(j)+")" for i in range(self.X_seq_length,0,-1) for j in range(dim)]\
                  + ["beta "+str(j) for j in range(dim)]+["y_(0,"+str(j)+")" for j in (set(range(dim)) - set(pred_cols))]
        self.coefs_ = pandas.DataFrame(index = indices)
        for num, pred in enumerate(pred_cols):
            self.coefs_[str(pred)] = self.model_coefs_[num]      
        for k in range(self.y_seq_length): 
            X_array = numpy.array(self.ravel(X_list))
            y_array = numpy.array(y_list[0][k]).reshape(1,-1)
            X_test = numpy.hstack([X_array, numpy.delete(y_array, pred_cols, axis=1)])
            y_pred = [[numpy.dot(X_test[0], self.model_coefs_[j]).item() for j in range(len(pred_cols))]]
            predictions.extend(y_pred) 
            y_pred = numpy.array(y_pred).ravel()
            X_new = y_array.ravel()
            for key, value in enumerate(pred_cols): 
                X_new[value] = y_pred[key]
            X_new = tuple(X_new)
            X_list = [X_list[0][1:-1] + [X_new] + [X_list[0][-1]]] 
        return numpy.array(predictions)                 
    def cross_val(self, X, y, test_indices, pred_cols): 
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           y_test = y.loc[[test_indices]]
           new_indices = list(range(test_indices, test_indices + self.y_seq_length)) 
           y_pred = numpy.array(self.fit_predict(X_test, y_test, pred_cols))
           y_test = numpy.array(y_test.values.tolist())[0][:,pred_cols]
           indices.extend(new_indices)
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                 [numpy.mean((y_test - y_pred)**2)])      
        self.cv_results = pandas.DataFrame({"indices":indices, "targets":targets, "predictions":predictions, \
                                           "residuals": residuals, "mse_fold": mse}).set_index("indices")
    def plot_targets_predictions(self, labels = ["a", "b", "c"], title = None):
       D = list(self.cv_results.index)
       R1 = list(zip(*self.cv_results["targets"].tolist()))
       R2 = list(zip(*self.cv_results["predictions"].tolist()))
       for k in range(len(R1)):
           pyplot.figure()
           pyplot.plot(D,R1[k],label="target_" + labels[k])
           pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
           r = int(len(self.cv_results)/self.y_seq_length)
           delta = self.y_seq_length
           for n in range(r):
               pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
           pyplot.legend()
           pyplot.grid()
           if title == None:
               pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
           else:
               pyplot.title(title)
       pyplot.show()
    def plot_targets_predictions_as_vectors(self, axes = None):
        if axes == None:
            fig = pyplot.figure() 
            ax = pyplot.axes()
        else:
            ax = axes
        T = list(zip(*self.cv_results["targets"].tolist()))
        P = list(zip(*self.cv_results["predictions"].tolist()))
        ax.plot(T[0], T[1], label="target_", color = "red", marker = ".")
        ax.plot(P[0], P[1], label="prediction_", linestyle="--", color = "orange")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for n in range(r):
            ax.plot(P[0][n*delta:(n+1)*delta], P[1][n*delta:(n+1)*delta], color = "black", marker = ".")
        ax.set_title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
#        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"]):  
       pyplot.figure()
       R = list(zip(*self.cv_results["residuals"].tolist()))
       for k in range(len(R)):
           pyplot.figure()
           pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
           pyplot.ylabel("count")
           pyplot.xlabel("residual")
           pyplot.title("Residual Histogram")
           pyplot.legend()
           pyplot.show()


class RNN_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within PyTorch's functional RNN models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, rnn_layers = [], nn_layers = [], activation = "relu",\
                 scaler = (), optimizer = (), loss = "MSE", epochs = 100, batch_size = 50):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.rnn_layers = rnn_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scaler_dict = {"unit": unit_scaler, "symmetric": symmetric_scaler, "log": ln_scaler, "ab": ab_scaler} 
        self.scaler_name = scaler
        self.scaler = None
        self.optimizer_dict = {"SGD": SGD, "Adam": Adam}
        self.optimizer_name = optimizer
        self.optimizer = None
        self.loss_dict = {"MSE": MSELoss(), "MAE": L1Loss()}
        self.loss_name = loss
        self.loss = None
        self.training_mse_list = []
        self.training_mse_results = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = None
        self.start = None
#        self.training_mse_list = []
#        self.training_mse_results = None
    def fit(self, X_train, y_train, val_data = None): 
        class rnn(nn.Module): 
            def __init__(self, rnn_layers, nn_layers, activation):
               super().__init__()
               """
               note below we're taking the zeroth component of rnn_layers and similar things because PyTorch only allows 
               RNN stacks with the same number of units, so I'll make this just the first number in the list. 
               """
               self.rnn_layers = rnn_layers
               self.nn_layers = nn_layers
               self.encoder = nn.RNN(input_size = 1, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True) 
               self.decoder = nn.RNN(input_size = 1, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True)         
               self.lin_layers = nn.ModuleList() 
               self.lin_layers.append(nn.Linear(in_features = rnn_layers[0], out_features = nn_layers[0])) 
               for k in range(1,len(nn_layers)): 
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k])) 
               self.act = activation 
               def initialize_rnn_weights(rnn):
                   for name, param in rnn.named_parameters():
                       if 'weight_ih' in name:
                           nn.init.xavier_uniform_(param)  
                       elif 'weight_hh' in name:
                           nn.init.orthogonal_(param)     
                       elif 'bias' in name:
                           nn.init.zeros_(param) 
               initialize_rnn_weights(self.encoder)
               initialize_rnn_weights(self.decoder)
               for layer in self.lin_layers:
                   nn.init.xavier_uniform_(layer.weight)
                   nn.init.zeros_(layer.bias)     
            def forward(self, X_enc, X_dec):
                h0 = torch.zeros(len(self.rnn_layers), X_enc.shape[0], self.rnn_layers[0], device = X_enc.device)
                _, h_enc = self.encoder(X_enc, h0)
                y_dec, _ = self.decoder(X_dec, h_enc)              
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length]) 
                return y_dec
        self.activation = self.activation_dict[self.activation_name]
        self.loss = self.loss_dict[self.loss_name]
        self.model = rnn(rnn_layers = self.rnn_layers, nn_layers = self.nn_layers, activation = self.activation)
        self.model.to(self.device)
        self.optimizer = self.optimizer_dict[self.optimizer_name[0]](self.model.parameters(), **self.optimizer_name[1])
        self.scaler = self.scaler_dict[self.scaler_name[0]](**self.scaler_name[1])        
        X_encoder_train = torch.tensor(X_train.iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_decoder_train = torch.tensor(X_train.iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
        y_train = torch.tensor(y_train.values, dtype = torch.float).to(self.device)
        self.scaler.fit(torch.cat((X_encoder_train.flatten(), y_train.flatten()), axis = 0))
        X_encoder_train = self.scaler.transform(X_encoder_train)
        y_train = self.scaler.transform(y_train)     
 #       self.X_enc_s = X_encoder_train
 #       self.X_dec_s = X_decoder_train
 #       self.y_s = y_train
        dataset_train = TensorDataset(X_encoder_train, X_decoder_train, y_train)
        batches_train = DataLoader(dataset_train, batch_size = self.batch_size, shuffle = True)         
        if val_data != None:
           X_encoder_val = torch.tensor(val_data[0].iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
           X_decoder_val = torch.tensor(val_data[0].iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
           y_val = torch.tensor(val_data[1].values, dtype = torch.float).to(self.device)
           X_encoder_val = self.scaler.transform(X_encoder_val)
           y_val = self.scaler.transform(y_val)
#           self.X_enc_vals = X_encoder_val
#           self.X_dec_vals = X_decoder_val
#           self.y_vals = y_val
#        training_mse = []
        self.model.train()
        for epoch in range(self.epochs):
            loss_epoch = 0
            for batch in batches_train:
                self.optimizer.zero_grad()
                X_enc_batch, X_dec_batch, y_batch = batch
                y_pred = self.model(X_enc_batch, X_dec_batch)
                loss_batch = self.loss(y_pred, y_batch)
                loss_batch.backward()
                self.optimizer.step()
                window, seq_len = y_pred.shape
                loss_epoch += loss_batch*window*seq_len                        # getting the total loss per epoch
            loss_epoch /= len(y_train)*self.y_seq_length                       # getting the loss per data point per component
#            training_mse.append(loss_epoch.item()) 
            if epoch%10 == 0:
                if val_data == None: 
                    print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(loss_epoch.item()))            
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    with torch.no_grad():
                        y_pred = self.model(X_encoder_val, X_decoder_val)
                        mse_val = ((y_val - y_pred)**2).mean().item()
                        print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(numpy.round(loss_epoch.item(),5)) \
                              + "\t" + "validation_loss: " + str(numpy.round(mse_val,5)))
                    self.model.train()
            else:
                pass
        print("\n")
#        self.training_mse_list.append(training_mse) 
    def predict(self, X_test):  
        self.model.eval()
        self.model.to(self.device)
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_encoder_test = self.scaler.transform(X_encoder_test)
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test)  
            return self.scaler.invert(y)
    def cross_val_roll(self, X, y, test_indices, window):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        keys = list(test_indices)
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = [X_test, y_test])  
           y_pred = self.predict(X_test).flatten()
           y_test = torch.tensor(y_test.values, dtype = float).flatten().to(self.device)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        keys = list(test_indices)
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = [X_test, y_test])
           y_pred = self.predict(X_test).flatten()
           y_test = torch.tensor(y_test.values, dtype = torch.float).flatten().to(self.device)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def plot_targets_predictions(self, kind, title = None):
        kind_dict = {"rolling": ("window", str(self.window)), "expanding": ("start", str(self.start))}
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + kind_dict[kind][0] + " = " + kind_dict[kind][1])
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class RNN_Vector_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within keras' functional RNN models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, rnn_layers = [], nn_layers = [], activation = "relu",\
                 scalers = [], optimizer = (), loss = "MSE", epochs = 100, batch_size = 50):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.rnn_layers = rnn_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scalers_dict = {"unit": unit_scaler, "symmetric": symmetric_scaler, "log": ln_scaler, "ab": ab_scaler} 
        self.scalers_names = scalers
        self.scalers = None
        self.optimizer_dict = {"SGD": SGD, "Adam": Adam}
        self.optimizer_name = optimizer
        self.optimizer = None
        self.loss_dict = {"MSE": MSELoss(), "MAE": L1Loss()}
        self.loss_name = loss
        self.loss = None
        self.training_mse_list = []
        self.training_mse_results = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = None
        self.start = None
        self.in_size = len(self.scalers_names)
    def fit(self, X_train, y_train, pred_cols, val_data = None):
        class rnn(nn.Module): 
            def __init__(self, rnn_layers, nn_layers, activation, in_size):
               super().__init__()
               """
               note below we're taking the zeroth component of rnn_layers and similar things because PyTorch only allows 
               RNN stacks with the same number of units, so I'll make this just the first number in the list. 
               """
               self.rnn_layers = rnn_layers
               self.nn_layers = nn_layers
               self.in_size = in_size
               self.encoder = nn.RNN(input_size = in_size, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True) 
               self.decoder = nn.RNN(input_size = in_size, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True)         
               self.lin_layers = nn.ModuleList() 
               self.lin_layers.append(nn.Linear(in_features = rnn_layers[0], out_features = nn_layers[0])) 
               for k in range(1,len(nn_layers)): 
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k])) 
               self.act = activation
               def initialize_rnn_weights(rnn):
                   for name, param in rnn.named_parameters():
                       if 'weight_ih' in name:
                           nn.init.xavier_uniform_(param)  
                       elif 'weight_hh' in name:
                           nn.init.orthogonal_(param)     
                       elif 'bias' in name:
                           nn.init.zeros_(param) 
               initialize_rnn_weights(self.encoder)
               initialize_rnn_weights(self.decoder)
               for layer in self.lin_layers:
                   nn.init.xavier_uniform_(layer.weight)
                   nn.init.zeros_(layer.bias)      
            def forward(self, X_enc, X_dec):
                output_size = self.nn_layers[-1]
                h0 = torch.zeros(len(self.rnn_layers), X_enc.shape[0], self.rnn_layers[0], device = X_enc.device)
                _, h_enc = self.encoder(X_enc, h0)
                y_dec, _ = self.decoder(X_dec, h_enc)                
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length, output_size]) 
                return y_dec
        self.activation = self.activation_dict[self.activation_name]
        self.loss = self.loss_dict[self.loss_name]
        self.model = rnn(rnn_layers = self.rnn_layers, nn_layers = self.nn_layers, activation = self.activation, in_size = self.in_size)
        self.model.to(self.device)
        self.optimizer = self.optimizer_dict[self.optimizer_name[0]](self.model.parameters(), **self.optimizer_name[1])
        self.scalers = [self.scalers_dict[self.scalers_names[k][0]](**self.scalers_names[k][1]) for k in range(self.in_size)]
        X_encoder_train = torch.tensor(X_train.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_train = torch.tensor(X_train.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        y_train = torch.tensor(y_train.values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size):
            array_of_values = X_encoder_train[:,:,k].flatten()
            if k not in pred_cols:
                array_of_values = torch.cat((array_of_values, X_decoder_train[:,:,k].flatten()), axis = 0)
            else:
                array_of_values = torch.cat((array_of_values, y_train[:,:,pred_cols.index(k)].flatten()), axis = 0)
            self.scalers[k].fit(array_of_values)        
        for k in range(self.in_size): 
            X_encoder_train[:,:,k] = self.scalers[k].transform(X_encoder_train[:,:,k])
            if k not in pred_cols:
                X_decoder_train[:,:,k] = self.scalers[k].transform(X_decoder_train[:,:,k])
            else: 
                index = pred_cols.index(k)
                y_train[:,:,index] = self.scalers[k].transform(y_train[:,:,index])   
#        self.X_enc_s = X_encoder_train
#        self.X_dec_s = X_decoder_train
#        self.y_s = y_train
        dataset_train = TensorDataset(X_encoder_train, X_decoder_train, y_train)
        batches_train = DataLoader(dataset_train, batch_size = self.batch_size, shuffle = True)
        if val_data != None: 
           X_encoder_val = torch.tensor(val_data[0].iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
           X_decoder_val = torch.tensor(val_data[0].iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
           y_val = torch.tensor(val_data[1].values.tolist(), dtype = torch.float).to(self.device)
           for k in range(self.in_size): 
               X_encoder_val[:,:,k] = self.scalers[k].transform(X_encoder_val[:,:,k])
               if k not in pred_cols:
                   X_decoder_val[:,:,k] = self.scalers[k].transform(X_decoder_val[:,:,k])
               else: 
                   index = pred_cols.index(k)
                   y_val[:,:,index] = self.scalers[k].transform(y_val[:,:,index]) 
           val_data = ([X_encoder_val, X_decoder_val], y_val) 
#        self.X_enc_vals = X_encoder_val
#        self.X_dec_vals = X_decoder_val
#        self.y_vals = y_val
        self.model.train()
#        training_mse = []
        for epoch in range(self.epochs):
            loss_epoch = 0
            for batch in batches_train:
                self.optimizer.zero_grad()
                X_enc_batch, X_dec_batch, y_batch = batch
                y_pred = self.model(X_enc_batch, X_dec_batch)
                loss_batch = self.loss(y_pred, y_batch)
                loss_batch.backward()
                self.optimizer.step()
                window, seq_len, size = y_pred.shape
                loss_epoch += loss_batch*window*seq_len                        # getting the total loss per epoch
            loss_epoch /= len(y_train)*self.y_seq_length                       # getting the loss per data point per component
#            training_mse.append(loss_epoch.item()) 
            if epoch%1 == 0:
                if val_data == None: 
                    print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(loss_epoch.item()))      
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    with torch.no_grad():
                        y_pred = self.model(X_encoder_val, X_decoder_val)
                        mse_val = ((y_val - y_pred)**2).mean().item()
                        print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(numpy.round(loss_epoch.item(),5)) \
                              + "\t" + "validation_loss: " + str(numpy.round(mse_val,5)))
                    self.model.train()
            else:
                pass
        print("\n")
    def predict(self, X_test, pred_cols): 
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size): 
            X_encoder_test[:,:,k] = self.scalers[k].transform(X_encoder_test[:,:,k])
            if k not in pred_cols:
                X_decoder_test[:,:,k] = self.scalers[k].transform(X_decoder_test[:,:,k])
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test) 
            for index, k in enumerate(pred_cols):
                y[:,:,index] = self.scalers[k].invert(y[:,:,index])
        return y 
    def cross_val_roll(self, X, y, test_indices, window, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = (X_test, y_test), pred_cols = pred_cols)
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start, pred_cols): 
        indices = []
        targets = []
        predictions = []
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = (X_test, y_test), pred_cols = pred_cols)  
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def plot_targets_predictions(self, kind, labels = ["a", "b", "c"], title = None):
        kind_dict = {"rolling": ("window", self.window), "expanding": ("start", self.start)}
        D = list(self.cv_results.index)
        R1 = list(zip(*self.cv_results["targets"].tolist()))
        R2 = list(zip(*self.cv_results["predictions"].tolist()))
        for k in range(len(R1)):
            pyplot.figure()
            pyplot.plot(D,R1[k],label="target_" + labels[k])
            pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
            r = int(len(self.cv_results)/self.y_seq_length)
            delta = self.y_seq_length
            for n in range(r):
                pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
            pyplot.legend()
            pyplot.grid()
            if title == None:
                pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + str(kind_dict[kind][0]) + " = " + str(kind_dict[kind][1]))
            else:
                pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()


class RNNS_Vector_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within keras' functional RNN models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, rnn_layers = [], nn_layers = [], activation = "relu", scalers = [], weights = None):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.rnn_layers = rnn_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scalers = scalers
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_size = len(self.scalers)
        self.weights = weights
    def fit_predict(self, X_test, pred_cols):
        class rnn(nn.Module): 
            def __init__(self, rnn_layers, nn_layers, activation, in_size):
               super().__init__()
               """
               note below we're taking the zeroth component of rnn_layers and similar things because PyTorch only allows 
               RNN stacks with the same number of units, so I'll make this just the first number in the list. 
               """
               self.rnn_layers = rnn_layers
               self.nn_layers = nn_layers
               self.in_size = in_size
               self.encoder = nn.RNN(input_size = in_size, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True) 
               self.decoder = nn.RNN(input_size = in_size, hidden_size = rnn_layers[0], num_layers = len(rnn_layers), batch_first = True)         
               self.lin_layers = nn.ModuleList() 
               self.lin_layers.append(nn.Linear(in_features = rnn_layers[0], out_features = nn_layers[0])) 
               for k in range(1,len(nn_layers)): 
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k])) 
               self.act = activation 
            def forward(self, X_enc, X_dec):
                output_size = self.nn_layers[-1]
                h0 = torch.zeros(len(self.rnn_layers), X_enc.shape[0], self.rnn_layers[0], device = X_enc.device)
                _, h_enc = self.encoder(X_enc, h0)
                y_dec, _ = self.decoder(X_dec, h_enc)                
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length, output_size]) 
                return y_dec
        self.activation = self.activation_dict[self.activation_name]
        self.model = rnn(rnn_layers = self.rnn_layers, nn_layers = self.nn_layers, activation = self.activation, in_size = self.in_size)
        self.model.to(self.device)
        self.model.load_state_dict(self.weights) 
        self.model.eval()
        self.model.to(self.device)
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size): 
            X_encoder_test[:,:,k] = self.scalers[k].transform(X_encoder_test[:,:,k])
            if k not in pred_cols:
                X_decoder_test[:,:,k] = self.scalers[k].transform(X_decoder_test[:,:,k])
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test) 
            for index, k in enumerate(pred_cols):
                y[:,:,index] = self.scalers[k].invert(y[:,:,index])
        return y 
    def cross_val(self, X, y, test_indices, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           y_test = y.loc[[test_indices]]  
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.fit_predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices, test_indices + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
    def plot_targets_predictions(self, labels = ["a", "b", "c"]):
       D = list(self.cv_results.index)
       R1 = list(zip(*self.cv_results["targets"].tolist()))
       R2 = list(zip(*self.cv_results["predictions"].tolist()))
       for k in range(len(R1)):
           pyplot.figure()
           pyplot.plot(D,R1[k],label="target_" + labels[k])
           pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
           r = int(len(self.cv_results)/self.y_seq_length)
           delta = self.y_seq_length
           for n in range(r):
               pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
           pyplot.legend()
           pyplot.grid()
           pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
       pyplot.show()
    def plot_targets_predictions_as_vectors(self, axes = None, title = None):
        if axes == None:
            fig = pyplot.figure() 
            ax = pyplot.axes()
        else:
            ax = axes
        T = list(zip(*self.cv_results["targets"].tolist()))
        P = list(zip(*self.cv_results["predictions"].tolist()))
        ax.plot(T[0], T[1], label="target_", color = "red", marker = ".")
        ax.plot(P[0], P[1], label="prediction_", linestyle="--", color = "orange")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for n in range(r):
            ax.plot(P[0][n*delta:(n+1)*delta], P[1][n*delta:(n+1)*delta], color = "black", marker = ".")
        if title == None:
            ax.set_title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
        else:
            ax.set_title(title)
#        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()


class LSTM_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within PyTorch's functional RNN models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, lstm_layers = [], nn_layers = [], activation = "relu",\
                 scaler = (), optimizer = (), loss = "MSE", epochs = 100, batch_size = 50):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.lstm_layers = lstm_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scaler_dict = {"unit": unit_scaler, "symmetric": symmetric_scaler, "log": ln_scaler, "ab": ab_scaler} 
        self.scaler_name = scaler
        self.scaler = None
        self.optimizer_dict = {"SGD": SGD, "Adam": Adam}
        self.optimizer_name = optimizer
        self.optimizer = None
        self.loss_dict = {"MSE": MSELoss(), "MAE": L1Loss()}
        self.loss_name = loss
        self.loss = None
        self.training_mse_list = []
        self.training_mse_results = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = None
        self.start = None
#        self.training_mse_list = []
#        self.training_mse_results = None
    def fit(self, X_train, y_train, val_data = None): 
        class lstm(nn.Module): 
            def __init__(self, lstm_layers, nn_layers, activation):
               super().__init__()
               self.lstm_layers = lstm_layers
               self.nn_layers = nn_layers
               self.encoder = nn.LSTM(input_size = 1, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True) 
               self.decoder = nn.LSTM(input_size = 1, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True)    
               self.lin_layers = nn.ModuleList()
               self.lin_layers.append(nn.Linear(in_features = lstm_layers[0], out_features = nn_layers[0]))
               for k in range(1,len(nn_layers)):
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k]))
               self.act = activation 
               def initialize_lstm_weights(lstm):
                   for name, param in lstm.named_parameters():
                       if 'weight_ih' in name:
                           nn.init.xavier_uniform_(param)  # Glorot for input-to-hidden 
                       elif 'weight_hh' in name:
                           nn.init.orthogonal_(param)      # Orthogonal for hidden-to-hidden (matches Keras)
                       elif 'bias' in name:
                           nn.init.zeros_(param) 
                           hidden_size = param.shape[0] // 4
                           param.data[hidden_size:2 * hidden_size] = 0.3    # adjusting forget gate bias.
               initialize_lstm_weights(self.encoder)
               initialize_lstm_weights(self.decoder)
               for layer in self.lin_layers:
                   nn.init.xavier_uniform_(layer.weight)
                   nn.init.zeros_(layer.bias)
            def forward(self, X_enc, X_dec):
                h0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device) 
                c0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device) 
                _, (h_enc,c_enc) = self.encoder(X_enc, (h0,c0))
                y_dec, _ = self.decoder(X_dec, (h_enc,c_enc))                
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length]) 
                return y_dec
        self.activation = self.activation_dict[self.activation_name]
        self.loss = self.loss_dict[self.loss_name]
        self.model = lstm(lstm_layers = self.lstm_layers, nn_layers = self.nn_layers, activation = self.activation)
        self.model.to(self.device)
        self.optimizer = self.optimizer_dict[self.optimizer_name[0]](self.model.parameters(), **self.optimizer_name[1])
        self.scaler = self.scaler_dict[self.scaler_name[0]](**self.scaler_name[1])        
        X_encoder_train = torch.tensor(X_train.iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_decoder_train = torch.tensor(X_train.iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
        y_train = torch.tensor(y_train.values, dtype = torch.float).to(self.device)
        self.scaler.fit(torch.cat((X_encoder_train.flatten(), y_train.flatten()), axis = 0))
        X_encoder_train = self.scaler.transform(X_encoder_train)
        y_train = self.scaler.transform(y_train)     
        self.X_enc_s = X_encoder_train
        self.X_dec_s = X_decoder_train
        self.y_s = y_train
        dataset_train = TensorDataset(X_encoder_train, X_decoder_train, y_train)
        batches_train = DataLoader(dataset_train, batch_size = self.batch_size, shuffle = True)         
        if val_data != None:
           X_encoder_val = torch.tensor(val_data[0].iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
           X_decoder_val = torch.tensor(val_data[0].iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
           y_val = torch.tensor(val_data[1].values, dtype = torch.float).to(self.device)
           X_encoder_val = self.scaler.transform(X_encoder_val)
           y_val = self.scaler.transform(y_val)
           self.X_enc_vals = X_encoder_val
           self.X_dec_vals = X_decoder_val
           self.y_vals = y_val
#        training_mse = []
        self.model.train()
        for epoch in range(self.epochs):
            loss_epoch = 0
            for batch in batches_train:
                self.optimizer.zero_grad()
                X_enc_batch, X_dec_batch, y_batch = batch
                y_pred = self.model(X_enc_batch, X_dec_batch)
                loss_batch = self.loss(y_pred, y_batch)
                loss_batch.backward()
                self.optimizer.step()
                window, seq_len = y_pred.shape
                loss_epoch += loss_batch*window*seq_len                        # getting the total loss per epoch
            loss_epoch /= len(y_train)*self.y_seq_length                       # getting the loss per data point per component
#            training_mse.append(loss_epoch.item()) 
            if epoch%10 == 0:
                if val_data == None: 
                    print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(loss_epoch.item()))            
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    with torch.no_grad():
                        y_pred = self.model(X_encoder_val, X_decoder_val)
                        mse_val = ((y_val - y_pred)**2).mean().item()
                        print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(numpy.round(loss_epoch.item(),5)) \
                              + "\t" + "validation_loss: " + str(numpy.round(mse_val,5)))
                    self.model.train()
            else:
                pass
        print("\n")
#        self.training_mse_list.append(training_mse) 
    def predict(self, X_test):  
        self.model.eval()
        self.model.to(self.device)
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values, dtype = torch.float).unsqueeze(2).to(self.device)
        X_encoder_test = self.scaler.transform(X_encoder_test)
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test)  
            return self.scaler.invert(y)
    def cross_val_roll(self, X, y, test_indices, window):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        keys = list(test_indices)
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = [X_test, y_test])  
           y_pred = self.predict(X_test).flatten()
           y_test = torch.tensor(y_test.values, dtype = float).flatten().to(self.device)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        keys = list(test_indices)
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = [X_test, y_test])
           y_pred = self.predict(X_test).flatten()
           y_test = torch.tensor(y_test.values, dtype = torch.float).flatten().to(self.device)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()

    def plot_targets_predictions(self, kind, title):
        kind_dict = {"rolling": ("window", str(self.window)), "expanding": ("start", str(self.start))}
        D = list(self.cv_results.index)
        R1 = self.cv_results["targets"]
        R2 = self.cv_results["predictions"]
        pyplot.figure()
        pyplot.plot(D,R1,label="target")
        pyplot.plot(D,R2,label="prediction", linestyle = "--")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for k in range(r):
            pyplot.plot(D[k*delta:(k+1)*delta], R2[k*delta:(k+1)*delta], color = "black")
        pyplot.legend()
        pyplot.grid()
        if title == None:
            pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + kind_dict[kind][0] + " = " + kind_dict[kind][1])
        else:
            pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins):  
        pyplot.figure()
        pyplot.hist(self.cv_results["residuals"], bins = n_bins)
        pyplot.ylabel("count")
        pyplot.xlabel("residue")
        pyplot.title("Residue Histogram")
        pyplot.show()


class LSTM_Vector_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within keras' functional LSTM models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, lstm_layers = [], nn_layers = [], activation = "relu",\
                 scalers = [], optimizer = (), loss = "MSE", epochs = 100, batch_size = 50):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.lstm_layers = lstm_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scalers_dict = {"unit": unit_scaler, "symmetric": symmetric_scaler, "log": ln_scaler, "ab": ab_scaler} 
        self.scalers_names = scalers
        self.scalers = None
        self.optimizer_dict = {"SGD": SGD, "Adam": Adam}
        self.optimizer_name = optimizer
        self.optimizer = None
        self.loss_dict = {"MSE": MSELoss(), "MAE": L1Loss()}
        self.loss_name = loss
        self.loss = None
        self.training_mse_list = []
        self.training_mse_results = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = None
        self.start = None
        self.in_size = len(self.scalers_names)
    def fit(self, X_train, y_train, pred_cols, val_data = None):
        class lstm(nn.Module): 
            def __init__(self, lstm_layers, nn_layers, activation, in_size):
               super().__init__()
               """
               note below we're taking the zeroth component of lstm_layers and similar things because PyTorch only allows 
               LSTM stacks with the same number of units, so I'll make this just the first number in the list. 
               """
               self.lstm_layers = lstm_layers
               self.nn_layers = nn_layers
               self.in_size = in_size
               self.encoder = nn.LSTM(input_size = in_size, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True) 
               self.decoder = nn.LSTM(input_size = in_size, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True)         
               self.lin_layers = nn.ModuleList() 
               self.lin_layers.append(nn.Linear(in_features = lstm_layers[0], out_features = nn_layers[0])) 
               for k in range(1,len(nn_layers)): 
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k])) 
               self.act = activation      
               def initialize_lstm_weights(lstm):
                   for name, param in lstm.named_parameters():
                       if 'weight_ih' in name:
                           nn.init.xavier_uniform_(param)  # Glorot for input-to-hidden 
                       elif 'weight_hh' in name:
                           nn.init.orthogonal_(param)      # Orthogonal for hidden-to-hidden (matches Keras)
                       elif 'bias' in name:
                           nn.init.zeros_(param) 
                           hidden_size = param.shape[0] // 4
                           param.data[hidden_size:2 * hidden_size] = 0.3    # adjusting forget gate bias.
               initialize_lstm_weights(self.encoder)
               initialize_lstm_weights(self.decoder)
               for layer in self.lin_layers:
                   nn.init.xavier_uniform_(layer.weight)
                   nn.init.zeros_(layer.bias)                 
            def forward(self, X_enc, X_dec):
                output_size = self.nn_layers[-1]
                h0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device)
                c0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device)          
                _, (h_enc, c_enc) = self.encoder(X_enc, (h0,c0))
                y_dec, _ = self.decoder(X_dec, (h_enc, c_enc))                
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length, output_size]) 
                return y_dec          
        self.activation = self.activation_dict[self.activation_name]
        self.loss = self.loss_dict[self.loss_name]
        self.model = lstm(lstm_layers = self.lstm_layers, nn_layers = self.nn_layers, activation = self.activation, in_size = self.in_size)
        self.model.to(self.device)
        self.optimizer = self.optimizer_dict[self.optimizer_name[0]](self.model.parameters(), **self.optimizer_name[1])
        self.scalers = [self.scalers_dict[self.scalers_names[k][0]](**self.scalers_names[k][1]) for k in range(self.in_size)]
        X_encoder_train = torch.tensor(X_train.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_train = torch.tensor(X_train.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        y_train = torch.tensor(y_train.values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size):
            array_of_values = X_encoder_train[:,:,k].flatten()
            if k not in pred_cols:
                array_of_values = torch.cat((array_of_values, X_decoder_train[:,:,k].flatten()), axis = 0)
            else:
                array_of_values = torch.cat((array_of_values, y_train[:,:,pred_cols.index(k)].flatten()), axis = 0)
            self.scalers[k].fit(array_of_values)        
        for k in range(self.in_size): 
            X_encoder_train[:,:,k] = self.scalers[k].transform(X_encoder_train[:,:,k])
            if k not in pred_cols:
                X_decoder_train[:,:,k] = self.scalers[k].transform(X_decoder_train[:,:,k])
            else: 
                index = pred_cols.index(k)
                y_train[:,:,index] = self.scalers[k].transform(y_train[:,:,index])   
#        self.X_enc_s = X_encoder_train
#        self.X_dec_s = X_decoder_train
#        self.y_s = y_train
        dataset_train = TensorDataset(X_encoder_train, X_decoder_train, y_train)
        batches_train = DataLoader(dataset_train, batch_size = self.batch_size)
        if val_data != None: 
           X_encoder_val = torch.tensor(val_data[0].iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
           X_decoder_val = torch.tensor(val_data[0].iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
           y_val = torch.tensor(val_data[1].values.tolist(), dtype = torch.float).to(self.device)
           for k in range(self.in_size): 
               X_encoder_val[:,:,k] = self.scalers[k].transform(X_encoder_val[:,:,k])
               if k not in pred_cols:
                   X_decoder_val[:,:,k] = self.scalers[k].transform(X_decoder_val[:,:,k])
               else: 
                   index = pred_cols.index(k)
                   y_val[:,:,index] = self.scalers[k].transform(y_val[:,:,index])
           val_data = ([X_encoder_val, X_decoder_val], y_val) 
#        self.X_enc_vals = X_encoder_val
#        self.X_dec_vals = X_decoder_val
#        self.y_vals = y_val
        self.model.train()
#        training_mse = []
        for epoch in range(self.epochs):
            loss_epoch = 0
            for batch in batches_train:
                self.optimizer.zero_grad()
                X_enc_batch, X_dec_batch, y_batch = batch
                y_pred = self.model(X_enc_batch, X_dec_batch)
                loss_batch = self.loss(y_pred, y_batch)
                loss_batch.backward()
                self.optimizer.step()
                window, seq_len, size = y_pred.shape
                loss_epoch += loss_batch*window*seq_len                        # getting the total loss per epoch
            loss_epoch /= len(y_train)*self.y_seq_length                       # getting the loss per data point per component
#            training_mse.append(loss_epoch.item()) 
            if epoch%1 == 0:
                if val_data == None: 
                    print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(loss_epoch.item()))      
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    with torch.no_grad():
                        y_pred = self.model(X_encoder_val, X_decoder_val)
                        mse_val = ((y_val - y_pred)**2).mean().item()
                        print("epoch: " + str(epoch) + "\t" + "training_loss: " + str(numpy.round(loss_epoch.item(),5)) \
                              + "\t" + "validation_loss: " + str(numpy.round(mse_val,5)))
                    self.model.train()
            else:
                pass
        print("\n")
    def predict(self, X_test, pred_cols):  
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size): 
            X_encoder_test[:,:,k] = self.scalers[k].transform(X_encoder_test[:,:,k])
            if k not in pred_cols:
                X_decoder_test[:,:,k] = self.scalers[k].transform(X_decoder_test[:,:,k])
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test)
            for index, k in enumerate(pred_cols):
                y[:,:,index] = self.scalers[k].invert(y[:,:,index])
        return y 
    def cross_val_roll(self, X, y, test_indices, window, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_rolling_nn(test_indices, window, self.y_seq_length)
        self.window = window
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = (X_test, y_test), pred_cols = pred_cols)  
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def cross_val_exp(self, X, y, test_indices, start, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        tf = tfold() 
        folds = tf.split_expanding_nn(test_indices, start, self.y_seq_length)
        self.start = start
        for train_indices, test_indices in folds: 
           X_train = X.loc[train_indices]
           y_train = y.loc[train_indices]
           X_test = X.loc[test_indices] 
           y_test = y.loc[test_indices]
           self.fit(X_train, y_train, val_data = (X_test, y_test), pred_cols = pred_cols)  
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices[0], test_indices[0] + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices[0], test_indices[0] + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
#        self.training_mse_results = pandas.DataFrame(dict(zip(keys, self.training_mse_list))).transpose()
    def plot_targets_predictions(self, kind, labels = ["a", "b", "c"], title = None):
        kind_dict = {"rolling": ("window", self.window), "expanding": ("start", self.start)}
        D = list(self.cv_results.index)
        R1 = list(zip(*self.cv_results["targets"].tolist()))
        R2 = list(zip(*self.cv_results["predictions"].tolist()))
        for k in range(len(R1)):
            pyplot.figure()
            pyplot.plot(D,R1[k],label="target_" + labels[k])
            pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
            r = int(len(self.cv_results)/self.y_seq_length)
            delta = self.y_seq_length
            for n in range(r):
                pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
            pyplot.legend()
            pyplot.grid()
            if title == None:
                pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " \
                     + str(self.y_seq_length) + ", " + str(kind_dict[kind][0]) + " = " + str(kind_dict[kind][1]))
            else:
                pyplot.title(title)
        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()

            
class LSTMS_Vector_Forecaster():
    """
    Description
    -----------
    This class enfolds some useful features within keras' functional RNN models.  
    Trying to make it have the same methods and things as the other classes above.
    """
    def __init__(self, X_seq_length, y_seq_length, lstm_layers = [], nn_layers = [], activation = "relu", scalers = [], weights = None):
        self.X_seq_length = X_seq_length
        self.y_seq_length = y_seq_length
        self.lstm_layers = lstm_layers
        self.nn_layers = nn_layers
        self.activation_dict = {"relu": nn.ReLU(), "linear": nn.Identity(), "softplus": nn.Softplus()}
        self.activation_name = activation
        self.activation = None
        self.scalers = scalers
        self.cv_results = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_size = len(self.scalers)
        self.weights = weights
    def fit_predict(self, X_test, pred_cols):
        class lstm(nn.Module): 
            def __init__(self, lstm_layers, nn_layers, activation, in_size):
               super().__init__()
               """
               note below we're taking the zeroth component of lstm_layers and similar things because PyTorch only allows 
               LSTM stacks with the same number of units, so I'll make this just the first number in the list. 
               """
               self.lstm_layers = lstm_layers
               self.nn_layers = nn_layers
               self.in_size = in_size
               self.encoder = nn.LSTM(input_size = in_size, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True) 
               self.decoder = nn.LSTM(input_size = in_size, hidden_size = lstm_layers[0], num_layers = len(lstm_layers), batch_first = True)         
               self.lin_layers = nn.ModuleList() 
               self.lin_layers.append(nn.Linear(in_features = lstm_layers[0], out_features = nn_layers[0])) 
               for k in range(1,len(nn_layers)): 
                  self.lin_layers.append(nn.Linear(in_features = nn_layers[k-1], out_features = nn_layers[k])) 
               self.act = activation 
            def forward(self, X_enc, X_dec):
                output_size = self.nn_layers[-1]
                h0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device)
                c0 = torch.zeros(len(self.lstm_layers), X_enc.shape[0], self.lstm_layers[0], device = X_enc.device)          
                _, (h_enc, c_enc) = self.encoder(X_enc, (h0,c0))
                y_dec, _ = self.decoder(X_dec, (h_enc, c_enc))                
                batch_size, seq_length, hidden_size = y_dec.shape
                y_dec = y_dec.reshape(-1,hidden_size)
                y_dec = self.act(self.lin_layers[0](y_dec))
                for k in range(1, len(self.nn_layers)):   
                    y_dec = self.act(self.lin_layers[k](y_dec)) 
                y_dec = y_dec.reshape([batch_size, seq_length, output_size]) 
                return y_dec   
        self.activation = self.activation_dict[self.activation_name]
        self.model = lstm(lstm_layers = self.lstm_layers, nn_layers = self.nn_layers, activation = self.activation, in_size = self.in_size)
        self.model.to(self.device)
        self.model.load_state_dict(self.weights) 
        self.model.eval()
        self.model.to(self.device)
        X_encoder_test = torch.tensor(X_test.iloc[:,:self.X_seq_length].values.tolist(), dtype = torch.float).to(self.device)
        X_decoder_test = torch.tensor(X_test.iloc[:,self.X_seq_length:].values.tolist(), dtype = torch.float).to(self.device)
        for k in range(self.in_size): 
            X_encoder_test[:,:,k] = self.scalers[k].transform(X_encoder_test[:,:,k])
            if k not in pred_cols:
                X_decoder_test[:,:,k] = self.scalers[k].transform(X_decoder_test[:,:,k])
        with torch.no_grad():
            y = self.model(X_encoder_test, X_decoder_test) 
            for index, k in enumerate(pred_cols):
                y[:,:,index] = self.scalers[k].invert(y[:,:,index])
        return y 
    def cross_val(self, X, y, test_indices, pred_cols):
        indices = [] 
        targets = [] 
        predictions = [] 
        residuals = []
        mse = []
        for test_indices in test_indices: 
           X_test = X.loc[[test_indices]] 
           y_test = y.loc[[test_indices]]  
           y_test = torch.tensor(y_test.values.tolist()).squeeze(0).to(self.device)
           y_pred = self.fit_predict(X_test, pred_cols).squeeze(0)
           indices.extend(list(range(test_indices, test_indices + self.y_seq_length)))
           targets.extend(y_test.tolist())
           predictions.extend(y_pred.tolist())
           residuals.extend((y_test - y_pred).tolist())
           mse.extend([numpy.nan for k in range(test_indices, test_indices + self.y_seq_length-1)]+\
                 [((y_test - y_pred)**2).mean().item()])
        self.cv_results = pandas.DataFrame({"indices": indices, "targets": targets, "predictions": predictions, \
                                            "residuals": residuals, "mse_fold": mse}).set_index("indices")   
    def plot_targets_predictions(self, labels = ["a", "b", "c"]):
       D = list(self.cv_results.index)
       R1 = list(zip(*self.cv_results["targets"].tolist()))
       R2 = list(zip(*self.cv_results["predictions"].tolist()))
       for k in range(len(R1)):
           pyplot.figure()
           pyplot.plot(D,R1[k],label="target_" + labels[k])
           pyplot.plot(D,R2[k],label="prediction_" + labels[k], linestyle = "--")
           r = int(len(self.cv_results)/self.y_seq_length)
           delta = self.y_seq_length
           for n in range(r):
               pyplot.plot(D[n*delta:(n+1)*delta], R2[k][n*delta:(n+1)*delta], color = "black")
           pyplot.legend()
           pyplot.grid()
           pyplot.title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
       pyplot.show()
    def plot_targets_predictions_as_vectors(self, axes = None, title = None):
        if axes == None:
            fig = pyplot.figure() 
            ax = pyplot.axes()
        else:
            ax = axes
        T = list(zip(*self.cv_results["targets"].tolist()))
        P = list(zip(*self.cv_results["predictions"].tolist()))
        ax.plot(T[0], T[1], label="target_", color = "red", marker = ".")
        ax.plot(P[0], P[1], label="prediction_", linestyle="--", color = "orange")
        r = int(len(self.cv_results)/self.y_seq_length)
        delta = self.y_seq_length
        for n in range(r):
            ax.plot(P[0][n*delta:(n+1)*delta], P[1][n*delta:(n+1)*delta], color = "black", marker = ".")
        if title == None:
            ax.set_title("X_seq_len = " + str(self.X_seq_length) + ", y_seq_len = " + str(self.y_seq_length))
        else:
            ax.set_title(title)
#        pyplot.show()
    def plot_res_hist(self, n_bins, labels = ["a", "b", "c"] ):  
        pyplot.figure()
        R = list(zip(*self.cv_results["residuals"].tolist()))
        for k in range(len(R)):
            pyplot.figure()
            pyplot.hist(R[k], bins = n_bins, label = "residual_" + labels[k])
            pyplot.ylabel("count")
            pyplot.xlabel("residual")
            pyplot.title("Residual Histogram")
            pyplot.legend()
            pyplot.show()

























