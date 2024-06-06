import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import RBF
from src.models.constants import Constants
from scipy.stats import norm

def get_Xy(df,metric='sqrtY0Y_pc',SD=False):

    #generate X
    X = df[Constants().METALS]
    X=X.copy(deep=True)
    diversity = X.astype(bool).sum(axis=1) #counts non-zero in a given row https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
    loading = X.sum(axis=1) 
    X['diversity']=diversity
    X['loading']=loading

    #generate y
    if SD:
        y = df[[metric,metric+'_SD']]
    else:
        y = df[metric]
    y=y.copy(deep=True)
    
    return X,y

def generate_grid(grid):
	if grid == 'dense':
		X_grid = np.mgrid[0:8:17j, 0:8:17j,0:8:17j,0:8:17j,0:8:17j].reshape(5,-1).T
	elif grid == 'coarse':
		X_grid = []
		vals = [0,1,4,8]
		for i in vals:
			for j in vals:
				for k in vals:
					for l in vals:
						for m in vals:
							X_grid.append([i,j,k,l,m])
		X_grid = np.asarray(X_grid)
	print(f'Generated a grid of {X_grid.shape[0]} possible catalysts.')
	return X_grid

def split(X,y,val=0.0,test=0.0,seed=1):
	#define train, val, test splits. See https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
	#Useful for hyperparameter tuning, assessing model performance.
	X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=float(test),random_state=seed)
	X_train,X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=float(val),random_state=seed) #.25 * .8 = .2 overall fraxn for val
	for label,arr in zip(["Train","Validation","Test"],[y_train, y_val, y_test]):
		print(f'{len(arr)} data points in {label} set.')

	return (X_train,y_train),(X_val,y_val),(X_test),(y_test)

def bootstrap(X,y,n=1000,seed=1,use_sd_sample=False):
	"""Produces sampled experimental data of n points. It does this by randomly sampling existing data with repeats possible. 
	Next, it assumes lifetime yield is normally distributed and uses the AVG and SD values for a given experimental datapoint
	to estimate the lifetime yield of a given datapoint."""
	rng = np.random.default_rng(seed=seed)
	datapts = rng.integers(low=0,high=len(X),size=n) #randomly get n sample indices
	if use_sd_sample:
		lifetime_yield = rng.normal(loc=y[datapts][:,0],scale=y[datapts][:,1]) #calculate lifetime yield by sampling normal distribution for each catalyst
	else:
		lifetime_yield = rng.normal(loc=y[datapts][:,0]) #calculate lifetime yield by sampling normal distribution for each catalyst

	return X[datapts],lifetime_yield #return the metal loadings and lifetime yield estimate

def EI(X_train,y_train,X_test,n_bootstrap=1000,n_split=20,pred_type = 'KNN_uniform',surrogate_args=None):
	"""Returns the expected improvement function for all points in X_test. Implements KNN.
	Return: (explore, exploit) where both are arrays of length len(X_test)"""
	mu_star = np.max(y_train[:,0])
	if pred_type == 'KNN_uniform':
		mu, sigma = KNN_regressor(X_train,y_train,X_test,n_split=n_split,n_bootstrap=n_bootstrap,weights='uniform',args=surrogate_args)
	elif pred_type == 'KNN_distance':
		mu, sigma = KNN_regressor(X_train,y_train,X_test,n_split=n_split,n_bootstrap=n_bootstrap,weights='distance',args=surrogate_args)
	elif pred_type == 'GPR':
		mu, sigma = GP_regressor(X_train,y_train,X_test,args=surrogate_args)
	Z = (mu-mu_star)/sigma #element-wise
	explore = sigma*norm.pdf(Z)
	exploit = norm.cdf(Z)*sigma*Z #element-wise
	
	return (explore,exploit,mu,sigma,Z,norm.pdf(Z),norm.cdf(Z)) 

def KNN_regressor(X_train,y_train,X_test,n_bootstrap=1000,n_split=20,weights='uniform',args=None):
	"""
	Implements the KNN Regressor with selected options of weight. Bootstraps the training data (default=1000 samples)
	and splits it into a chosen number of arrays. Returns the average and std dev of the predictions.
	Default weighting of KNN Regressor is 'uniform', but 'distance' is also possible
	Return (mu {length = len(X_test)}, sigma {length = len(X_test)})"""
	if args == None:
		K = 5
		p = 2
		seed = 1
		use_sd_sample = False
	else:
		K = args['K']
		p = args['p']
		seed = args['seed']
		use_sd_sample=args['use_sd_sample']
	X,y = bootstrap(X_train,y_train,n=n_bootstrap,seed=seed,use_sd_sample=args['use_sd_sample'])
	Xs = np.array_split(X,n_split)
	ys = np.array_split(y,n_split)
	print("KNN Weighting: {}".format(weights))
	print("Using SD samples: {}".format(use_sd_sample))
	predictions = np.asarray([KNeighborsRegressor(weights=weights,n_neighbors=K,p=p).fit(Xs[i],ys[i]).predict(X_test) for i in range(n_split)])
#     print(normaltest(predictions,axis=0))
	mu = np.average(predictions,axis=0)
	sigma = np.std(predictions,axis=0)

	return mu,sigma



def GP_regressor(X_train,y_train,X_test,n_bootstrap=1000,n_split=20,args=None,verbose=True):
	"""
	Implements the Gaussian Process Regressor with selected options of weight. Bootstraps the training data (default=1000 samples)
	and splits it into a chosen number of arrays. Returns the average and std dev of the predictions.
	Default weighting of KNN Regressor is 'uniform', but 'distance' is also possible
	Return (mu {length = len(X_test)}, sigma {length = len(X_test)})"""

	if args == None:
		kernel_type = 'RBF'
		seed = 1
		use_sd_sample = False
	else:
		kernel_type = args['kernel_type']
		seed = args['seed']
		use_sd_sample=args['use_sd_sample']

	if kernel_type == 'RBF':
		kernel = 1*RBF(length_scale=1.0,length_scale_bounds=(1e-2,1e2))
	else:
		raise NotImplementedError(f'Kernel {kernel_type} not implemented yet')
	if use_sd_sample:
		alpha = y_train[:,1]
	else:
		alpha = 1e-10 #the init value of default GPR in sklearn

	gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5,alpha=alpha)
	gp.fit(X_train,y_train[:,0])
	if verbose:
		print("Trained Kernel: ",gp.kernel_)

	mu, sigma= gp.predict(X_test, return_std=True)

	return mu, sigma


def generate_prediction_array(X_grid,EI_out):
	explore,exploit,mu,sigma,Z,pdf,cdf = EI_out
	df_pred = pd.DataFrame(X_grid,columns=Constants().METALS)
	df_pred["Explore"] = explore
	df_pred["Exploit"] = exploit
	df_pred["EI_Score"] = df_pred["Explore"] + df_pred["Exploit"]
	df_pred["mu"] = mu
	df_pred["sigma"] = sigma
	df_pred["Z"] = Z
	df_pred["pdf"] = pdf
	df_pred["cdf"] = cdf
	df_pred.sort_values(by=["EI_Score"],ascending=False,inplace=True)
	return df_pred
