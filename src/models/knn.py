import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import time
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

#get input file

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filename', type=str)
@click.option('--output_dir', default="./models/predictions",type=click.Path(exists=True))
@click.option('--weighting', default="uniform",type=str)
@click.option('--seed', default=1,type=int)
@click.option('--n_neighbors', default=5,type=int)
@click.option('--p', default=2,type=int)
@click.option('--use_sd_sample',default=False,type=bool)
@click.option('--metric',default='Y_pc',type=str)
def main(input_filepath, output_filename,output_dir,weighting,seed,n_neighbors,p,use_sd_sample,metric):

	X_grid = np.mgrid[0:8:17j, 0:8:17j,0:8:17j,0:8:17j,0:8:17j].reshape(5,-1).T
	print(f'Generated a grid of {X_grid.shape[0]} possible catalysts.')

	df_avg = pd.read_csv(input_filepath,index_col=0)

	metals = ["Sn","Ga","Fe","Cu","Ca"]
	cols = metals.copy()
	if metric == "Y_pc":
		cols.append("Y_pc")
		cols.append("Y_pc_sd")
	elif metric == "Y_1c": #lifetime yield
		cols.append("lifetime_yield")
		cols.append("lifetime_yield_sd")
	print(f'Metric: {metric}')
	expt_data = df_avg[cols].to_numpy()
	X_expt = expt_data[:,:-2]
	y_expt = expt_data[:,-2:]
	num_datapts = len(expt_data) #number of distinct inputs

	#define train, val, test splits. See https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
	#Useful for hyperparameter tuning, assessing model performance.
	X_train,X_test, y_train, y_test = train_test_split(X_expt,y_expt,test_size=0,random_state=seed)
	X_train,X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0,random_state=seed) #.25 * .8 = .2 overall fraxn for val
	for label,arr in zip(["Train","Validation","Test"],[y_train, y_val, y_test]):
	    print(f'{len(arr)} data points in {label} set.')

	print("Training and running model")

	if weighting == "uniform":
		pred_type = "KNN_uniform"
	elif weighting == "distance":
		pred_type = "KNN_distance"


	t0 = time.time()
	explore,exploit,mu,sigma,Z,pdf,cdf = EI(X_expt,y_expt,X_grid,pred_type=pred_type,surrogate_args={'K':n_neighbors,'p':p,'seed':seed,'use_sd_sample':use_sd_sample})
	t1 = time.time()
	print(f'EI Run Time {t1-t0:.5} seconds')

	df_pred = pd.DataFrame(X_grid,columns=metals)
	df_pred["Explore"] = explore
	df_pred["Exploit"] = exploit
	df_pred["EI_Score"] = df_pred["Explore"] + df_pred["Exploit"]
	df_pred["mu"] = mu
	df_pred["sigma"] = sigma
	df_pred["Z"] = Z
	df_pred["pdf"] = pdf
	df_pred["cdf"] = cdf
	df_pred.sort_values(by=["EI_Score"],ascending=False,inplace=True)

	n=20
	print(f'Successfully predicted catalysts. Top {n} catalysts displayed below.')
	print(df_pred.head(n=n))

	outfile_path = output_dir + "/" + output_filename + "_top.csv"
	df_pred.head(n=10000).to_csv(outfile_path)
	print(f'Wrote 10000 top predictions to {outfile_path}')
	outfile_path = output_dir + "/" + output_filename + "_bottom.csv"
	df_pred.tail(n=10000).to_csv(outfile_path)
	print(f'Wrote 10000 bottom predictions to {outfile_path}')

def EI(X_train,y_train,X_test,n_bootstrap=1000,n_split=20,pred_type = 'KNN_uniform',surrogate_args=None):
    """Returns the expected improvement function for all points in X_test. Implements KNN.
    Return: (explore, exploit) where both are arrays of length len(X_test)"""
    mu_star = np.max(y_train)
    if pred_type == 'KNN_uniform':
        mu, sigma = KNN_regressor(X_train,y_train,X_test,n_split=n_split,n_bootstrap=n_bootstrap,weights='uniform',args=surrogate_args)
    elif pred_type == 'KNN_distance':
        mu, sigma = KNN_regressor(X_train,y_train,X_test,n_split=n_split,n_bootstrap=n_bootstrap,weights='distance',args=surrogate_args)
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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()