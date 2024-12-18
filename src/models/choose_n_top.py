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
from src.models.predict_model import write_prediction_to_log
from sklearn.metrics import pairwise_distances
from itertools import combinations
#get input file

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filename', type=str)
@click.option('--output_dir', default="./models/predictions",type=click.Path(exists=True))
@click.option('--algorithm', default="alg_1",type=str)
@click.option('--n', default=6,type=int)
def main(input_filepath, output_filename,output_dir,algorithm,n):
	df = pd.read_csv(input_filepath)
	df = df.sort_values("EI_Score",ascending=False)

	if algorithm == "alg_1":
		print(f'Utilizing algorithm {algorithm}.')
		print(f'Choosing top {n} items...')
		df_top_n = get_n_best(n,df)
		print(df_top_n.head())
		df_top_n.to_csv(output_dir+"/"+output_filename+".csv")
		print(f'Wrote selections to {output_dir+"/"+output_filename+".csv"}.')

def choose_max_dist(n,df,random=False, features = ['Sn','Ga','Fe','Cu','Ca'],score='EI_Score'):
    if random:
        print("Random Choices")
        chosen_idx = df.index[np.random.randint(0,len(df),size=n)]
        return df.loc[chosen_idx,:]
    else:
        scores = df[score].to_numpy()
        indices = []
        indices.append(np.argmin(-1*scores)) #add top EI score
        dist_arr = pairwise_distances(df[features].to_numpy()) #pairwise distances from sklearn
        print(dist_arr)
        for i in range(n-1): #for each of the n top catalyst you want

            curr_top_idx = None
            curr_top_dist = -1
            np_indices = np.array(indices)
            time0=time.time()
            for j in range(scores.shape[0]): #for each catalyst in dataset
                dist = np.sum(dist_arr[j,np_indices])
                if dist > curr_top_dist and j not in indices:
                    curr_top_idx=j
                    curr_top_dist=dist
            indices.append(curr_top_idx)
            print(f'Catalyst {curr_top_idx} added with total distance of {curr_top_dist}')
            print(f'Current indices: {indices}')
            print(f'Elapsed Time: {time.time()-time0:.3}')
        return np.array(indices)

def get_n_best(n,df):
    print(f'n: {n}')
    n_found=0
    df_return = pd.DataFrame(columns=df.columns)
    while len(df_return) < n:

        max_ei = max(df["EI_Score"])
        print(f'max_ei: {max_ei}')
        df_slice = df[df["EI_Score"] == max_ei]
        print(f'len slice: {len(df_slice)}')
        if n_found+len(df_slice) > n:
            if len(df_slice) == len(df):
                df_return=df_return.append(choose_max_dist(n,df))
            else:
                df_return=df_return.append(get_n_best(n-n_found,df_slice))
        else:
            df_return=df_return.append(df_slice)

        
        df = df[df["EI_Score"] < max_ei]
        n_found = len(df_return)
        print(f'End of it, n_found: {n_found}')
    
    print(f'Returning {n_found} items!')
    return df_return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()