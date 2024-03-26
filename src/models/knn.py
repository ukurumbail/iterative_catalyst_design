import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import time
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.predict_model import write_prediction_to_log
from src.models.util import EI, bootstrap, generate_grid, generate_prediction_array
from src.models.constants import Constants
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
@click.option('--metric',default='sqrtY0Y_pc',type=str)
@click.option('--grid',default='coarse',type=str)
def main(input_filepath, output_filename,output_dir,weighting,seed,n_neighbors,p,use_sd_sample,metric,grid):
	X_grid = generate_grid(grid)

	df = pd.read_csv(input_filepath,index_col=0)

	metals = Constants().METALS

	df_X = df[metals]

	X_train = df[metals].to_numpy()
	y_train = df[[metric,f'{metric}_sd']].to_numpy()

	print("Training and running model")

	if weighting == "uniform":
		pred_type = "KNN_uniform"
	elif weighting == "distance":
		pred_type = "KNN_distance"


	t0 = time.time()
	EI_out = EI(X_train,y_train,X_grid,pred_type=pred_type,surrogate_args={'K':n_neighbors,'p':p,'seed':seed,'use_sd_sample':use_sd_sample})
	t1 = time.time()
	print(f'EI Run Time {t1-t0:.5} seconds')

	df_pred = generate_prediction_array(X_grid,EI_out)

	n=20
	print(f'Successfully predicted catalysts. Top {n} catalysts displayed below.')
	print(df_pred.head(n=n))

	outfile_path = output_dir + "/" + output_filename + "_top.csv"
	df_pred.head(n=100).to_csv(outfile_path)
	print(f'Wrote 100 top predictions to {outfile_path}')
	outfile_path = output_dir + "/" + output_filename + "_bottom.csv"
	df_pred.tail(n=100).to_csv(outfile_path)
	print(f'Wrote 100 bottom predictions to {outfile_path}')

	write_prediction_to_log(input_filepath,output_filename,"knn",{"grid_type":grid,
																"weighting":weighting,
																"metric":metric,
																"n_neighbors":n_neighbors,
																"p":p,
																"seed":seed,
																"use_sd_sample":use_sd_sample})









if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()