import numpy as np
import pandas as pd

from mendeleev import element
from sqlalchemy import exc

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import time
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.predict_model import write_prediction_to_log
from src.models.util import generate_grid,split,EI,generate_prediction_array
from src.models.constants import Constants
#get input file

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filename', type=str)
@click.option('--output_dir', default="./models/predictions",type=click.Path(exists=True))
@click.option('--kernel_type', default="RBF",type=str)
@click.option('--seed', default=1,type=int)
@click.option('--use_sd_sample',default=False,type=bool)
@click.option('--metric',default='sqrtY0Y_pc',type=str)
@click.option('--grid',default='coarse',type=str)
def main(input_filepath, output_filename,output_dir,kernel_type,seed,use_sd_sample,metric,grid):
	metals = Constants().METALS
	X_grid = generate_grid(grid)
	X_grid = pd.DataFrame(X_grid,columns=metals)
	diversity = X_grid.astype(bool).sum(axis=1) #counts non-zero in a given row https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
	loading = X_grid.sum(axis=1)
	X_grid['diversity'] = diversity + 1.0 #include Pt
	X_grid['loading'] = loading + 1.0 #include Pt
	print(X_grid.head())
	X_grid = X_grid.to_numpy()
	df = pd.read_csv(input_filepath,index_col=0)

	df_features,df_targets = featurize(df)

	X = df_features.to_numpy()
	y = df_targets[[metric,f'{metric}_sd']].to_numpy()

	# (X_train,y_train),(X_val,y_val),(X_test),(y_test) = split(X,y,seed=seed)
	X_train,y_train = X,y

	print("Training and running model")
	print(f'Number of features: {X_train.shape[1]} Number of Training Points: {X_train.shape[0]}')


	t0 = time.time()
	EI_out = EI(X,y,X_grid,pred_type='GPR',surrogate_args={'kernel_type':kernel_type,'seed':seed,'use_sd_sample':use_sd_sample})
	t1 = time.time()
	print(f'EI Run Time {t1-t0:.5} seconds')

	df_pred = generate_prediction_array(X_grid[:,:-2],EI_out) #remove diversity, loading



	n=20
	print(f'Successfully predicted catalysts. Top {n} catalysts displayed below.')
	print(df_pred.head(n=n))

	outfile_path = output_dir + "/" + output_filename + "_top.csv"
	df_pred.head(n=100).to_csv(outfile_path)
	print(f'Wrote 100 top predictions to {outfile_path}')
	outfile_path = output_dir + "/" + output_filename + "_bottom.csv"
	df_pred.tail(n=100).to_csv(outfile_path)
	print(f'Wrote 100 bottom predictions to {outfile_path}')

	if grid == "coarse":
		outfile_path = output_dir + "/" + output_filename + "_all.csv"
		df_pred.head(n=1024).to_csv(outfile_path)
		print(f'Wrote all predictions to {outfile_path}')

	write_prediction_to_log(input_filepath,output_filename,"gp",{"grid_type":grid,
																"kernel_type":kernel_type,
																"metric":metric,
																"seed":seed,
																"use_sd_sample":use_sd_sample})

def get_elements(columns):
	elements = []
	for name in columns:
		try: 
			element(name) #check for error
			elements.append(name)
		except exc.NoResultFound:
			continue
		except ValueError:
			continue
	return elements

def blacklist(df,blacklist = ['Mn','Zn']):
	return df[[i for i in df.columns if i not in blacklist]]
	#removes metals you don't want
	blacklist

def featurize(df,include_alumina_ratio=False):
	#returns a feature array and a target array
	elements = get_elements(df.columns)
	df_target = df[[i for i in df.columns if i not in elements]]
	df_features = df[elements]
	df_features = blacklist(df_features) #removes Mn, Zn, other non-utilized elements. Minimizes # features
	metals = df_features.columns
	diversity = df_features.astype(bool).sum(axis=1) #counts non-zero in a given row https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
	loading = df_features.sum(axis=1)
	if include_alumina_ratio:
		#Compute ratio of each metal to alumina
		basis = 1 #g total
		wt_pt = .01 #Pt loading, fractional
		mol_pt = wt_pt*basis / element('Pt').atomic_weight #moles of Pt per masis
		metal_dict = {metal:[] for metal in metals} #ratio to alumina

		for _,ratios in df_features.iterrows():
			total_metal = sum([element(metal).atomic_weight*ratios[metal] for metal in metals])*mol_pt
			mass_alumina = basis - total_metal
			mol_alumina = mass_alumina/(element('Al').atomic_weight*2+element('O').atomic_weight*3)
			for metal in metals:
				metal_dict[metal].append(ratios[metal]*mol_pt/mol_alumina) #molar ratio of metal i to alumina

		for metal in metal_dict.keys():
			df_features[f'{metal}_alumina_ratio'] = metal_dict[metal]
		
	df_features['diversity'] = diversity
	df_features['loading'] = loading
	df_features.drop('Pt',axis=1,inplace=True)
	return df_features,df_target
if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()