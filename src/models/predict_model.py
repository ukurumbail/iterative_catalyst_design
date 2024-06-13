import pandas as pd

def write_prediction_to_log(input_file,output_file,model_type,params):
	"""Writes the given info to the prediction log so each entry is marked.
	input_file --> str
	output_file --> str
	model_type --> str
	params --> dictionary
	"""
	df = pd.read_csv("./models/predictions/prediction_log.csv")
	new_row = pd.DataFrame()
	new_row["Prediction_ID"] = [df["Prediction_ID"].iloc[-1]+1]
	new_row["Model_Type"] = model_type
	new_row["Input_File"] = input_file
	new_row["Output_File"] = output_file
	for i,(key,val) in enumerate(params.items()):
		new_row[f'Model_Param_Name_{i}'] = key
		new_row[f'Model_Param_Val_{i}'] = val

	df = pd.concat([df,new_row],ignore_index=True)

	print(f'Logging prediction to ./models/predictions/prediction_log.csv with id {df["Prediction_ID"].iloc[-1]}')
	df.to_csv("./models/predictions/prediction_log.csv",index=False)





