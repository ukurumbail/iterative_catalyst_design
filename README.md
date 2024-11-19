# *iterative_catalyst_design*
==============================

A software package that predicts improved catalysts based on experimental and computational data.

Usage:

## (1) Extracting Raw Data

Raw data from our reactors typically contains lots of meta-data such as reactor logs. The first step is to extract the meaningful data from the raw data folders. Some defitions:

* Reaction: One run of the 6-flow system. Data for any given reaction is accessible from ./data/raw/ with each reaction folder named by the reaction ID #.
* Reactor: One reactor. The 6-flow system has 6 reactors so 1 reaction can contain data from 6 reactors. 

The data for all reactors for a given reaction are stored in a common Excel file in the raw data folder. 

The list of reactions you wish to bring in for a given processed dataset should be stored at "./data/raw/Runs_To_Analyze" as a .txt file with a format identical to the ones already there.

For example:

`python ./src/data/make_dataset.py "Round 8 Predictions.txt" Round8-Standardized --cleanup=True --averaging=True`

will take the file "./data/raw/Runs_To_Analyze/Round 8 Predictions.txt", load in the raw datafiles with folder names listed in that text file and parse through the raw data to create your dataset. 

You should always have --cleanup=True because this eliminates data that, for one reason or another, we have chosen to exclude (e.g. it was data being used for transport limitation testing). --averaging=True generates another dataset with averaged values.

This will produce output files under ./data/processed with the names "1-preprocessed_Round8-Standardized.csv", "2-cleanedup_Round8-Standardized.csv", and "3-averaged_Round8-Standardized.csv." You should use file #2 if you want the individual datapoints (e.g. you are incorporating uncertainty) and file #3 if you are using the averaged datapoints to make predicitons (e.g. if a catalyst was tested 3x the averaged would average the 3 to produce 1 datapoint whereas the cleanedup would report 3 separate datapoints).

## (2) Making Predictions

Once you have processed data you can then make predictions by running each model script. Examples are below:

`python ./src/models/knn.py "./data/processed/3-averaged_Round8-Standardized.csv" knn-dense-round-9-predictions --grid=dense`

`python ./src/models/gaussian_process.py "./data/processed/3-averaged_Round8-Standardized.csv" gp-dense-round-9-predictions --grid=dense`

`python ./src/models/gp_white_kernel.py "./data/processed/2-cleanedup_Round8-Standardized.csv" go-white-dense-round-9-predictions --grid=dense`

All other models in the project can be ignored / aren't supported. knn and gaussian_process used averaged data (file #3 above) whereas gp_white_kernel uses cleaned up data (file #2 above). This will output model predictions at ./models/predictions along with giving you some details in the command line interface. 

Any questions please feel free to reach out to ukurumbail@wisc.edu or the corresponding author of the associated manuscript. Thanks!

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources. (NOT USED)
    │   ├── interim        <- Intermediate data that has been transformed. (NOT USED)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (NOT USED)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries 
    │   ├── predictions    <- Predictions from the models as well as a log of parameters and inputs for each set of predictions.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. (NOT USED)
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. (NOT USED)
    │   └── figures        <- Generated graphics and figures to be used in reporting (NOT USED)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt` (NOT USED)
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project. 
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py <- Allows you to convert the raw data in our specific auto_rxn output format into processed data for the ML model.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling (NOT USED)
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations (NOT USED)
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io (NOT USED)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
