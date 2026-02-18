# Helper Files
These files are here to help.

## conv_entire_dir.sh
This is a script that recursively converts every `.sigmf-meta` `.sigmf-data` file in the directory into a binary file. containing the IQ sample pairing. 

Currently it is not used in the workflow, but it's kept here just in case. 

## conv_simgf_to_iq_csv.py
This converts `.sigmf-meta` `.sigmf-data` pairings into csv files of the IQ samples. 

## plotting_iq.ipynb
This notebook simply plots IQ files in a pretty manner. 

See the ./../finished-plots folder for plots. 

## preprocessing.py
This file is used in the current workflow (01/07/2026). It holds all the importnat preprocessing functions used by all the models. 

## util.py
This file is used in the current workflow (01/07/2026). This file holds extra functions that are useful and used by some of the models. 