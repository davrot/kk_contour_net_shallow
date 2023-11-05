Folder minimal_architecture:

1. config.json:
* json file with all configurations and cnn parameters 

2. training_loop.sh:
* bash script to train the 64 cnns 


3. get_trained_models:
* searches for the saved trained models in a directory
* chooses model based on the largest saved epoch in the save-name 


4. pfinkel_performance_test64:
* load all models extracted by 'get_trained_models'
* test them on all stimulus conditions
* sort their performances either after number of free parameters, or architecture