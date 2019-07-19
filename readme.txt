#Experiment 1
########## data preprocessing ############################
process_data.py - > preprocess data for experiment 1.
    data_merger.py -> merge individual csv files in to a single file following same format (same column structure).
    data_sampler.py -> sample specified number of records using stratified sampling technique. 
    data_preprocess_all_features.py -> preprocess data using all features, saves fully preprocessed data in binary format (as numpy array with format .npy)
    data_preprocess_selected_features.py -> preprocess data using selected features, saves fully preprocessed data in binary format (as numpy array with format .npy)
    data_preprocess_domain_features.py -> preprocess data using domain features, saves fully preprocessed data in binary format (as numpy array with format .npy)
	#for the experiment 2 that exclude one attack at a time from training set, use following preprocessing script, saves fully preprocessed data in binary format (as numpy array with format .npy)
	data_preprocessing_ex2_all_features.py -> excludes one attack at a time from training set with all features and test it against the original test set.
	data_preprocessing_ex2_selected_features.py -> excludes one attack at a time from training set with selected features and test it against the original test set.
	data_preprocessing_ex2_domain_features.py -> excludes one attack at a time from training set with domain features and test it against the original test set.
	
######### classifier run ###############################	
run_classifiers.py -> run all classifiers one by one
    classifier_lr.py -> Logistic Regression (LR)  classifier
	classifier_dt.py -> Decision Tree (DT)
	classifier_rf.py -> Random Forest (RF)
	classifier_et -> Extra Trees (ET)
	classifier_gradient_boosting.py - > Gradient Boosting (GB)
	classifier_adaboost.py -> Adaboost
	classifier_nb.py -> Naive Bayes
	classifier_mda_qda.py -> Multiple Discriminant Analysis (MDA)
	classifier_svm.py -> Support Vector Machine (SVM)
	classifier_ann.py -> Artificial Neural Network (ANN)
	  

#Experiment 2
########## data preprocessing ############################
process_data_exp2.py -> preprocess data fro experiment 2. 
    data_preprocessing_ex2_all_features.py -> excludes one attack at a time from training set with all features and test it against the original test set.
	data_preprocessing_ex2_selected_features.py -> excludes one attack at a time from training set with selected features and test it against the original test set.
	data_preprocessing_ex2_domain_features.py -> excludes one attack at a time from training set with domain features and test it against the original test set.
	#before running above three preprocessing script (or the combined one process_data_exp2.py), make necessary changes in config.txt file.
config.txt -> set use_resample = 1 to use resampled data, use feature_set =1 to use all features, 2 for selected features, and 3 for domain features.

######### classifier run ###############################	
#run classifier as before by making necessary changes in  config.txt -> set use_resample = 1 to use resampled data, use feature_set =1 to use all features, 2 for selected features, and 3 for domain features.
#after running a classifier, call analysis_excludeing_1_attack.py to analyze the predicted result where the training data for an attack is absent. 
#for experiment 2, the batch file: 
run_classifiers_exp2.py -> can be used to do the preprocessing and prediction together with all configurations. 
