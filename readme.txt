########## data preprocessing ############################

data_merger.py -> merge individual csv files in to a single file following same format (same column structure).
data_sampler.py -> sample specified number of records using stratified sampling technique. 
data_preprocess.py -> preprocess data using selected features, saves fully preprocessed data in binary format (as numpy array with format .npy)
data_preprocess_alt.py -> preprocess data using all features.
	#for the experiment that exclude one attack at a time from training set, use following preprocessing script, saves fully preprocessed data in binary format (as numpy array with format .npy)
	data_preprocessing_ex2.py -> excludes one attack at a time from training set.
	data_preprocessing_alt_ex2.py -> excludes one attack at a time from training set when all features are in use.


######### classifier run ###############################	
classifier_lr.py -> Logistic Regression (LR)  classifier
    analysis_excluding_1_attack.py -> analyze the effect of excluding the records of one attack from training set.
	
	classifier_dt.py -> Decision Tree (DT)
	classifier_rf.py -> Random Forest (RF)
	classifier_et -> Extra Trees (ET)
	classifier_gradient_boosting.py - > Gradient Boosting (GB)
	classifier_adaboost.py -> Adaboost
	classifier_nb.py -> Naive Bayes
	classifier_mda_qda.py -> Multiple Discriminant Analysis (MDA)
	classifier_svm.py -> Support Vector Machine (SVM)
	classifier_rough_set.py -> Rough Set (RS)
	classifier_ann.py -> Artificial Neural Network (ANN)
	classifier_ga.py -> Genetic Algorithm (GA)
	  

