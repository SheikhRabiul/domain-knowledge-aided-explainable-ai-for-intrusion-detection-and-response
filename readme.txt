process_data.py -> preprocess data by running following 4 files
	data_extract.py -> read the dataset; store it in sqlite database.
	data_transform.py -> clean data; sample data
	data_load.py -> move cleaned and sampled data from raw tables to main table. Also write into data/data_preprocessed.csv.
	data_conversion.py -> remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
		save the fully processed data as numpy array (binary: data/____.npy)
    data_conversion_alt.py -> also make numerical version of original and resampled data. 

run_classifiers.py
	classifier_lr.py -> Logistic Regression (LR)
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
	  

