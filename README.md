# ADNEXT_predict
Framework for supervised classification experiments, applied in the context of the ADNEXT project

Experiment-dir:
    exp/ ---> rank: Data, Features, Weights, Classifier
    per setting: feature file, pickles: vocabulary;classifier, folds, results (performance, plots, featurefiles), config
    data/ ---> storage of all csv-files
    data/raw/
    data/frogged/
    data/formatted/

Pipeline
    1a: doc_2_csv
        possible input: excel, txt
        arguments: column
        output: csv with fields doc_id, user_id, date, time, user
    1b: json_2_csv
        input: json-file
        arguments: how to 
        output: 
    2: frog_csv
        input: csv-file
        arguments: tokenize only --> ucto
        uses: csv-reader
        output: csv-file with extra (frogged) column
    3: format_instances
        - remove 
            * by string (e.g. RT)
            * by time window
            * by end hashtag 
        - add label
        - combine files
        uses: csv-reader
        output: csv-file, log-file
    4: extract features
        input: csv-file (frogged or not)
        arguments: 
        uses: featurizer-class, csv-reader
        output: [class,vector], csv-file
    5: classifier
        arguments:
            classifiers
            10-fold
            - config for classifiers, make defaults in etc/ - 
            train-test
            validate
    6: report
        * precision, recall, f1
        * confusion matrix
        * top features

Classes:
    Datareader
    Featurizer
    Classifier
    Evaluation
    Experiment

Functions:
    Utils

Procedures:
    doc_2_csv
    json_2_csv
    frog_csv
