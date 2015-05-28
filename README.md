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
    1a: doc2csv
        possible input: excel, txt, json
        arguments: infile, outfile, configfile specifying the columns
        uses: utils
        output: csv with fields doc_id, user_id, date, time, user, text
    2: frog_data
        input: csv-file
        arguments: tokenize only --> ucto
        uses: datahandler, utils
        output: csv-file with extra (frogged) column
    3: format_instances
        - remove 
            * by string (e.g. RT)
            * by time window
            * by end hashtag 
        - add label
        - combine files
        uses: datahandler
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
    Datahandler
    Featurizer
    Classifier
    Evaluation
    Experiment

Functions:
    Utils

Procedures:
    doc2csv
    frog_data
