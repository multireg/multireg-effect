# Accompanying source code for "Bias Identification and Analysis in NLP Models With Regression and Effect Sizes" submission



##  R Package Dependencies
In order to run the scripts, the following R packages must be installed in the system:
* relaimpo
* dominanceanalysis
* pscl
* DescTools

## Python Dependencies
This repository uses the official evaluation scripts released by [Webster et al., 2018](https://github.com/google-research-datasets/gap-coreference)
to obtain F-Scores of coreference resolution models.

* Experiment-1: Analysis of Emotion Intensity Prediction models
  * To replicate the results on Table-6:

   ```
   $ cd ./better_understanding_bias/experiments1
   $ Rscript regression_analysis_with_lmg.R <model_name>
   # <model_name> can be any of the followings: cnn,rnn,attn,svm,bert
   ```

* Experiment-2: Analysis of Coreference Resolution systems
  * To replicate the results on Table-8:

   ```
   $ cd ./better_understanding_bias/experiments2/gap-coreference
   $ python gap_scorer.py --gold_tsv gap-development.tsv --system_tsv ../predictions/gap_predictions_<model_name>.tsv
   # <model_name> can be any of the followings: lee13,clark_and_manning,wiseman,lee17,lee18,spanbert_large
   ```
  * To replicate the results on Table-9:

   ```
   $ cd ./better_understanding_bias/experiments2
   $ Rscript logistic_regression_with_da.R <model_name>
   # <model_name> can be any of the followings: lee13,clark_and_manning,wiseman,lee17,lee18,spanbert_large
   ```
# Regression analysis input file format:
Please see the paper for further details.

## Intensity Prediction (Experiment-1)
* `Col-1`: Name of the person.
* `Col-2`: Emotion intensity value predicted by the corresponding model.
* `Col-3`: Frequency of the name, as approximated by the length of theirGoogle News Skipgram vector.
* `Col-4`: Discretize age, using 40 as theyoung/old boundary.
* `Col-5`: Gender variable.
* `Col-6`: Race variable.

## Coreference Resolution (Experiment-2)
* `Col-1`: Example ID
* `Col-2`: Gender variable shared by the two named entities in the example.
* `Col-3,7`: Frequency of the correct named entity.
* `Col-4,8`: Diff is number of tokens between the correct/incorrect named entity and target pronoun.
* `Col-5,9`: States whether the correct/incorrect named entity is a single word or not.
* `Col-6,10`: Indicates whether the pronoun and named entity are in thesame sentence or not.
* `Col-11,12`: Gold labels for two named entities.
* `Col-13,14`: Predicted labels for two named entities.
