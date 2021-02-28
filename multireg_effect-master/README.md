# Accompanying source code for "Analyzing Bias in NLP Models Using Multivariate Regression and Effect Sizes" submission 
 

# Reproducing results reported in the Paper

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
  * To replicate the results on Table-4:

   ```
   $ cd ./better_understanding_bias/experiments1
   $ Rscript regression_analysis_with_lmg.R <model_name>
   # <model_name> can be any of the followings: cnn,rnn,attn,svm,bert 
   ```

* Experiment-2: Analysis of Coreference Resolution systems
  * To replicate the results on Table-5:
   
   ```
   $ cd ./better_understanding_bias/experiments2/gap-coreference
   $ python gap_scorer.py --gold_tsv gap-development.tsv --system_tsv ../predictions/gap_predictions_<model_name>.tsv
   # <model_name> can be any of the followings: lee13,clark_and_manning,wiseman,lee17,lee18,spanbert_large
   ```
  * To replicate the results on Table-6:

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

# Training Emotion Intensity Prediction models from scratch
* Setup for training (Make sure that you're at the root of the project directory):
  * Download EI-reg training data and place it into `experiments1/eip_models/data`:
    ```
    wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip
    unzip SemEval2018-Task1-all-data.zip 
    mkdir -p experiments1/eip_models/data/Ereg
    cp SemEval2018-Task1-all-data/EI-reg/training/* experiments1/eip_models/data/Ereg/.
    cp SemEval2018-Task1-all-data/EI-reg/development/* experiments1/eip_models/data/Ereg/.
    cp SemEval2018-Task1-all-data/EI-reg/test-gold/* experiments1/eip_models/data/Ereg/.
    ```
  * Download ECC data and place it into `experiments1/eip_models/data`:
     ```
     wget http://saifmohammad.com/WebDocs/EEC/Equity-Evaluation-Corpus.zip
     unzip Equity-Evaluation-Corpus.zip 
     mv Equity-Evaluation-Corpus experiments1/eip_models/data/. 
     cp  experiments1/eip_models/data/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv experiments1/eip_models/data/Ereg/Equity-Evaluation-Corpus.csv
     ```
  * Download the pretrained embeddings from [here](https://code.google.com/archive/p/word2vec/) and [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip
) and extract them into `experiments1/eip_models/data/embeddings/`.
  
* To train any of the four emotion intensity prediction model (cnn,bert,attn,svm):
  * `cd experiments1/eip_models/ && pip install -r requirements.txt`
  * `cd experiments1/eip_models/<model_name> && CUDA_VISIBLE_DEVICES=0 python main.py`
* To train bert based emotion intensity prediction model:
  * Setup the environment required to train bert based model:
     ```$ git clone https://github.com/anonymous-user-14/transformers 
        $ cd transformers 
        $ pip install . 
        $ pip install -r ./examples/requirements.txt```
  * Run the following command to start training:  
      ```
      $ cd ./examples/text-classification
      $ export DATA_DIR= ../../experiments1/eip_models/data/Ereg/
      $ export TASK_NAME=emotion

      $ CUDA_VISIBLE_DEVICES=0,1 python run_glue.py \
          --model_name_or_path bert-base-cased \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --do_predict \
          --data_dir $DATA_DIR \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 8 \
          --output_dir /tmp/$TASK_NAME/
      ```
# Training Coreference Resolution models from scratch
* If you're interested in training the coreference resolution models from scratch or using pre-trained models, please follow the instructions in the repositories below:
  * [SpanBERT et al.,(2020)](https://github.com/mandarjoshi90/coref)
  * [Lee et al.,(2018)](https://github.com/kentonl/e2e-coref)
  * [Lee et al., (2017)](https://github.com/kentonl/e2e-coref)
  * [Wiseman et al.,(2016)](https://github.com/swiseman/nn_coref)
  * [Clark and Manning (2015)](https://stanfordnlp.github.io/CoreNLP/coref.html#statistical-system)
  * [Lee et al.,(2013)](https://nlp.stanford.edu/software/dcoref.shtml)
* Note that, you need to have a Google Natural Language [account](https://cloud.google.com/natural-language/) and API key in order to process the output of coreference resolution models.

