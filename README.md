# User Intent Prediction in Information-seeking Conversations
This is the implementation "User Intent Prediction in Information-seeking Conversations". This paper has been accepted to CHIIR 2019. If you find our repo useful in your paper, please cite our work.

Our models predict user intent for utterances in information-seeking conversations. In other words, the models conduct utterance type classification. One utterance can have more than one user intent label. These models are built with Python 2.7 and Keras.

## Data
We used the "MSDialog-Intent" data in the [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) dataset. 

## Code
* ./fetch_utterances_from_db.ipynb and ./gen_ground_truth.ipynb: conduct preprocessing as described in the paper.
* ./features/: extract features
* ./neural_models/: different stand-alone neural models 

## Notes
At the time of developing, we preprocessed the data and extracted features during the process of pulling the data from our database. These codes can be adapted to work with data in json format.
