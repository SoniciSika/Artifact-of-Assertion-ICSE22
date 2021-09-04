# Artifact of "Automated Assertion Generation via Information Retrieval and Its Integration with Deep Learning"

## Usage
Depedency:
* pytorch
* javalang


### Step1: Retrieval
```
python ./Retrieval/IR.py $input_config $retrieval_result_path
```
### Step2: Training Neural Models

Train:
Data Preparation
```
python ./NeuralModel/DataPrepration $input_config $retrieval_result_path $neural_data_path_train train
```
Model Training
```
python ./NeuralModel/main $neural_data_path_train $neural_result_path train
```
Evaluate:
Data Preparation
```
python ./NeuralModel/DataPrepration.py $input_config $retrieval_result_path $neural_data_path_evaluate evaluate
```
Model Evaluating
```
python ./NeuralModel/main.py $neural_data_path_evaluate  $neural_result_path evaluate
```
### Step3: Evaluating

Generate Result from Adapt NN and Integration
```
python AdaptionIntegration.py $input_config $retrieval_result_path $neural_result_path $integration_threshold $adaption_integration_nn_result
```
Evaluate Result
