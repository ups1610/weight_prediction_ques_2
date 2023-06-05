## End to End ML Project

### created a environment
```
conda create -p venv python==3.8
```

### Install all necessary librries
```
pip insatll -r requirements.txt
```
### Install the entire package
```
python setup.py install
```

### About the project

The above dataset get cleaned and preprocessed using label encoding technique.

Since from the observation and by doing EDA analysis, its a classification problem so we apply 

Logistic Regression , Decision Tree , Random Forest , SVM on the dataset and the best fitted model is Random Forest

with an accuracy of 96.17%

The project analysis can be present in a file :

```
EDA.ipynb
```

### Project run command

```
cd weight_prediction
python src/pipeline/training_pipeline.py
```

### Prediction of the model command

```
python src/pipeline/prediction_pipeline.py
```
