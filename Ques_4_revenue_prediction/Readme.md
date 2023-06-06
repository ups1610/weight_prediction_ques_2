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

Imagine you working as a sale manager now you need to predict the Revenue

and whether that particular revenue is on the weekend or not and find the

Informational_Duration using the Ensemble learning algorithm


### Description

The above dataset get cleaned and preprocessed using label encoding technique.

Since from the observation and by doing EDA analysis, its a classification problem so we apply 

ensemble learning algorithm such Random Forest , Ada Boost etc.. on the dataset and the best fitted model is Random Forest

with an accuracy of 90.1%

The project analysis can be present in a file :

```
EDA.ipynb
```

### Project run command

```
cd Ques_4_revenue_prediction
python src/pipeline/training_pipeline.py
```

### Prediction of the model command

```
python src/pipeline/prediction_pipeline.py
```
