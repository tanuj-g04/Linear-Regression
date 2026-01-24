#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np


# In[24]:


class Normalizer:
    def __init__(self):
        self.means={}
        self.std={}
        self.fitted=False
        
    def fit(self, df):
        for col in df.columns:
            values=df[col]
            mean=sum(values)/len(values)
            variance=sum((x-mean)**2 for x in values)/len(values)
            std=variance**0.5
            
            self.means[col]=mean
            self.std[col]=std
        
        self.fitted=True
        
    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        
        new_data={}
        for col in df.columns:
            values=df[col]
            mean=self.means[col]
            std=self.std[col]
            
            new_data[col]=[(x-mean)/std for x in values]
        return pd.DataFrame(new_data)


# In[41]:


def traintestsplit(df, seed=None):
    if seed is not None:
        df=df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n=len(df)
    i=int(0.8*n)
    train=df.iloc[0:i]
    test=df.iloc[i:n]
    return train, test


# In[26]:


def featuretargetsplit(df, target_column):
    features=df.drop(columns=[target_column])
    target=df[target_column]
    return features, target


# In[57]:


class LinearRegressionModel:
    def __init__(self):
        self.weights={}
        self.bias=0
        self.initialized=False
        
    def initialize(self,df):
        for col in df.columns:
            self.weights[col]=0
        self.bias=0
        self.initialized=True
        
    def predict_row(self, data):
        if not self.initialized:
            raise RuntimeError("Model not initialized")
            
        prediction = self.bias
        for feature, value in data.items():
            prediction+=self.weights[feature]*value
        return prediction
    
    def predict(self, df):
        return [self.predict_row(df.iloc[i]) for i in range(len(df))]


# In[72]:


def error(prediction, target):
    total=0.0
    for y_hat, y in zip(prediction, target):
        diff=y_hat-y
        total+=diff**2
    return total/(2*len(target))


# In[62]:


def compute_gradient(model, features_df, targets):
    weight_grads={feature: 0.0 for feature in model.weights}
    bias_grad=0.0
    n=len(features_df)   
    for i in range(n):
        row=features_df.iloc[i]
        y_hat=model.predict_row(row)
        error=y_hat - targets[i]
        
        for feature, value in row.items():
            weight_grads[feature]+=error*value
        bias_grad+=error
    
    for feature in weight_grads:
        weight_grads[feature]/=n
    
    bias_grad/=n
    return weight_grads, bias_grad


# In[63]:


class GradientDescentTrainer:
    def __init__(self, learning_rate):
        self.learning_rate=learning_rate
    
    def step(self, model, features_df, targets):
        weights_grads, bias_grad= compute_gradient(
            model, features_df, targets
        )
        
        for feature in model.weights:
            model.weights[feature]-= self.learning_rate* weights_grads[feature]
            
        model.bias -= self.learning_rate * bias_grad


# In[64]:


def train(model, df, targets, epochs, learning_rate):
    trainer = GradientDescentTrainer(learning_rate)
    
    for epoch in range(epochs):
        trainer.step(model, df, targets)
        predictions=model.predict(df)
        loss=error(predictions, targets)
        
        print(f"Epoch {epoch+1}: Loss: {loss}")
    return model.weights, model.bias


# In[65]:


def predict_test(model, features,targets):
    predictions=model.predict(features)
    mse=error(targets,predictions)
    return mse


# In[86]:


def lin_reg_pipeline(df, target_column, learning_rate=0.01, epochs=100):
    
    train_df,test_df=traintestsplit(df)
    
    x_train,y_train=featuretargetsplit(train_df,target_column)
    
    x_test, y_test=featuretargetsplit(test_df,target_column)
    
    normalizer=Normalizer()
    normalizer.fit(x_train)
    
    x_train_norm=normalizer.transform(x_train)
    
    x_test_norm=normalizer.transform(x_test)
    
    model=LinearRegressionModel()
    model.initialize(x_train_norm)
    
    train(model, x_train_norm, y_train, epochs=epochs, learning_rate=learning_rate)
    
    test_pred=model.predict(x_test_norm)
    test_mse=error(test_pred, y_test)
    
    results_df=x_test.copy()
    results_df['y_true']=list(y_test)
    results_df['y_pred']=test_pred
    
    acc=0.0
    true_y=list(y_test)
    for i in range (len(y_test)):
        acc+=(abs(true_y[i]-test_pred[i]))/true_y[i]
    acc=acc/len(y_test)
    
    return{
        "model":model,
        "test_mse":test_mse,
        "results_df":results_df,
        "accuracy":100-acc
    }


# In[ ]:




