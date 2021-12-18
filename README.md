![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

<img src= "https://user-images.githubusercontent.com/86251750/146631168-9e92029f-0aeb-47fb-afd3-2eeeb1cf8530.png" width="800" height="400">

## Public Relation Department

* Public Relations Department supervises and assesses public attitudes, and maintaining mutual relations and understanding between an organization and its public.

* It improves channels of communication and to institute new ways of setting up a two-way flow of information and understanding.

* Natural language processing can be used to build predictive models to perform sentiment analysis on social media posts and reviews and predict if customers are happy or not. 

* Natural language processors work by converting words into numbers and training a machine learning models to make predictions. 

* That way, you we automatically know if customers are happy or not without manually going through massive number of tweets or reviews!

## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 
## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Run Locally](#run)
* [Exploring the Data](#viz)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [solving the task](#fe)
* [prediction](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

Based on the reviews ( in text format), we would like to predict whether their customers are satisfied with the product or not. 

## Dataset Used:<a name="data"></a>

The Public relations department team has collected extensive data on their customers such as product reviews. 

`rating : From 0 to 5`

`date`

`variation`

`verified_reviews`

`feedback`

dataset can be found at [link](https://github.com/pradeepsuyal/public_relation_department/tree/main/dataset)

## Run locally:<a name="run"></a>

Clone the project

```bash
https://github.com/pradeepsuyal/public_relation_department.git
```

Go to the project directory

```bash
  cd public_relation_department
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```

If you output `pip freeze` to a file with redirect >, you can use that file to install packages of the same version as the original environment in another environment.

First, output requirements.txt to a file.

```bash
  $ pip freeze > requirements.txt
```

Copy or move this `requirements.txt` to another environment and install with it.

```bash
  $ pip install -r requirements.txt
```

## Exploring the Data:<a name="viz"></a>

I have used pandas, matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

![matplotlib](https://eli.thegreenplace.net/images/2016/animline.gif)

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
![seaborn](https://i.stack.imgur.com/uzyHd.gif)

for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/146631469-ec4167f6-2285-49dd-a47d-460df0c71e1b.png)

![download](https://user-images.githubusercontent.com/86251750/146631441-4920998c-3265-4930-ba7f-7b03bb3aed0f.png)

![download](https://user-images.githubusercontent.com/86251750/146631460-f47e4f93-0248-4b17-97fd-880eefbb877d.png)

## approach for making prediction<a name="fe"></a>
-------

* performed data cleaning and applied pandas dummy variable encoding for categorical data 
* created a pipeline to perform two task
       
      (1) remove punctuation, (2) remove stopwords using Natural Language tool kit(nltk)
      
* Define the cleaning pipeline we defined earlier to perform countvectorizer
* Train a NaiveByes classifier model
* Evalute Trained model performance
* Finally I trained and evalute a Logistic Regression Classifier.


## Prediction:<a name="models"></a>
------

**TOKENIZATION(COUNT VECTORIZER)**

![image](https://user-images.githubusercontent.com/86251750/146632134-05185bd0-f7f3-4007-b126-244897bfa5ea.png)

**NaiveByes Intuition**
       
* Naïve Bayes is a classification technique based on Bayes’ Theorem.
* Let’s assume that you are data scientist working major bank in NYC and you want to classify a new client as eligible to retire or not.
* Customer features are his/her age and salary.

![image](https://user-images.githubusercontent.com/86251750/146632205-89d43a94-2c18-40c7-bd11-65131c61781c.png)

    NAÏVE BAYES: 1. PRIOR PROBABILITY

    * Points can be classified as RED or BLUE and our task is to classify a new point to RED or BLUE.
    * Prior Probability: Since we have more BLUE compared to RED, we can assume that our new point is twice as likely to be BLUE than RED. 

![image](https://user-images.githubusercontent.com/86251750/146632266-59600fdc-0ca3-400d-964b-2afba009c3cb.png)

![image](https://user-images.githubusercontent.com/86251750/146632476-78f668d5-408a-4910-952d-8876caf14c40.png)
    
    NAÏVE BAYES: 2. LIKELIHOOD 
    
    * For the new point, if there are more BLUE points in its vicinity, it is more likely that the new point will be classified as BLUE. 
    * So we draw a circle around the point
    * Then we calculate the number of points in the circle belonging to each class label.

![image](https://user-images.githubusercontent.com/86251750/146632538-5966ea1e-54af-4232-930a-224a5adf7672.png)

![image](https://user-images.githubusercontent.com/86251750/146632597-cd7a3b55-c8c1-444f-b60d-1964e7ff6caa.png)

    NAÏVE BAYES: 3. POSTERIOR PROBABILITY

    * Let’s combine prior probability and likelihood to create a posterior probability. 
    * Prior probabilities: suggests that X may be classified as BLUE Because there are twice as much blue points.
    * Likelihood: suggests that X is RED because there are more RED points in the vicinity of X.
    * Bayes’ Rule combines both to form a posterior probability.
    
![image](https://user-images.githubusercontent.com/86251750/146632731-d161f9b2-11f4-4e09-b3e2-6eaf8f3813a6.png)

![image](https://user-images.githubusercontent.com/86251750/146632810-d82847a4-9071-49dd-a153-84dc4100309d.png)

    NAIVE BYES: SOME MATH
    
![image](https://user-images.githubusercontent.com/86251750/146632858-ea707b4d-e7b9-4c26-8bce-152740e8e862.png)

    * Naïve Bayes is a classification technique based on Bayes’ Theorem.
    * 𝑋: New Customer’s features; age and savings
    * 𝑃(𝑅𝑒𝑡𝑖𝑟𝑒┤|𝑋): probability of customer retiring given his/her features, such as age and savings
    * 𝑃(𝑅𝑒𝑡𝑖𝑟𝑒): Prior probability of retiring, without any prior knowledge 
    * 𝑃(𝑋|𝑅𝑒𝑡𝑖𝑟𝑒): likelihood
    * 𝑃(𝑋): Marginal likelihood, the probability of any point added lies into the circle

    𝑃(𝑅𝑒𝑡𝑖𝑟𝑒)=(# of Retiring)/(Total points)= 40/60
    
    𝑃(𝑋│𝑅𝑒𝑡𝑖𝑟𝑒)=(# of smilar observations for retiring)/(𝑇𝑜𝑡𝑎𝑙 # 𝑟𝑒𝑡𝑖𝑟𝑖𝑛𝑔)=1/40

    𝑃(𝑋)=(# of Similar observations)/(𝑇𝑜𝑡𝑎𝑙 # 𝑃𝑜𝑖𝑛𝑡𝑠)=4/60

    𝑃(𝑅𝑒𝑡𝑖𝑟𝑒|𝑋) = (40/60∗1/40)/(4/60) = (1/60)/(4/60)=0.25

*ASSESS TRAINED MODEL PERFORMANCE(NaiveByes)*

    Tarin
 
![download](https://user-images.githubusercontent.com/86251750/146633150-ecff291f-2437-41e8-bc99-b462c2ce5d37.png)

    Test

![download](https://user-images.githubusercontent.com/86251750/146633156-313cf3e8-85d3-41d7-adff-708bdae55bb5.png)

                 precision    recall  f1-score   support

           0       0.50      0.39      0.44        46
           1       0.95      0.97      0.96       584

    accuracy                           0.93       630

**Logistic Regression**

* Linear regression is used to predict outputs on a continuous spectrum. 

      Example: predicting revenue based on the outside air temperature. 

* Logistic regression is used to predict binary outputs with two possible values labeled "0" or "1"

      Logistic model output can be one of two classes: pass/fail, win/lose, healthy/sick
      
![image](https://user-images.githubusercontent.com/86251750/146035729-a43e75d7-765f-42d9-83d8-73f1e7f55516.png)

* Logistic regression algorithm works by implementing a linear equation first with independent predictors to predict a value. 

* We then need to convert this value into a probability that could range from 0 to 1.

      Linear equation:
      𝑦=𝑏_0+𝑏_1∗𝑥

      Apply Sigmoid function:
      𝑃(𝑥)= 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 (𝑦)
      𝑃(𝑥)=1/1+𝑒^(−𝑦)
      𝑃(𝑥)=1/1+𝑒^−(𝑏_0+𝑏_1∗𝑥)
      
![image](https://user-images.githubusercontent.com/86251750/146035395-b52a4449-cdbc-4230-b46e-07933f13b6b6.png)

* Now we need to convert from a probability to a class value which is “0” or “1”.

![image](https://user-images.githubusercontent.com/86251750/146036372-d6ed12d6-a414-450a-9ce2-bfdd1c46c2b4.png)

*evaluting performance*

![download](https://user-images.githubusercontent.com/86251750/146633226-4a1a936d-6e85-4a28-9ab8-b96e17faeff5.png)

              precision    recall  f1-score   support

           0       0.88      0.39      0.54        54
           1       0.95      0.99      0.97       576

    accuracy                           0.94       630

## CONCLUSION:<a name="conclusion"></a>
-----
With Naive Byes I was able to get good accuracy score of 93 and than I applied LogisticRegression which even give me better accruracy pf 94 and also precision and f1 score is better.

    NOTE--> we can further improve the performance by using other classification model such as CART model(XGBOOST, LIGHTGBM, CATBOOST, DecissionTree etc) and many more. Further performance can be improved by using various hyperparameter optimization technique such as optuna,hyperpot, Grid Search, Randomized Search, etc.   
