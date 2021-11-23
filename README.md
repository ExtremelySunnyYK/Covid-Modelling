Background

In the covid pandemic, Singapore's economy was hit quite badly. However what if we can find out the indicators that best impacts economy in this crisis.

Our design project aims to protect Singapore's economy emits covid-19 based on various factors the reason why this is important is firstly we know what is favorable that affects or has a strong correlation with the economy and therefore when coming up with strategies we know what to prioritize.

Secondly, because by comparing the predicted growth rate against the actual growth rate, we know can determine how effective singapore is at coping against covid 19 (doesn't really make sense).

Because Singapore's GDP is only released once a year if we are not able to accurately determine the effect of covid on a month-to-month basis on the impact of singapore's economy. Because of that we have chosen to use the straits times index as the variable we want to predict.

Therefore our problem statement is to predict the growth rate of the straits times index given the covid-19 data and the other factors that affect singapore's economy.

Assumptions that we are making:

- STI is a good predictor of the economy as STI is relatively stable and singapore stock market is strongly regulated which means a more rational and efficent market.
- COVID 19 has a strong correlation with the economy.

Datasets
We have used the following features in our overall datasets:

- Time : time is a key factor in the progression of the pandemic and the growth rate of the economy.
- Vaccination rate : Vacinnation rate reduces the overall health impact of the pandemic on the population and allows the economy to open up.
- Daily new cases : Daily new cases is the number of new cases that are diagnosed in a given day, which allows us to see the change in COVID 19.
- Hospitalized : High rate of hospitalization puts a huge strain on the healthcare industry, and more economic resources has to be spent to maintain the health care infrastructure.
- Recovered : Recovered is the number of people that have recovered from the disease. It tells us what is the rate of recovery from the disease.
- Phases : The strategy that Singapore Government has adopted to combat the pandemic is to divide the pandemic into phases. It is significant as government's strategy has a high impact on the openess and the attractiveness of the economy to foreign investors. reference : https://www.straitstimes.com/business/companies-markets/singapore-stocks-start-seeing-hit-from-pre-emptive-phase-two-measures

Our data is collected from the following sources:

- Straits times index data : https://www.wsj.com/market-data/quotes/index/SG/STI/historical-prices
- Singapore COVID-19 data : https://data.world/hxchua/covid-19-singapore

Data preprocessing
Since we are taking vaccination rate into account and vaccination rate has a strong correlation (from our Visualisation) , therefore our start date is pushed to when Singapore has rolled out vaccination.

Model

- For our initial model, we are using a linear regression model with gradient descent.
- We are using the straits times index as our target variable.
- We are using the following features:
  - Time
  - Vaccination rate
  - Daily new cases
  - Hospitalized
  - Recovered
  - Phases

The initial result of our model is that the model is not very accurate. It has a R2 score of 0.2. We are going to try to improve the model by using the following features:

- Normalising the data
- Reducing the number of features
- Transforming the data with Polynomial Features
- Adding non covid related features

After our improvements, our model has a R2 score of 0.65.
Notable Improvements:

- normalising the data. This is because the features are not on the same scale and cause a large variance in the data, which negatively affects the model.
- reducing the number of features. Some features might not be correlated with Straits times index and therefore we are reducing the number of features after performing feature selection through data visualization.
- transforming the data with polynomial features. This is because the straits times index is not linear in relation with the features and therefore we are transforming the data with polynomial features.

Web App:
For our web app, the problem that we are trying to solve is how can we help the user to predict the growth rate of the straits times index given the covid-19 data and the other factors that affect singapore's economy and visualise the factors that affect singapore's economy.

The user can choose the features that they want to predict the growth rate of the straits times index given the covid-19 data and the other factors that affect singapore's economy.

The web app will have the following features:

- data visualization of the features and straits times index
- prediction of the straits times index price given their input

We believe that the web app will be a great tool to help the user understand the effect of covid on singapore's economy.

Architecture of the web app
Front end:

- Stream lit for data visualization
- Flask Jinja2 for templating for the user input app

Back end:

- Flask for the endpoints
