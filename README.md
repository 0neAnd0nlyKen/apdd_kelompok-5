# Steam Games Sentiment Analysis

## DEPLOYED MODELS
[Steam Games Sentiment Analysis on Hugging Face](https://huggingface.co/spaces/SatuSatunyaKen/Steam-Games-Sentiment-Analysis)

## EDA and Modeling (Data Science)
[Google Colab Notebook](https://colab.research.google.com/drive/1oJFHu1hA-cQWcEPENAppZPI_dTrQPXhA#scrollTo=59385b50)

This project aims to study popular Steam games and analyze relevant factors by examining SteamDB data.

![SteamDB Page](public/steamdb%20page.jpg)


### Analysis Done in Google Colab
![Google Colab Screenshot](public/g%20colab%20screenshot.png)

Data cleaning and feature evaluation were performed, resulting in the following features:

```python
Index(['followers', 'Gain', 'user-reviews',
       'twitch-viewers-24-hours_twitch-viewers-all-time_mean',
       'Peak_Average_mean', 'Gain-pct_Avg-Gain-pct_mean',
       'owner-estimations-vg_owner-estimations-SteamSpy_owner-estimations-PlayTracker_mean'],
      dtype='object')
```


### Cleaned Features Correlation Heatmap
![Cleaned Features Correlation Heatmap](public/cleaned%20features%20corelation%20heatmap.png)

### Modeling Results
| Model               | MAE        | RMSE       |
|---------------------|------------|------------|
| Linear Regression    | 10.209282  | 12.767920  |
| Random Forest        | 4.043232   | 4.994123   |
| XGBoost             | 0.000640   | 0.000897   |

Models are deployed with a Streamlit app on the Hugging Face platform.
![Streamlit App](public/streamlit%20app.png)