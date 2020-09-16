# Collabrative-Filtering-Recommendation
Collabrative Filtering Recommendation including Item-based CF and User-based CF, implementing with Sklearn and Surprise.
## Sklearn Implementation
Similarity function: pairwaise consine distances

Item-based CF Prediction:
![](https://github.com/ZorrowHu/Collabrative-Filtering-Recommendation/blob/master/CF_sklearn/img2.png)

User-based CF Prediction:
![](https://github.com/ZorrowHu/Collabrative-Filtering-Recommendation/blob/master/CF_sklearn/img1.png)

Evaluation using Root Mean Square Error:
![](https://github.com/ZorrowHu/Collabrative-Filtering-Recommendation/blob/master/CF_sklearn/img3.png)

## Surprise Implementation
Similarity function: Pearson Baseline

Item-based CF Prediction: Recommending 10 items via 1 item based on Pearon Similarity. That 10 items are neighbors of the given item using KNN method 

User-based CF Prediction: Recommending 10 items via 1 user based on Pearson Similarity. That 10 items are the top rating items of the given user's neighbors, whose count is also 10. In the program I simply make a item list of given users' neighbors and select top 10 rating items.  
