# FPL bot
An attempt to optimise and automate team selection for Fantasy Premier League using Machine Learning and Linear Programming techniques.

### Current project workflow
Data for previous seasons was obtained from: https://github.com/vaastav/Fantasy-Premier-League .

Two models have been trained on this data; a classification model used to predict whether a player will play the next game, and a regression model to predict the number of points a player will score (if they play).
The GradientBoostingClassifier class from the sklearn library is used for the classification model and LGBMRegressor from the lightgbm library is used for the regression model.

After each Gameweek (GW), data is gathered from the FPL API. The models are then used to predict points of each player for the following GW. The best team for the following GW is then determined by linear programming. We maximise the predicted points of the starting 11 subject to a variety of constraints (see team_optimisation.py for details). The problem is solved using the pulp package.

Once the best team has been determined, the team is updated on the FPL website (with minimal user input) via interaction with the FPL API (see web_service.py for details).

### Improvements
- More data and better feature engineering always helps.
- Explore other models (Neural net?) and further optimise current ones.
- Improve interaction with API. Would be good for updates to be carried out without any user input.
- Improve how programme deals with the FPL chips (wildcard, triple captain etc.). This is included at the moment but could be improved.

