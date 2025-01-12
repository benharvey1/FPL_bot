# FPL bot
An attempt to optimise and automate team selection for Fantasy Premier League using Machine Learning and Linear Programming techniques.

### Dataset
Data for previous seasons was obtained from: https://github.com/vaastav/Fantasy-Premier-League .
The FPL API is used to get data for the current season

### Machine Learning
Two machine learning models are trained; a classification model used to predict whether a player will play the next game, and a regression model to predict the number of points a player will score (if they play).
The GradientBoostingClassifier class from the sklearn library is used for the classification model and LGBMRegressor from the lightgbm library is used for the regression model.

### Optimisation
The best team is determined by linear programming. We maximise the predicted points of the starting 11 subject to a variety of constraints (see team_optimisation.py for details). The problem is solved using the pulp package.

### Updating team
Once the best team has been determined, the team is updated on the FPL website with minimal user input. See web_service.py for details.