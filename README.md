# IPL Win Predictor

## Project Overview

The IPL Win Predictor is a machine learning application that predicts the probability of a team winning an Indian Premier League (IPL) cricket match based on real-time match statistics.

## Features

- Predicts win probability for batting and bowling teams
- Supports all current IPL teams
- Considers multiple match parameters:
  - Current score
  - Overs completed
  - Wickets fallen
  - Target score
  - Hosting city

## Technology Stack

- Python
- Streamlit (Web Application)
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- RandomForestClassifier (Prediction Model)

## Prerequisites

- Python 3.7+
- Required libraries:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - pickle

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/ipl-win-predictor.git
cd ipl-win-predictor
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

3. Download datasets
- `matches.csv`
- `deliveries.csv`

## Model Training Process

1. Data Preprocessing
   - Load IPL match and delivery datasets
   - Clean and map team names
   - Filter matches for current teams
   - Handle missing values and outliers

2. Feature Engineering
   - Calculate current score
   - Track balls bowled and remaining
   - Compute run rates (current and required)
   - Create percentage-based features

3. Model Training
   - Use RandomForestClassifier
   - Perform one-hot encoding for categorical features
   - Scale numerical features
   - Split data into training and testing sets
   - Achieve model accuracy around 70-80%

## Running the Application

```bash
streamlit run app.py
```

## Model Inputs

- Batting Team
- Bowling Team
- Host City
- Target Score
- Current Score
- Completed Overs
- Wickets Fallen

## Model Output

Probability of winning for:
- Batting Team
- Bowling Team

## Model Performance

- Accuracy: Varies based on training data (typically 70-80%)
- Features considered: 15+ match parameters
- Classifier: Random Forest

## Visualizations

The project includes three key visualizations:
1. Distribution of Runs Left Percentage
2. Relationship between Balls Left and Runs Left
3. Feature Importance Ranking

## Limitations

- Predictions based on historical IPL data
- Accuracy depends on match complexity
- Does not account for player-specific performance
- Real-time match dynamics can vary

## Future Improvements

- Integrate player-specific statistics
- Add more sophisticated machine learning models
- Improve feature engineering
- Create more detailed visualizations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

[Specify your license, e.g., MIT License]

## Contact

[Your Name/Contact Information]
#
