import pandas as pd
from argparse import ArgumentParser
from os.path import isfile
from sportsreference.ncaab.teams import Teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATASET_NAME = 'dataset.pkl'
FIELDS_TO_DROP = ['away_points', 'home_points', 'date', 'location',
                  'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                  'winning_name', 'home_ranking', 'away_ranking', 'pace']

def load_saved_dataset():
    if isfile(DATASET_NAME):
        return pd.read_pickle(DATASET_NAME)
    return pd.DataFrame()

def create_dataset(teams):
    dataset = load_saved_dataset()
    if not dataset.empty:
        return dataset

    for team in teams:
        dataset = pd.concat([dataset, team.schedule.dataframe_extended])
    dataset.to_pickle(DATASET_NAME)
    return dataset.drop_duplicates()

def process_data(dataset):
    X = dataset.drop(FIELDS_TO_DROP, 1).dropna()
    y = dataset[['home_points', 'away_points']].values
    return train_test_split(X, y)

def build_model(X_train, y_train):
    parameters = {'bootstrap': False,
                  'min_samples_leaf': 3,
                  'n_estimators': 50,
                  'min_samples_split': 10,
                  'max_features': 'sqrt',
                  'max_depth': 6}
    model = RandomForestRegressor(**parameters)
    model.fit(X_train, y_train)
    return model

def add_features(stats):
    if 'defensive_rating' not in stats and \
       'offensive_rating' in stats and \
       'net_rating' in stats:
        stats['defensive_rating'] = stats['offensive_rating'] - \
            stats['net_rating']
    defensive_rebound_percentage = 100.0 * stats['defensive_rebounds'] /\
        (stats['defensive_rebounds'] + stats['offensive_rebounds'])
    stats['defensive_rebound_percentage'] = defensive_rebound_percentage
    return stats

def replace_feature_names(team, away=False):
    team = team.drop(team.filter(regex='opp_').columns, axis=1)
    team = add_features(team)
    if away:
        columns = ['away_%s' % col for col in team]
    else:
        columns = ['home_%s' % col for col in team]
    team.columns = columns
    return team.reset_index()

def create_matchup_data(home, away):
    home_stats = replace_feature_names(home)
    away_stats = replace_feature_names(away, away=True)
    return pd.concat([away_stats, home_stats], axis=1)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('home', help='Specify the name of the home team as '
    'noted on sports-reference.com, such as "purdue".')
    parser.add_argument('away', help='Specify the name of the away team as '
    'noted on sports-reference.com, such as "indiana".')
    return parser.parse_args()

args = parse_arguments()
teams = Teams()
dataset = create_dataset(teams)
X_train, X_test, y_train, y_test = process_data(dataset)
model = build_model(X_train, y_train)
match_stats = create_matchup_data(teams(args.home).dataframe,
                                  teams(args.away).dataframe)
df = match_stats.loc[:, X_train.columns]
result = model.predict(df).astype(int)
print('%s %s - %s %s' % (args.home, result[0][0], result[0][1], args.away))