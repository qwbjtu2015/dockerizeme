import dshelper as dsh
import numpy as np
import os
import pandas as pd
import sys
import time

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker

DeclarativeBase = declarative_base()

class Model(DeclarativeBase):
    __tablename__ = 'models'

    id = Column(Integer, primary_key = True)
    name = Column(String, nullable = False)
    __table_args__ = (UniqueConstraint('name', name='unco1'),)

class Forecast(DeclarativeBase):
    __tablename__ = 'forecasts'

    id = Column(Integer, primary_key = True)
    model = Column(Integer, ForeignKey('models.id'), nullable = False)
    ts = Column(DateTime)
    fore = Column(Float)
    details = Column(String)
    __table_args__ = (UniqueConstraint('model', 'ts', name='unco1'),)

class MlLoop:
    def __init__(self, model_name, log_file, index_format='%Y-%m-%d', db_url=None):
        self.model_name = model_name # The model name to use for the database
        self.log_file = log_file
        self.index_format = index_format

        self.db_url = db_url
        self.db_session = None

        if self.db_url is not None:
            self.init_db()

    def init_db(self):
        engine = create_engine(self.db_url)
        DeclarativeBase.metadata.create_all(engine)
        Session = sessionmaker(bind = engine)
        self.db_session = Session()
        try:
            self.db_session.add(Model(name = self.model_name))
            self.db_session.commit()
        except:
            self.db_session.rollback()
            pass
        self.model_id = self.db_session.query(Model.id).filter(Model.name == self.model_name).first()[0] 

    def run(self, features, response, forecast_locations, max_history=1e6):
        assert len(features) == len(response)

        db_session = None
        if self.db_url is not None:
            self.init_db()

        timer = None
        if sys.platform == 'win32':
            timer = time.clock
        else:
            timer = time.time

        for ii in range(0, forecast_locations.len()):
            # Prepare the range for training for this iteration
            history_end = forecast_locations.starts[ii]
            history_start = 0
            if (history_end - history_start + 1) > max_history:
                history_start = history_end - max_history + 1 
            xx = features.iloc[history_start:history_end].as_matrix()
            yy = response.iloc[history_start:history_end].as_matrix()

            # Train the model
            start = timer()
            clf = QDA()
            clf.fit(xx, yy)
            training_time = timer() - start

            # Forecast
            fore_xx = features.iloc[forecast_locations.starts[ii]:(forecast_locations.ends[ii]+1)].as_matrix()
            start = timer()
            fore = clf.predict(fore_xx)
            fore_df = clf.predict_proba(fore_xx)
            fore_df = pd.DataFrame(fore_df, index=features.iloc[forecast_locations.starts[ii]:(forecast_locations.ends[ii]+1)].index)
            # Generate proper column names. Map -1,0,1 to 'short','out','long'
            fore_df.columns = np.array(['short','out','long'])[clf.classes_.astype(int) + 1]
            forecasting_time = timer() - start
            # fore = [-1, -1, 0]
            # metric = [0.16, 0.56, 0.34]
            # fore = [-1.11]
            metric = [0.0]

            # Save results to a database or somewhere else
            if self.db_session is not None:
                for jj in range(len(fore)):
                    ts = features.index[forecast_locations.starts[ii] + jj]
                    details = fore_df.to_json(orient='split', date_format='iso')
                    rs = self.db_session.query(Forecast.id).filter(Forecast.ts == ts).filter(Forecast.model == self.model_id).first()
                    if rs is None:
                        ff = Forecast(model = self.model_id, ts = ts, fore = fore[jj], details = details)
                        self.db_session.add(ff)
                    else:
                        ff = Forecast(id = rs[0], model = self.model_id, ts = ts, fore = fore[jj], details = details)
                        self.db_session.merge(ff)

            # Log output
            if self.log_file is not None:
                out_str = "\n" + features.index[forecast_locations.starts[ii]].strftime(self.index_format) + " - " + \
                    features.index[forecast_locations.ends[ii]].strftime(self.index_format) + "\n" + \
                    "=======================\n" + \
                    "    history: from: " + features.index[history_start].strftime(self.index_format) + ", to: " + \
                    features.index[history_end - 1].strftime(self.index_format) + \
                    ", length: " + str(history_end - history_start) + "\n" + \
                    "    forecast length: " + str(forecast_locations.ends[ii] - forecast_locations.starts[ii] + 1) + "\n" + \
                    "    forecast: [" + ','.join(str(ff) for ff in fore) + "]\n" + \
                    "    probs: [" + ','.join(str(mm) for mm in metric) + "]\n" + \
                    "    time [training]: " + str(training_time) + "\n" + \
                    "    time [forecasting]: " + str(forecasting_time) + "\n"
                with open(self.log_file, "a") as ff:
                    print(out_str, file=ff)

        if self.db_session is not None:
            self.db_session.commit()

def drive_mlloop():
    all_data = dsh.load('all_data.bin')

    data = all_data['HO2']

    # Sanity checks
    combined = pd.concat([data['features'], data['full']['entry']], axis=1)
    combined = combined.dropna()
    if len(data['features']) != len(combined):
        raise RuntimeError('Some observations were removed while merging the series. Ensure there are no NAs and the series length match.')

    response = combined.iloc[:,-1]
    features = combined.iloc[:,:-1]

    fl = dsh.ForecastLocations(features.index)

    ml = MlLoop('test_model', 'ml.log', db_url = 'sqlite:///ml.sqlite')
    ml.run(features, response, fl)
    
def main():
    drive_mlloop()

if __name__ == "__main__":
    main()