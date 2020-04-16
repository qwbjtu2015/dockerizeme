from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
from sklearn.externals import joblib


def make_predictions(sc, df, feature_cols, model_path):
    """
    Make predictions.
    
    Arguments:
        sc: SparkContext.
        df (pyspark.sql.DataFrame): Input data frame containing feature_cols.
        feature_cols (list[str]): List of feature columns in df.
        model_path (str): Path to model on Spark driver

    Returns:
        df (pyspark.sql.DataFrame): Output data frame with probability column.
    """
    # Load classifier and broadcast to executors.
    clf = sc.broadcast(joblib.load(model_path))

    # Define Pandas UDF
    @F.pandas_udf(returnType=DoubleType(), functionType=F.PandasUDFType.SCALAR)
    def predict(*cols):
        # Columns are passed as a tuple of Pandas Series'.
        # Combine into a Pandas DataFrame
        X = pd.concat(cols, axis=1)

        # Make predictions and select probabilities of positive class (1).
        predictions = clf.value.predict_proba(X)[:, 1]

        # Return Pandas Series of predictions.
        return pd.Series(predictions)

    # Make predictions on Spark DataFrame.
    df = df.withColumn("predictions", predict(*feature_cols))
    
    return df