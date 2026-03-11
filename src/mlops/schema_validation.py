import pandas as pd

def validate_schema(df: pd.DataFrame, metadata: dict):
     """
     Validate dataframe schema against model metadata.
     """

     dataset_meta = metadata["dataset"]

     expected_columns = dataset_meta["columns"]
     expected_schema = dataset_meta["schema"]

     # Check column set
     incoming_columns = list(df.columns)

     if incoming_columns != expected_columns:
          raise RuntimeError(
               f"Feature mismatch.\n"
               f"Expected columns {expected_columns}"
               f"Received columns {incoming_columns}"
          )
     # Check data types
     for col, dtype in expected_schema.items():
          incoming_dtype = str(df[col].dtype)

          if incoming_dtype != dtype:
               raise RuntimeError(
                    f"Type mismatch for column '{col}': "
                    f"expected {dtype} got {incoming_dtype}"
               )