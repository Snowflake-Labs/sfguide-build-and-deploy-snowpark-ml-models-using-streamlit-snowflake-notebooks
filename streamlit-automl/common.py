import logging
import re
from functools import wraps
from pathlib import Path

from snowflake.snowpark import types as T


def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Finished {func.__name__} with result: {result}")
        except Exception as e:
            logging.error(f"Error occurred in {func.__name__}: {e}")
            raise
        else:
            return result

    return wrapper


class Images:
    resource_path = Path(__file__).parent / "resources"
    alert_icon = str(resource_path / "Snowflake_ICON_Alert.png")
    check_icon = str(resource_path / "Snowflake_ICON_Check.png")
    loading_animation = str(resource_path / "loader.gif")


def convert_to_all_caps(c):
    """
    Converts a given string to all capital letters and separates words with underscores.

    Args:
        c (str): The input string to be converted.

    Returns:
        str: The converted string with all capital letters and underscores separating words.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", c).upper()


def rename_columns_all_caps(df):
    """
    Renames all columns in the DataFrame to uppercase.

    Args:
        df (snowpark.DataFrame): The input DataFrame.

    Returns:
        snowpark.DataFrame: The DataFrame with all column names converted to uppercase.
    """
    return df.to_df([convert_to_all_caps(c) for c in df.columns])


def get_col_types(df, type):
    """
    Returns a list of column names in a DataFrame that match the specified data type.

    Args:
        df: The DataFrame to search for column types.
        type (str): The data type to filter columns by. Valid values are "string" and "numeric".

    Returns:
        list: A list of column names that match the specified data type.

    Raises:
        ValueError: If the specified type is not "string" or "numeric".
    """
    if type == "string":
        return [
            c.name
            for c in df.schema
            if isinstance(c.datatype, (T.StringType, T.BooleanType))
        ]
    elif type == "numeric":
        return [
            c.name
            for c in df.schema
            if isinstance(
                c.datatype, (T.DoubleType, T.IntegerType, T.LongType, T.FloatType)
            )
        ]
    else:
        raise ValueError(f"Invalid type: {type}")
