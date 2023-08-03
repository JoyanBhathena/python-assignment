import unittest
import pandas as pd
from sqlalchemy import create_engine, inspect
from assignment_solution import DataFrameToSQL, LoadData

class TestDataFrameToSQL(unittest.TestCase):
    def setUp(self):
        """
        Set up a dummy DataFrame for testing.

        You can use a randomly generated DataFrame for testing, or load the CSV file here.
        For simplicity, we are creating a dummy DataFrame with sample data.
        """
        data = {'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']}
        self.test_dataframe = pd.DataFrame(data)

    def test_load_dataframe_to_sql(self):
        """
        Test the 'load_dataframe_to_sql' method of the DataFrameToSQL class.

        This test case checks if the DataFrame is correctly loaded into the database as a table.
        """
        # Define the test database connection string (using SQLite in-memory database)
        test_db_connection = 'sqlite:///my_database.db'

        # Instantiate the DataFrameToSQL object with imaginary CSV file paths
        df_to_sql = DataFrameToSQL('train_MSCS.csv', 'test_MSCS.csv', 'ideal_MSCS.csv')

        # Call the method to load the DataFrame into the test database
        table_name = 'test_table'
        df_to_sql.load_dataframe_to_sql(self.test_dataframe, table_name, test_db_connection)

        # Verify that the table exists in the database
        engine = create_engine(test_db_connection)
        inspector = inspect(engine)
        table_exists = inspector.has_table(table_name)
        engine.dispose()

        # Assert the table exists in the database
        self.assertTrue(table_exists, f"The table '{table_name}' should exist in the database.")

if __name__ == '__main__':
    unittest.main()
