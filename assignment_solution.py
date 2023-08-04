import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine



print("Libraries loaded successfully!")

# Custom Exception classes
class DataLoadException(Exception):
    """Custom exception class for catching data load exceptions."""
    pass


class DatabaseConnectionException(Exception):
    """Custom exception class for database connection exceptions."""
    pass


class DataFrameToSQLException(Exception):
    """Custom exception class for DataFrame to SQL exception."""
    pass


class VisualizationException(Exception):
    """Custom exception class for visualization exceptions."""
    pass


class IdealFunctionSelectorException(Exception):
    """Custom exception class for ideal function selector exceptions."""
    pass


class IdealFunctionMapperException(Exception):
    """Custom exception class for ideal function mapper exceptions."""
    pass


class LoadData:
    """
    This class will load the data into the database and will also return the files as DataFrames which can easily be used
    for data manipulation.
    This class also has a function to load a DataFrame to the DB as a table
    """

    def __init__(self, train_path, test_path, ideal_functions_path):
        """
        Initialize the LoadData object with file paths for train, test, and ideal functions.

        :param train_path: The file path for the training data CSV.
        :param test_path: The file path for the test data CSV.
        :param ideal_functions_path: The file path for the ideal functions CSV.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.ideal_functions_path = ideal_functions_path

    def load_csv_to_database(self):
        """
        Load CSV files into a database using SQLAlchemy.

        :return: A tuple containing DataFrames for train, test, and ideal functions if loading is successful.
        """
        database_connection = 'sqlite:///my_database.db'
        try:
            # Create a database connection using SQLAlchemy
            engine = create_engine(database_connection)
            files_list = [self.train_path, self.test_path, self.ideal_functions_path]
            error_flag = False  # Initialize error flag as False

            # Load CSV files into DataFrames and store them in the database as tables
            for file in files_list:
                if 'train' in file:
                    try:
                        local_train = pd.read_csv(file)
                        local_train.to_sql('train_data', engine, if_exists='replace', index=False)
                    except Exception as e_train:
                        raise DataLoadException(f"Error in loading Train Set from {file}:\n{str(e_train)}")
                        error_flag = True

                elif 'ideal' in file:
                    try:
                        local_ideal_functions = pd.read_csv(file)
                        local_ideal_functions.to_sql('ideal_functions_data', engine, if_exists='replace', index=False)
                    except Exception as e_ideal:
                        raise DataLoadException(f"Error in loading Ideal Set from {file}:\n{str(e_ideal)}")
                        error_flag = True

                elif 'test' in file:
                    try:
                        local_test = pd.read_csv(file)
                        local_test.to_sql('test_data', engine, if_exists='replace', index=False)
                    except Exception as e_test:
                        raise DataLoadException(f"Error in loading Test Set from {file}:\n{str(e_test)}")
                        error_flag = True

            # Close the database connection
            engine.dispose()

            # Print "All Files loaded successfully!" if no errors occurred during loading
            if not error_flag:
                print("All Files loaded into the DB successfully!")
                return local_train, local_ideal_functions, local_test
        except Exception as e_data_load:
            raise DatabaseConnectionException(f"Error in creating the database connection:\n{str(e_data_load)}")


# Child Class
class DataFrameToSQL(LoadData):
    def __init__(self, train_path, test_path, ideal_functions_path):
        """
        Initialize the DataFrameToSQL object with file paths for train, test, and ideal functions.

        :param train_path: The file path for the training data CSV.
        :param test_path: The file path for the test data CSV.
        :param ideal_functions_path: The file path for the ideal functions CSV.
        """
        super().__init__(train_path, test_path, ideal_functions_path)

    def load_dataframe_to_sql(self, dataframe, table_name, database_connection='sqlite:///my_database.db'):
        """
        Load a DataFrame into a database as a table using SQLAlchemy.

        :param dataframe: The DataFrame to be loaded into the database.
        :param table_name: The name of the table to be created in the database.
        :param database_connection: The database connection string (default is 'sqlite:///my_database.db').
        """
        # Create a database connection using SQLAlchemy
        engine = create_engine(database_connection)

        try:
            # Load the DataFrame into the database as a table
            dataframe.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"DataFrame loaded as table '{table_name}' into the database successfully!")
        except Exception as e_dataframe:
            raise DataFrameToSQLException(f"Error in loading DataFrame as table into the database:\n{str(e_dataframe)}")

        # Close the database connection
        engine.dispose()

class Visualise:
    def __init__(self, train, test, ideal_functions):
        """
        Initialize the Visualise object with DataFrames for train, test, and ideal functions.

        :param train: DataFrame containing the training data.
        :param test: DataFrame containing the test data.
        :param ideal_functions: DataFrame containing the ideal functions data.
        """
        self.train = train
        self.test = test
        self.ideal_functions = ideal_functions


    def visualise_train(self):
        """
        Visualize the training data using separate line charts for each y column.
        Each y column will have its own scatter plot with x as the common x-axis.

        :raises VisualizationException: If there is an error in visualizing the training data.
        """
        try:
            # Get x-values
            x = self.train['x']
            # Get y-values for each column
            y_values = self.train[['y1', 'y2', 'y3', 'y4']]
            # Create the "train_set_visualisations" folder if it doesn't exist
            if not os.path.exists("train_set_visualisations"):
                os.makedirs("train_set_visualisations")
            # Loop over all the y columns of the train set and create a separate scatter plot for each x - y pair
            for column in y_values.columns:
                y = y_values[column]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, y, label=column)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Training Data Line Chart ({column})')
                ax.legend()
                # Save the figure with the filename as the chart title inside the "train_set_visualisations" folder
                filename = f"train_set_visualisations/{column}_line_chart.png"
                plt.savefig(filename)
                plt.close()
        except Exception as e_visualise_train:
            raise VisualizationException(f"Error in Visualising Train Set\nError: {e_visualise_train}")


    def visualise_test(self):
        """
        Visualize the test data using a scatter plot.

        :raises VisualizationException: If there is an error in visualizing the test data.
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.test['x'], self.test['y'])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Test Data Scatter Plot')
            # Create the "test_set_visualisation" folder if it doesn't exist
            if not os.path.exists("test_set_visualisation"):
                os.makedirs("test_set_visualisation")
            # Save the figure as "test_scatter_plot.png" inside the "test_set_visualisation" folder
            filename = "test_set_visualisation/test_scatter_plot.png"
            plt.savefig(filename)
            plt.close()
        except Exception as e_visualise_test:
            raise VisualizationException(f"Error in Visualising Test Set\nError: {e_visualise_test}")


    def visualise_ideal(self):
        """
        Visualize the ideal functions data using separate scatter plots for each ideal function.
        Each ideal function will have its own scatter plot with x as the common x-axis.

        :raises VisualizationException: If there is an error in visualizing the ideal functions data.
        """
        try:
            data = self.ideal_functions
            # We know that our data set always contains one "x" column
            x = data['x']
            # Get the list of 'y' column names (excluding the 'x' column)
            y_columns = [col for col in data.columns if col != 'x']

            # Plotting separate scatter plots for each 'x' and 'y' pair
            num_plots = len(y_columns)
            num_cols = 3  # Number of columns in the grid of subplots
            num_rows = (num_plots + num_cols - 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3*num_rows))
            fig.suptitle('Scatter Plot For Each Ideal Function', fontsize=12)

            for i, y_col in enumerate(y_columns):
                row_idx = i // num_cols
                col_idx = i % num_cols
                ax = axes[row_idx, col_idx]
                ax.scatter(data['x'], data[y_col])
                ax.set_xlabel('x')
                ax.set_ylabel(y_col)

            # Remove any empty subplots
            if num_plots < num_rows * num_cols:
                for i in range(num_plots, num_rows * num_cols):
                    fig.delaxes(axes.flatten()[i])

            # Adjust spacing between subplots
            plt.tight_layout(rect=[0, 0.03, 1, 0.94])

            # Create the "ideal_set_visualisations" folder if it doesn't exist
            if not os.path.exists("ideal_set_visualisations"):
                os.makedirs("ideal_set_visualisations")
                filename = "ideal_set_visualisations/ideal_functions_scatter_plot.png"
                plt.savefig(filename)
                plt.close()
            else:
                filename = "ideal_set_visualisations/ideal_functions_scatter_plot.png"
                plt.savefig(filename)
                plt.close()

        except Exception as e_visualise_ideal:
            raise VisualizationException(f"Error in Visualising Ideal Set\nError: {e_visualise_ideal}")


    def visualise_task_2_output(self, data):
        """
        Visualize the task 2 output data using a scatter plot, grouped by "Assigned Ideal Function".

        :param data: DataFrame containing the task 2 output data.
        :raises VisualizationException: If there is an error in visualizing the task 2 output data.
        """
        try:
            plt.figure(figsize=(10, 6))

            # Group the DataFrame by "Assigned Ideal Function"
            grouped_data = data.groupby("Assigned Ideal Function")

            # Create a scatter plot for each group with a unique color
            for name, group in grouped_data:
                plt.scatter(group["Test (X)"], group["Test (Y)"], label=name)

            # Set plot labels and title
            plt.xlabel("Test (X)")
            plt.ylabel("Test (Y)")
            plt.title("Scatter Plot of Task 2 Results")
            plt.legend()
            return plt.savefig("Scatter Plot of Task 2 Results.png")

        except Exception as e_visualise:
            raise VisualizationException(f"Error in Visualising Task 2 Output\nError: {e_visualise}")

class IdealFunctionSelector:
    def __init__(self, train, ideal_functions):
        """
        Initialize the IdealFunctionSelector object with DataFrames for train and ideal functions.

        :param train: DataFrame containing the training data.
        :param ideal_functions: DataFrame containing the ideal functions data.
        """
        self.train = train
        self.ideal_functions = ideal_functions

    def calculate_lse(self, y1, y2):
        """
        Calculate the Least Squared Error (LSE) between two 'y' features.

        :param y1: The first 'y' feature as a NumPy array or Pandas Series.
        :param y2: The second 'y' feature as a NumPy array or Pandas Series.

        :return: The LSE value as a float.
        """
        return np.sum((y1 - y2) ** 2)

    def select_ideal_functions(self):
        """
        Select the ideal functions that best match the training data based on LSE.

        For each 'y' feature in the training data, find the 'y' feature from the ideal functions
        that has the least squared error (LSE) with the corresponding 'y' feature in the training data.

        :return: A DataFrame containing the results of the ideal function selection, including the
                 train 'y' column, the corresponding ideal 'y' column, and the calculated LSE.
        :raises IdealFunctionSelectorException: If there is an error during the ideal function selection process.
        """
        try:
            # Retrieve all the "y" columns for train and ideal
            y_columns_train = [col for col in self.train.columns if col.startswith('y')]
            y_columns_ideal = [col for col in self.ideal_functions.columns if col.startswith('y')]

            # Create an empty list to store the results
            results_list_lse = []

            # Loop through each 'y' feature in the train set
            for y_col_train in y_columns_train:
                # Get the 'y' feature from the train set
                y_train = self.train[y_col_train]

                # Initialize variables to keep track of the best match
                best_match = None
                best_lse = float('inf')  # Initialize with a high value

                # Loop through each 'y' feature in the ideal set
                for y_col_ideal in y_columns_ideal:
                    # Get the 'y' feature from the ideal set
                    y_ideal = self.ideal_functions[y_col_ideal]

                    # Calculate the squared deviation between the train and ideal 'y' features and assign to lse
                    lse = self.calculate_lse(y_train, y_ideal)

                    # Update the best match if a better match is found
                    if lse < best_lse:
                        best_match = y_col_ideal
                        best_lse = lse

                # Append the best match for the current 'y' feature to the results list
                results_list_lse.append({
                    'train': y_col_train,
                    'ideal': best_match,
                    'LSE': best_lse
                })

            # Create a DataFrame from the results list
            results_df_lse = pd.concat([pd.DataFrame([result]) for result in results_list_lse], ignore_index=True)

            # Return the final DataFrame with the results
            return results_df_lse

        except Exception as e_ideal_functions_selector:
            raise IdealFunctionSelectorException(f"Error encountered: {e_ideal_functions_selector}")


class IdealFunctionMapper:
    def __init__(self, test, task_1_output, ideal_functions):
        """
        Initialize the IdealFunctionMapper object with DataFrames for test, task 1 output, and ideal functions.

        :param test: DataFrame containing the test data.
        :param task_1_output: DataFrame containing the output of the IdealFunctionSelector for task 1.
        :param ideal_functions: DataFrame containing the ideal functions data.
        """
        self.test = test
        self.task_1_output = task_1_output
        self.ideal_functions = ideal_functions
        # initialise the IdealFunctionSelector class so that we can reuse the calculate_lse() function
        self.selector = IdealFunctionSelector(train, ideal_functions)

    def map_ideal_functions(self):
        """
        Map the test data to the best matching ideal function based on the LSE comparison.

        For each row in the test data, find the best matching ideal function by calculating the LSE with each ideal
        'y' feature. If the lowest LSE is less than a threshold value (MAX(LSE from task_1_output) * sqrt(2)),
        the ideal function is assigned; otherwise, "None" is assigned to denote no matching ideal function.

        :return: A DataFrame containing the results of the ideal function mapping for the test data.
        :raises IdealFunctionMapperException: If there is an error during the ideal function mapping process.
        """
        try:
            # Create an empty list to store the results
            task2_results_list = []

            # Get the list of ideal 'y' column names from task_1_output
            selected_ideal_y_columns = list(self.task_1_output['ideal'])

            # Loop through each row in the test set
            for index, row in self.test.iterrows():
                x_value = row['x']
                y_value = row['y']

                # Initialize variables to keep track of the best match for the current row
                best_match = None
                best_lse = float('inf')  # Initialize with a high value

                # Calculate LSE for each 'y' feature in the ideal set for the current row's test "y" value
                for y_col_ideal in selected_ideal_y_columns:
                    # Get the 'y' feature from the ideal set
                    y_ideal_values = self.ideal_functions[y_col_ideal]

                    # Loop through each value in the 'y' feature column of the ideal set
                    for y_ideal_value in y_ideal_values:
                        # Calculate the LSE between the test 'y' value and the current 'y' feature value in the ideal set
                        lse = self.selector.calculate_lse(y_value, y_ideal_value)

                        # Update the best match if a better match is found
                        if lse < best_lse:
                            best_match = y_col_ideal
                            best_lse = lse

                # Compare with MAX(LSE from task_1_output) * sqrt(2)
                max_lse_task1 = self.task_1_output['LSE'].max()
                threshold_lse = max_lse_task1 * np.sqrt(2)

                # Assign the ideal function if lowest LSE < threshold LSE
                if best_lse < threshold_lse:
                    assigned_ideal = best_match
                    deviation = best_lse
                # if no value lesser than threshold than assign "None" to denote no matching ideal function
                else:
                    assigned_ideal = "None"
                    deviation = None

                # Store the result in the DataFrame
                task2_results_list.append({
                    'Test (X)': x_value,
                    'Test (Y)': y_value,
                    'Deviation': deviation,
                    'Assigned Ideal Function': assigned_ideal})

            # Create a DataFrame from the results list
            task2_results_df_lse = pd.concat([pd.DataFrame([result]) for result in task2_results_list], ignore_index=True)

            # return the final DataFrame with the results for task 2
            return task2_results_df_lse

        except Exception as e_ideal_functions_mapper:
            raise IdealFunctionMapperException(f"Error encountered: {e_ideal_functions_mapper}")

if __name__ == "__main__":
    try:
        # Constant Variables
        # Update the PATH variables as per your system
        TRAIN_PATH = 'train_MSCS.csv'
        TEST_PATH = 'test_MSCS.csv'
        IDEAL_FUNCTIONS_PATH = 'ideal_MSCS.csv'

        # create an instance of LoadData Class, with file paths as the parameters
        load_data = LoadData(TRAIN_PATH, TEST_PATH, IDEAL_FUNCTIONS_PATH)

        # call the function load_csv_to_database to store the data into the DB and return the files as DataFrames
        train, ideal_functions, test = load_data.load_csv_to_database()
        print(f"Train Shape: {train.shape}\nIdeal Functions Shape: {ideal_functions.shape}\nTest Shape: {test.shape}\n")

        # create an instance of Visualise class and now pass the 3 DataFrames as parameters
        visualiser = Visualise(train, test, ideal_functions)
        visualiser.visualise_train()
        visualiser.visualise_test()
        visualiser.visualise_ideal()

        """
        Run the select_ideal_functions() function from the IdealFunctionSelector Class
        The output will be a DataFrame which would be used as an input parameter to the next Class
        """
        task_1_output = IdealFunctionSelector(
            train, ideal_functions).select_ideal_functions()
        print("Selected Ideal Functions based on Train Set\n",task_1_output)

        """
        Run map_ideal_functions() from IdealFunctionMapper with 
        test, task 1 output and ideal function DataFrames as parameters
        """
        task_2_output = IdealFunctionMapper(
            test, task_1_output, ideal_functions).map_ideal_functions()

        # Visualise the task 2 results with each data point color coded with the assigned ideal function
        visualiser.visualise_task_2_output(task_2_output)

        # the final output Dataframe will be loaded into the DB using the DataFrameToSQL child class
        data_loader = DataFrameToSQL(
            TRAIN_PATH, TEST_PATH, IDEAL_FUNCTIONS_PATH)
        # load_dataframe_to_sql() takes in DataFrame, table name as function aruguments
        data_loader.load_dataframe_to_sql(task_2_output, 'task_2_output')
        print("Task Completed!")

    except DataLoadException as e_data_load:
        print(f"Data loading error: {str(e_data_load)}")
    except DatabaseConnectionException as e_db_connection:
        print(f"Database connection error: {str(e_db_connection)}")
    except VisualizationException as e_visualize_exp:
        print(f"Visualization error: {str(e_visualize_exp)}")
    except IdealFunctionSelectorException as e_ideal_selector:
        print(f"Ideal function selection error: {str(e_ideal_selector)}")
    except IdealFunctionMapperException as e_ideal_mapper:
        print(f"Ideal function mapping error: {str(e_ideal_mapper)}")
    except DataFrameToSQLException as e_df_to_sql:
        print(f"DataFrame to SQL error: {str(e_df_to_sql)}")
    except Exception as e_general:
        print(f"An unexpected error occurred:\n{str(e_general)}")
