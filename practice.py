import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker
from math import sqrt
import sys
import re

print("Libraries loaded successfully!")

# Custom Exception classes
class DataLoadException(Exception):
    pass

class DatabaseConnectionException(Exception):
    pass

class DataFrameToSQLException(Exception):
    pass

class VisualizationException(Exception):
    pass

class IdealFunctionSelectorException(Exception):
    pass

class IdealFunctionMappingException(Exception):
    pass

class LoadData():
    def __init__(self,train_path, test_path, ideal_functions_path):
        self.train_path = train_path
        self.test_path = test_path
        self.ideal_functions_path = ideal_functions_path
    
    def load_csv_to_database(self):
        database_connection = 'sqlite:///my_database.db'
        try:
            # Create a database connection using SQLAlchemy
            engine = create_engine(database_connection)
            files_list = [self.train_path, self.test_path, self.ideal_functions_path]
            error_flag = False  # Initialize error flag as False

            # Load CSV files into DataFrames
            for file in files_list:
                if 'train' in file:
                    try:
                        train = pd.read_csv(file)
                        # Load DataFrames into the database as tables
                        train.to_sql('train_data', engine, if_exists='replace', index=False)
                    except Exception as e:
                        raise DataLoadException(f"Error in loading Train Set from {file}:\n{str(e)}")
                        error_flag = True

                elif 'ideal' in file:
                    try:
                        ideal_functions = pd.read_csv(file)
                        # Load DataFrames into the database as tables
                        ideal_functions.to_sql('ideal_functions_data', engine, if_exists='replace', index=False)
                    except Exception as e:
                        raise DataLoadException(f"Error in loading Ideal Set from {file}:\n{str(e)}")
                        error_flag = True

                elif 'test' in file:
                    try:
                        test = pd.read_csv(file)
                        # Load DataFrames into the database as tables
                        test.to_sql('test_data', engine, if_exists='replace', index=False)
                    except Exception as e:
                        raise DataLoadException(f"Error in loading Test Set from {file}:\n{str(e)}")
                        error_flag = True

            # Close the database connection
            engine.dispose()

            # Print "All Files loaded successfully!" if no errors occurred during loading
            if not error_flag:
                print("All Files loaded into the DB successfully!")
                return train, ideal_functions, test
        except Exception as e:
            raise DatabaseConnectionException(f"Error in creating the database connection:\n{str(e)}")

#Child Class
class DataFrameToSQL(LoadData):
    def __init__(self, train_path, test_path, ideal_functions_path):
        super().__init__(train_path, test_path, ideal_functions_path)

    def load_dataframe_to_sql(self, dataframe, table_name, database_connection='sqlite:///my_database.db'):
        # Create a database connection using SQLAlchemy
        engine = create_engine(database_connection)

        try:
            # Load the DataFrame into the database as a table
            dataframe.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"DataFrame loaded as table '{table_name}' into the database successfully!")
        except Exception as e:
            raise DataFrameToSQLException(f"Error in loading DataFrame as table into the database:\n{str(e)}")

        # Close the database connection
        engine.dispose()

class Visualise():
    def __init__(self, train, test, ideal_functions):
        self.train = train
        self.test = test
        self.ideal_functions = ideal_functions
    
    def visualise_train(self):
        try:
            # Get x-values
            x = self.train['x']
            # Get y-values for each column
            y_values = self.train[['y1', 'y2', 'y3', 'y4']]
            # Create a separate plot for each y-value
            for column in y_values.columns:
                y = y_values[column]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, y, label=column)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Training Data Line Chart ({column})')
                ax.legend()
                plt.show()
        except Exception as e:
            raise VisualizationException(f"Error in Visualising Train Set\nError:{e}")
        
    def visualise_test(self):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.test['x'], self.test['y'])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Test Data Scatter Plot')
            plt.show()
        except Exception as e:
            raise VisualizationException(f"Error in Visualising Test Set\nError:{e}")
    
    def visualise_ideal(self):
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

            plt.tight_layout(rect=[0, 0.03, 1, 0.94])  # Adjust spacing between subplots
            plt.show()
        
        except Exception as e:
            raise VisualizationException(f"Error in Visualising Ideal Set\nError:{e}")
    
    
    def visualise_task_2_output(self,data):
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
        
        except Exception as e:
            raise VisualizationException(f"Error in Visualising Test Set\nError:{e}")
    

class IdealFunctionSelector():
    def __init__(self, train, ideal_functions):
        self.train = train
        self.ideal_functions = ideal_functions
    
    def calculate_lse(self, y1, y2):
        return np.sum((y1 - y2) ** 2)

    def select_ideal_functions(self):
        try:
            # Columns to consider for matching (exclude 'x' column)
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

                    # Calculate the sum of squared errors (SSE) between the train and ideal 'y' features
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
        
        except Exception as e:
            raise IdealFunctionSelectorException(f"Error encountered:{e}")

class IdealFunctionMapper():
    def __init__(self, test, task_1_output, ideal_functions):
        self.test = test
        self.task_1_output = task_1_output
        self.ideal_functions = ideal_functions
        self.selector = IdealFunctionSelector(train, ideal_functions)
    
    def map_ideal_functions(self):
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

                # Calculate LSE for each 'y' feature in the ideal set for the current row
                for y_col_ideal in selected_ideal_y_columns:
                    # Get the 'y' feature from the ideal set
                    y_ideal_values = self.ideal_functions[y_col_ideal]

                    # Loop through each value in the 'y' feature column of the ideal set
                    for y_ideal_value in y_ideal_values:
                    # Calculate the LSE between the test 'Y' value and the current 'y' feature value in the ideal set
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
        
        except Exception as e:
            raise IdealFunctionMapperException(f"Error encountered:{e}")

            
if __name__ == "__main__":
    try:
        # Constant Variables
        TRAIN_PATH = 'train_MSCS.csv'
        TEST_PATH = 'test_MSCS.csv'
        IDEAL_FUNCTIONS_PATH = 'ideal_MSCS.csv'

        load_data = LoadData(TRAIN_PATH, TEST_PATH, IDEAL_FUNCTIONS_PATH)
        train, ideal_functions, test = load_data.load_csv_to_database()

        visualiser = Visualise(train, test, ideal_functions)
        visualiser.visualise_train()
        visualiser.visualise_test()
        visualiser.visualise_ideal()

        task_1_output = IdealFunctionSelector(train, ideal_functions).select_ideal_functions()

        task_2_output = IdealFunctionMapper(test, task_1_output, ideal_functions).map_ideal_functions()

        visualiser.visualise_task_2_output(task_2_output)

        data_loader = DataFrameToSQL(TRAIN_PATH, TEST_PATH, IDEAL_FUNCTIONS_PATH)
        data_loader.load_dataframe_to_sql(task_2_output, 'task_2_output')

    except DataLoadException as e:
        print(f"Data loading error: {str(e)}")
    except DatabaseConnectionException as e:
        print(f"Database connection error: {str(e)}")
    except VisualizationException as e:
        print(f"Visualization error: {str(e)}")
    except IdealFunctionSelectorException as e:
        print(f"Ideal function selection error: {str(e)}")
    except IdealFunctionMappingException as e:
        print(f"Ideal function mapping error: {str(e)}")
    except DataFrameToSQLException as e:
        print(f"DataFrame to SQL error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred:\n{str(e)}")