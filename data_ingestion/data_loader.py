import pandas as pd
import os

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self,csv_data='datasets/dataset.csv'):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            data = pd.read_csv(csv_data) # reading the data file
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return data
        except Exception as e:
            os.remove(csv_data)
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()