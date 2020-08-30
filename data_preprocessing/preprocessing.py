import pandas as pd
import numpy as np
import nltk
import vaderSentiment
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training .
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.analyzer = SentimentIntensityAnalyzer()
        self.new_stop = stopwords

    def initialize_columns(self, data):
        """
                Method Name: initialize_columns
                Description: This method gives columns names for a pandas dataframe.
                Output: A pandas DataFrame initializing each column a proper name .
                On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the initialize_columns method of the Preprocessor class')
        self.data = data

        try:
            self.data.columns = ['Id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
            self.logger_object.log(self.file_object,
                                   'Column name initialized successfully . Exited the initialize_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in initialize_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Column name initialization Unsuccessful. Exited the initialize_columns method of the Preprocessor class')
            raise Exception()

    def replace_missing_in_condition(self, data):
        """
                Method Name: replace_missing_values
                Description: This method replaces all null/missing values to Not listed or Other category in condtion column .
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the replace_missing_in_condition method of the Preprocessor class')
        self.data = data
        try:
            self.data.condition = np.where(data["condition"] == "Not Listed / Othe", "Not Listed / Other",
                                           data["condition"])
            if (self.data.condition.isnull().sum() > 0):
                # replacing missing values to "Not Listed / Other" in condition column
                self.data.condition = np.where(data["condition"].isnull(), "Not Listed / Other", data["condition"])
                self.logger_object.log(self.file_object,
                                       'Null values in condition column replaced . Exited the replace_missing_in_condition method of the Preprocessor class')
            else:
                self.logger_object.log(self.file_object,
                                       "No null values to replace in condition column . Exited the replace_missing_in_condition method of the Preprocessor class'")
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in replace_missing_in_condition method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Replacing null values in condition column Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def remove_null_values(self, data):
        """
                Method Name: remove_null_values
                Description: This method drops the rows with null values in the pandas dataframe.
                Output: A pandas DataFrame after removing the rows with null values .
                On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the remove_null_values method of the Preprocessor class')
        self.data = data
        try:
            if (self.data.isnull().sum().any() == True):
                self.data = self.data.dropna(axis=0)  # droping rows having null values
                self.logger_object.log(self.file_object,
                                       'Rows removal Successful.Exited the remove_null_values method of the Preprocessor class')
                return self.data
            else:
                self.logger_object.log(self.file_object,
                                       'No null values present . Exited the remove_null_values method of the Preprocessor class')
                return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_null_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Rows removal Unsuccessful. Exited the remove_null_values method of the Preprocessor class')
            raise Exception()

    def separate_imp_feature(self, data):
        """
                        Method Name: separate_imp_feature
                        Description: This method separates the important features and and save it to a seperate pandas dataframe .
                        Output: Returns a separate Dataframes, one containing only important features for prediction .
                        On Failure: Raise Exception
         """
        self.logger_object.log(self.file_object, 'Entered the separate_imp_feature method of the Preprocessor class')
        try:
            self.new_data = self.data[['Id', 'review', 'rating']].copy()  # using
            # important features only ..
            self.logger_object.log(self.file_object,
                                   'Features Separation Successful . Exited the separate_imp_feature method of the Preprocessor class ')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_imp_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Features Separation Unsuccessful. Exited the separate_imp_feature method of the Preprocessor class')
            raise Exception()

    def remove_stopwords(self, new_data):
        """
                        Method Name: remove_stopwords
                        Description: This method removes stopwords from  .
                        Output: Returns a Dataframes, one in which stopwords are removed in review column .
                        On Failure: Raise Exception .
        """
        self.logger_object.log(self.file_object, 'Entered the remove_stopwords method of the Preprocessor class')
        try:
            ss = self.new_stop.words('english')
            # remove stopwords from review
            self.new_data['review'] = self.new_data["review"].map(lambda x: x.lower())
            self.new_data['cleanReview'] = self.new_data['review'].apply(
                lambda x: ' '.join([item for item in x.split() if item not in ss]))
            self.logger_object.log(self.file_object,
                                   "Stopwords removal Successful . Exited the remove_stopwords method of the Preprocessor class")

            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_stopwords method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Stopwords removal Unsuccessful. Exited the separate_imp_feature method of the Preprocessor class')
            raise Exception()

    def adding_new_features(self, new_data):
        """
                        Method Name: adding_new_features
                        Description: This method adds new features like vaderReviewScore , ratingSentiment , etc in the pandas dataframe .
                        Output: Returns a Dataframes, one containing few features which are useful in prediction .
                        On Failure: Raise Exception .
        """
        self.logger_object.log(self.file_object, 'Entered the adding_new_features method of the Preprocessor class')
        try:
            # vander review score
            self.new_data['vaderReviewScore'] = self.new_data['cleanReview'].apply(
                lambda x: self.analyzer.polarity_scores(x)['compound'])
            # making reviews as positive , negative and neutral
            positive_num = len(self.new_data[self.new_data['vaderReviewScore'] >= 0.05])
            neutral_num = len(
                self.new_data[(self.new_data['vaderReviewScore'] > -0.05) & (self.new_data['vaderReviewScore'] < 0.05)])
            negative_num = len(self.new_data[self.new_data['vaderReviewScore'] <= -0.05])
            # Sentiment analysis ..
            # 0-neutral , 1-negative , 2-positive ..
            self.new_data['vaderSentiment'] = self.new_data['vaderReviewScore'].map(
                lambda x: int(2) if x >= 0.05 else int(1) if x <= -0.05 else int(0))
            # Classifying result
            self.new_data.loc[self.new_data['vaderReviewScore'] >= 0.05, "vaderSentimentLabel"] = "positive"
            self.new_data.loc[(self.new_data['vaderReviewScore'] > -0.05) & (
                        self.new_data['vaderReviewScore'] < 0.05), "vaderSentimentLabel"] = "neutral"
            self.new_data.loc[self.new_data['vaderReviewScore'] <= -0.05, "vaderSentimentLabel"] = "negative"
            positive_rating = len(self.new_data[self.new_data['rating'] >= 7.0])
            neutral_rating = len(self.new_data[(self.new_data['rating'] >= 4) & (self.new_data['rating'] < 7)])
            negative_rating = len(self.new_data[self.new_data['rating'] <= 3])
            self.new_data['ratingSentiment'] = self.new_data['rating'].map(
                lambda x: int(2) if x >= 7 else int(1) if x <= 3 else int(0))
            self.new_data.loc[self.new_data['rating'] >= 7.0, "ratingSentimentLabel"] = "positive"
            self.new_data.loc[
                (self.new_data['rating'] >= 4.0) & (self.new_data['rating'] < 7.0), "ratingSentimentLabel"] = "neutral"
            self.new_data.loc[self.new_data['rating'] <= 3.0, "ratingSentimentLabel"] = "negative"
            self.new_data = self.new_data[
                ['Id', 'review', 'cleanReview', 'rating', 'ratingSentiment', 'ratingSentimentLabel', 'vaderReviewScore',
                 'vaderSentiment', 'vaderSentimentLabel']]

            self.logger_object.log(self.file_object,
                                   "New Features Addition Successful . Exited the adding_new_features method of the Preprocessor class")
            return self.new_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in adding_new_features method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'New Features Addition Unsuccessful. Exited the adding_new_features method of the Preprocessor class')
            raise Exception()

    '''
    def save_preprocessed_file(self,new_data):
        """
                       Method Name: save_preprocessed_file
                       Description: This method saves the pandas Dataframe in pickle file .
                       Output: Returns a pickle file containing a pandas Dataframe .
                       On Failure: Raise Exception .
       """
        self.logger_object.log(self.file_object, 'Entered the save_preprocessed_file method of the Preprocessor class')
        try:
            self.new_data.to_csv('D:\Drug_Review_Sentiment\processed.csv')
            # Compressing csv file ..
            self.new_data.to_csv('D:\Drug_Review_Sentiment\processed.csv.gz',compression='gzip')
            self.logger_object(self.file_object,"New csv file created successfully . Exited the save_preprocessed_file method of the Preprocessor class")
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in save_preprocessed_file method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'New csv file creation Unsuccessful . Exited the save_preprocessed_file method of the Preprocessor class')
            raise Exception()
        '''

    def vectorizor(self, new_data):
        """         Method Name: vertorizor .
                    Description: This method adds new features like vaderReviewScore , ratingSentiment , etc in the pandas dataframe .
                    Output: Returns a Dataframes , one containing few features which are useful in prediction .
                    On Failure: Raise Exception .
        """
        self.logger_object.log(self.file_object, 'Entered the save_preprocessed_file method of the Preprocessor class')
        try:
            # Vectorizing .. (using Term Frequency – Inverse Document Frequency)
            self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(
            1, 2))  # n_grams-can be used for auto spell checks , auto spell checks..
            # bigram model used ..
            self.features = self.tfidf.fit_transform(self.new_data.cleanReview)
            self.labels = self.new_data.vaderSentiment
            self.logger_object.log(self.file_object,
                                   "Vertorization Successful . Exited the vertorizor method of the Preprocessor class")
            return self.features, self.labels, self.tfidf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in vertorizor method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Vectortization Unsuccessful. Exited the vertorizor method of the Preprocessor class')
            raise Exception()