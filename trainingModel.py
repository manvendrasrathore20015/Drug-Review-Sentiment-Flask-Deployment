from applicaton_logging import logger
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        #self.file_object = open("log_file/ModelTraining_log.txt", 'a+')
        self.model = LinearSVC('l2')
        self.normalize = Normalizer()

    def trainingModel(self,csv_data="old_data",file_object=open("log_file/ModelTraining_log.txt", 'a+')):
        # Logging the start of Training
        self.log_writer.log(file_object, '=========== Start of Training =============')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(file_object, self.log_writer)
            if (csv_data=="old_data"):
                data = data_getter.get_data()
            else:
                data = data_getter.get_data(csv_data)
            """ doing the data preprocessing . 
            All the pre processing steps are based on the EDA done previously
            """
            """
            1. Initializing columns
            2. Changing null values to Other category in Condtion column
            3. Null removal
            4. Removing stopwords 
            5. Adding important Features 
            6. Vectorization 
            """
            # initializing preprocessor class
            preprocessor = preprocessing.Preprocessor(file_object, self.log_writer)
            # initializing columns in data
            data = preprocessor.initialize_columns(data)
            # replacing missing values in condition feature in data
            data = preprocessor.replace_missing_in_condition(data)
            # removing rows containing null values
            data = preprocessor.remove_null_values(data)
            # seperating important features
            new_data = preprocessor.separate_imp_feature(data)
            # removing stopwords from review column
            new_data = preprocessor.remove_stopwords(new_data)
            # adding new features n scores for better results
            new_data = preprocessor.adding_new_features(new_data)
            # vectorizing the dataset into features and labels
            features, labels, tfidf = preprocessor.vectorizor(new_data)

            model = LinearSVC('l2')
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
            x_train = self.normalize.fit_transform(x_train)
            x_test = self.normalize.transform(x_test)
            model.fit(x_train, y_train)

            # saving the data n model in pickle files ..
            import pickle
            file = open('pickle_files/drug_LinearSVC.pkl', 'wb')
            pickle.dump(model, file)

            file = open('pickle_files/d_transform.pkl', 'wb')
            pickle.dump(tfidf, file)

            file = open('pickle_files/drug_LinearSVC.pkl', 'rb')
            ml = pickle.load(file)
            t = pickle.load(open('pickle_files/d_transform.pkl', 'rb'))
            self.log_writer.log(file_object, '=========== Training Succesfull =============')
        except Exception as e:
            self.log_writer.log(file_object,
                                'Exception occured in trainingModel method of the trainModel class. Exception message:  ' + str(e))
            raise Exception()

