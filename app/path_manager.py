import sys
import os

class Datapath:

    def __init__(self, data_dir, track_year, features_type):
        self.data_dir = data_dir
        self.track_year = track_year
        self.features_type = features_type


    def get_root_dir(self):
        '''
            returns the root directory of the data
        '''
        return self.data_dir


    def get_data_dir(self):
        '''
            returns the directory of the data
        '''
        return os.path.join(self.get_root_dir(), 'data')


    def get_data_raw_dir(self):
        '''
            returns the directory of the raw data
        '''
        return os.path.join(self.get_data_dir(), 'raw', str(self.track_year))


    def get_data_raw_train_dir(self):
        '''
            returns the directory of the raw train dataset
        '''
        return os.path.join(self.get_data_raw_dir(), 'train')


    def get_data_raw_test_dir(self):
        '''
            returns the directory of the raw test dataset
        '''
        return os.path.join(self.get_data_raw_dir(), 'test')


    def get_data_processed_dir(self):
        '''
            returns the directory of the processed data
        '''
        return os.path.join(self.get_data_dir(), 'processed', str(self.track_year))


    def get_resources_dir(self):
        '''
            returns the directory of the features
        '''
        return os.path.join(self.get_root_dir(), 'resources', str(self.track_year))


    def get_resources_word2vec_dir(self):
        '''
            returns the directory of the word2vec models
        '''
        return os.path.join(self.get_resources_dir(), 'word2vec')


    def get_resources_word2vec_googlenews_model_path(self):
        '''
            returns the path of the word2vec googlenews model
        '''
        return os.path.join(self.get_resources_word2vec_dir(), 'googlenews', 'GoogleNews-vectors-negative300.bin')


    def get_features_train_dir(self):
        '''
            returns the directory of the trainset features
        '''
        return os.path.join(self.get_root_dir(), 'features', str(self.track_year), 'train', self.features_type)


    def get_features_test_dir(self):
        '''
            returns the directory of the testset features
        '''
        return os.path.join(self.get_root_dir(), 'features', str(self.track_year), 'test', self.features_type)


    def get_models_dir(self):
        '''
            returns the directory of the models
        '''
        return os.path.join(self.get_root_dir(), 'models', str(self.track_year), self.features_type)


    def get_prediction_dir(self):
        '''
            returns the prediciton directory
        '''
        return os.path.join(self.get_root_dir(), 'prediction', str(self.track_year), self.features_type)


    def get_output_dir(self):
        '''
            returns the outputt directory
        '''
        return os.path.join(self.get_root_dir(), 'output', str(self.track_year), self.features_type)


    def get_results_dir(self):
        '''
            returns the directory of the results
        '''
        return os.path.join(self.get_root_dir(), 'results', str(self.track_year), self.features_type)


    def get_visualization_dir(self):
        '''
            returns the directory of the visualization
        '''
        return os.path.join(self.get_root_dir(), 'visualization', str(self.track_year), self.features_type)

