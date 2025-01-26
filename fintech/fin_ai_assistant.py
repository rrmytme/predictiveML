# Version: 1.0

# Description: This file contains the necessary imports for the fin_ai_assistant.py file.

# os is used to interact with the operating system
# json is used to work with JSON data
# random is used to generate random numbers
import os
import json
import random

# pickle is used to serialize and deserialize a Python object structure
import pickle

# union is used to specify that the function can return either a string or a list of strings
from typing import Union

# nltk is a leading platform for building Python programs to work with human language data 
import nltk
# numpy is a library used for working with arrays
import numpy as np

# environ is used to get the value of the environment variable  
# here we are setting the value of the environment variable to 3
# env values are 0, 1, 2, 3 -> 0: no logs, 1: logs, 2: warnings, 3: errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tensorflow is an open-source machine learning library
# sequential is used to create a linear stack of layers and load_model is used to load a model
from tensorflow.keras.models import Sequential, load_model # type: ignore

# layers is used to add layers to the model
# dense is used to add a densely-connected neural network layer
# dropout is used to prevent overfitting by randomly setting a fraction of input units to 0
# input_layer is used to add an input layer to the model  
from tensorflow.keras.layers import Dense, Dropout # type: ignore

# adam is an optimization algorithm used to update network weights iteratively based on training data
# optimizer is used to compile the model with the optimizer
from tensorflow.keras.optimizers import Optimizer # type: ignore

class FinAssistant:
    # The __init__ method is a special method that initializes the object
    def __init__(self, intents_data: Union[str, os.PathLike, dict], method_mappings: dict = {}, hidden_layers: list = None, model_name: str = "fin_basic_model") -> None:
        
        # Download the required nltk data
        # punkt is used for tokenization
        # wordnet is used for lemmatization 
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)

        if isinstance(intents_data, dict):
            self.intents_data = intents_data
        else:
            if os.path.exists(intents_data):
                with open(intents_data, "r") as f:
                    self.intents_data = json.load(f)
            else:
                raise FileNotFoundError

        self.method_mappings = method_mappings
        self.model = None
        self.hidden_layers = hidden_layers
        self.model_name = model_name
        self.history = None

        # stemmer is used to reduce words to their root form exapmle: running -> run
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.words = []
        self.intents = []

        self.training_data = []

    # _prepare_intents_data is a private method that prepares the intents data
    def _prepare_intents_data(self, ignore_letters: tuple = ("!", "?", ",", ".")):
        # Prepare the training data
        # documents is a list of tuples containing the tokenized words and the intent tag
        documents = []

        # Iterate through the intents data
        for intent in self.intents_data["intents"]:
            # Add the intent tag to the list of intents
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])

            # Tokenize the patterns and add the words to the list of words
            for pattern in intent["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                self.words += pattern_words
                documents.append((pattern_words, intent["tag"]))

        # Lemmatize the words and remove the ignore letters
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        # Sort and remove duplicates from the list of words
        self.words = sorted(set(self.words))

         # empty_output is a list of zeros with the length of the intents
        empty_output = [0] * len(self.intents)

        # iterate through the documents
        for document in documents:
            # bag_of_words is a list of zeros with the length of the words
            bag_of_words = []

            # lemmatize the words in the pattern and convert them to lowercase
            pattern_words = document[0]
            pattern_words = [self.lemmatizer.lemmatize(w.lower()) for w in pattern_words]
            # iterate through the words
            for word in self.words:
                # append 1 if the word is in the pattern words, otherwise append 0
                bag_of_words.append(1 if word in pattern_words else 0)

            # output_row is a copy of empty_output
            output_row = empty_output.copy()
            # set the value at the index of the intent tag to 1
            output_row[self.intents.index(document[1])] = 1
            # append the bag_of_words and output_row to the training data
            self.training_data.append([bag_of_words, output_row])

        random.shuffle(self.training_data)
        # Convert the training data to a numpy array
        self.training_data = np.array(self.training_data, dtype=object)

        # data[0] is the bag_of_words and data[1] is the output_row
        X = np.array([np.array(data[0], dtype=np.float32) for data in self.training_data])
        y = np.array([np.array(data[1], dtype=np.float32) for data in self.training_data])

        # Return the X -> bag_of_words and y -> tag  
        return X, y
        
    # _create_model is a private method that creates the model
    def _create_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(128, input_shape=(input_dim,), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
    
    # train_model is a method that trains the model
    def train_model(self, epochs: int = 200, optimizer: Optimizer = None):
        # Prepare the intents data
        X, y = self._prepare_intents_data()
        # Create the model
        self.model = self._create_model(X.shape[1], y.shape[1],)
        # Fit the model with the training data
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=5, verbose=1) 

    # save_model is a method that saves the model
    def save_model(self):
        # Save the model
        self.model.save(f"{self.model_name}.keras", self.history)
        # Save the words and intents
        pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
        pickle.dump(self.intents, open(f'{self.model_name}_intents.pkl', 'wb'))
          
    # load_model is a method that loads the model and the words and intents from files
    def load_model(self):
        # Load the model
        self.model = load_model(f'{self.model_name}.keras')
        # Load the words and intents
        self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
        self.intents = pickle.load(open(f'{self.model_name}_intents.pkl', 'rb'))

    # _predict_intent is a private method that predicts the intent of the input text

    def _predict_intent(self, input_text: str):
        # Tokenize the input text
        input_words = nltk.word_tokenize(input_text)
        # Lemmatize the words and convert them to lowercase
        input_words = [self.lemmatizer.lemmatize(w.lower()) for w in input_words]
        # Create a bag_of_words with zeros
        bag_of_words = [0] * len(self.words)
        # Iterate through the words
        for word in input_words:
            # Set the value at the index of the word to 1
            if word in self.words:
                bag_of_words[self.words.index(word)] = 1

        # Reshape the bag_of_words
        bag_of_words = np.array(bag_of_words).reshape(1, -1)
        # Predict the intent with the model
        prediction = self.model.predict(bag_of_words)
        # Get the index of the highest value
        max_index = np.argmax(prediction)
        # Get the tag with the highest value
        tag = self.intents[max_index]
        # Get the probability of the tag
        probability = prediction[0][max_index]
        return tag, probability

    # get_response is a method that gets the response based on the predicted intent
    def get_response(self, input_text: str):
        # Predict the intent
        intent, probability = self._predict_intent(input_text)
        # Get the responses from the intents data
        for intent_data in self.intents_data["intents"]:
            if intent_data["tag"] == intent:
                responses = intent_data["responses"]
                break

        # Get the method mapping
        method = self.method_mappings.get(intent)
        if method:
            return method()
        else:
            return random.choice(responses)

    # process_input is a method that processes the input text and returns a response
    def process_input(self, input_text: str):
        return self.get_response(input_text)

    # get_intents is a method that returns the intents
    def get_intents(self):
        return self.intents

    # get_words is a method that returns the words
    def get_words(self):
        return self.words

    # get_model is a method that returns the model
    def get_model(self):
        return self.model

    # get_history is a method that returns the history
    def get_history(self):
        return self.history

    # get_training_data is a method that returns the training data
    def get_training_data(self):
        return self.training_data

    # get_intents_data is a method that returns the intents data
    def get_intents_data(self):
        return self.intents_data

    # get_method_mappings is a method that returns the method mappings
    def get_method_mappings(self):
        return self.method_mappings

class BotAssistant(FinAssistant):
    def __init__(self, *args, **kwargs):
    # import warnings
    # warnings.warn("Test warnings if any :P", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)    


         
    
