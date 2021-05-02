import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

#Install the requisite libraries
import nltk #Text preprocessing library
import joblib #For saving and loading ML models
import keras #For loading saved models

#Downloading the relevant libraries and dependencies in NLTK module for preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Regular Expressions
import re

#Text to numerical features - ML algorithms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Initialising the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


#Pre-defining the vocabulary size to be 10000, sentence 
vocab_size = 10000
sent_length = 25
embedding_vector_features = 40

#Text preprocessing
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences


#Load the saved models
#Loading Random Forest Classifier
rf_classifier = joblib.load('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/src/models/saved-models-vectorizer/finalized_ht.sav')
#Loading Multinomial Byes Classifier
mnb_classifier = joblib.load('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/src/models/saved-models-vectorizer/finalized_mnb.sav')
#Loading Keras basic RNN model
rnn_base_classifier = keras.models.load_model('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/src/models/saved-models-vectorizer/RNN_basic')
#Loading KeraS basic ANN model
ann_base_classifier = keras.models.load_model('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/src/models/saved-models-vectorizer/NN_basic')
#Loading the vectorizer
tfv = joblib.load('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/src/models/saved-models-vectorizer/finalized_tfv.sav')
#Load the  dataset for performing visualizations

vis_data = pd.read_csv('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/data/processed/processed_data.csv')
vis_turney = pd.read_csv('/content/drive/MyDrive/Datasets/Indian Financial News Headlines/data/processed/turney2.csv')
vis_data['Date'] = pd.to_datetime(vis_data['Date'], infer_datetime_format=True)
vis_data['Year'] = vis_data['Date'].dt.year
vis_data_sorted = vis_data.sort_values(by='Year', ascending=False)
vis_data_sorted.drop(['Unnamed: 0'], axis=1, inplace=True)

#Custom Test Prediction, for checks
def clean_raw(text):
   new_review = str(text)
   new_review = re.sub('[^a-zA-Z]', ' ', new_review)
   new_review = new_review.lower()
   new_review = new_review.split()
   all_stopwords = stopwords.words('english')
   new_review = [lemmatizer.lemmatize(word) for word in new_review]
   new_review = ' '.join(new_review)

   return new_review

def transformer_tf(classifier, text, vectorizer):
   cleaned_corpus = [clean_raw(text)]
   new_X_test = vectorizer.transform(cleaned_corpus).toarray()
   new_y_pred = classifier.predict(new_X_test)

   opinion = " "
   if new_y_pred[0] == 0:
      opinion = "Negative statement or no opinion"
   else:
      opinion = "Positive opinion"

   return opinion

def transformer_oh(text):
   cleaned_text = clean_raw(text)
   oh_encoded_text = [one_hot(cleaned_text, vocab_size)]
   embedded_encoded_text = pad_sequences(oh_encoded_text, padding='post', maxlen=sent_length)
   return embedded_encoded_text

def get_imp(bow, mf, ngram1, ngram2):
   cvt = CountVectorizer(bow, ngram_range=(ngram1, ngram2), max_features=mf,stop_words='english')
   matrix=cvt.fit_transform(bow)
   return pd.Series(np.array(matrix.sum(axis=0))[0], index=cvt.get_feature_names()).sort_values(ascending=False).head(100)

def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = 180
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

def wordcloud_generator(data):
   wordcloud = WordCloud(
                  background_color='white',
                  stopwords=STOPWORDS,
                  max_words=200,
                  max_font_size=40, 
                  random_state=42).generate(str(data['Combined_Text']))
   return wordcloud


   

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(**PAGE_CONFIG)

def main():

   header_text = ''' 
                  <h3 style="text-align:center; text-transform:uppercase; text-decoration:none; letter-spacing:2px;">
                  Sentiment Analysis of Financial News Headlines
                  </h3>

                 '''

   st.markdown(header_text, unsafe_allow_html=True)

   menu = ['Classification','Visualization']
   st.sidebar.subheader("Choose to predict or visualize")
   choice = st.sidebar.selectbox("Click the desired option", menu)

   if choice == 'Classification':

      st.markdown('''
                  <h3 style="text-align:center; text-transform:uppercase; text-decoration:none;>
                  Enter a news headline in the text area corresponding to a stock or company
                  </h3>
                  ''')
      
      menupred = ['Random Forest Classifier', 'Multinomial Naive Byes', 'ANN-basic', 'RNN-basic']
      st.sidebar.subheader("Choose your classifier")
      predchoice = st.sidebar.selectbox("Click the desired option", menupred)

      if predchoice == 'Random Forest Classifier':
         st.success("You have successfully selected the {} classifier".format(predchoice))
         with st.beta_container():
            text = st.text_area('Enter text', height=100)
            button = st.button("Predict")
            classifier = rf_classifier
            
            if button and len(text) != 0:
              pred_text = transformer_tf(classifier, text, tfv)
              st.write(pred_text)

            if len(text) == 0:
              st.write("Please enter some valid text")


      elif predchoice == 'Multinomial Naive Byes':
         st.success("You have successfully selected the {} classifier".format(predchoice))
         with st.beta_container():
            text = st.text_area('Enter text', height=100)
            button = st.button("Predict")
            classifier = mnb_classifier
            
            if button and len(text) != 0:
                pred_text = transformer_tf(classifer, text, tfv)
                st.write(pred_text)

            if len(text) == 0:
                st.write("Please enter some valid text")



      elif predchoice == 'Sequence Model':
         st.success("You have successfully selected the {} classifier".format(predchoice))
         with st.beta_container():
            text = st.text_area('Enter text', height=100)
            button = st.button("Predict")
            classifier = rnn_base_classifier
            oh_encoded_text = transformer_oh(text)

            if button and len(text) != 0:
               pred_label = classifier.predict_classes(oh_encoded_text)


               
               st.write(pred_label)
              #  if int(pred_label) == 1:
              #     st.write("WELL, HAVE FAITH AND ALL WILL BE FINE!")

              #  else:
              #     st.write("WELL, ALL IS NOT WELL OR SEEMINGLY NEUTRAL TO TAKE A STAND")

            if len(text) == 0:
               st.write("PLEASE ENTER SOME VALID TEXT")
            
            else:
               pass



      elif predchoice == 'ANN model':
         st.success("You have successfully selected the {} classifier".format(predchoice))
         with st.beta_container():
            text = st.text_area('Enter text', height=100)
            button = st.button("Predict")
            classifier = ann_base_classifier
            oh_encoded_text = transformer_oh(text)
            
            if button and len(text) != 0:
               pred_label = classifier.predict_classes(oh_encoded_text)



               st.write(pred_label)
              #  if int(pred_label[0]) == 1:
              #     st.write("WELL, HAVE FAITH AND ALL WILL BE FINE!")

              #  else:
              #     st.write("WELL, ALL IS NOT WELL OR SEEMINGLY NEUTRAL TO TAKE A STAND")

            if len(text) == 0:
               st.write("PLEASE ENTER SOME VALID TEXT")
            
            else:
               pass


   elif choice == 'Visualization':
      st.markdown("Here are some interesting and fun visualizations we get on the processed dataset")
      st.markdown(''' 
      
      
                  ''')
      #Display the first 5 rows of the processed dataset

      #Extracting the corpus
      corpus = vis_data['Combined_Text']
      bow = vis_data_sorted['Combined_Text']

      
      #The code for following visualizations - mention the inspirations later

      #Storing important unigrams in 1-year gap
      total_data_unigram = get_imp(bow.tolist(), 5000, ngram1=1, ngram2=1)
      imp_unigrams = {}
      for year in vis_data_sorted['Year'].unique():
         _bow = vis_data_sorted[vis_data_sorted['Year'] == year]['Combined_Text'].tolist()
         imp_unigrams[year] = get_imp(_bow, mf=5000, ngram1=1, ngram2=1)


      #Storing common unigrams in 1-year gap
      com_unigrams = {}
      for year in np.arange(2014, 2020, 1):
        if year == 2020:
            com_unigrams[year] = set(imp_unigrams[year].index).intersection(set(imp_unigrams[year-1].index))

        else:
            com_unigrams[year] = set(imp_unigrams[year].index).intersection(set(imp_unigrams[year+1].index))


      #Storing important bigrams in 1-year gap
      total_data_bigram = get_imp(bow.tolist(), 5000, ngram1=2, ngram2=2)
      imp_bigrams = {}
      for year in vis_data_sorted['Year'].unique():
         _bow = vis_data_sorted[vis_data_sorted['Year'] == year]['Combined_Text'].tolist()
         imp_bigrams[year] = get_imp(_bow, mf=5000, ngram1=2, ngram2=2)


      #Storing common bigrams in 1-year gap
      com_bigrams = {}
      for year in np.arange(2014, 2020, 1):
         if year == 2020:
            com_bigrams[year] = set(imp_bigrams[year].index).intersection(set(imp_bigrams[year-1].index))

         else:
            com_bigrams[year] = set(imp_bigrams[year].index).intersection(set(imp_bigrams[year+1].index))


      #Storing important trigrams in 1-year gap
      total_data_trigram = get_imp(bow.tolist(), 5000, ngram1=3, ngram2=3)
      imp_trigrams = {}
      for year in vis_data_sorted['Year'].unique():
         _bow = vis_data_sorted[vis_data_sorted['Year'] == year]['Combined_Text'].tolist()
         imp_trigrams[year] = get_imp(_bow, mf=5000, ngram1=3, ngram2=3)


      #Storing common trigrams in 1-year gap
      com_trigrams = {}
      for year in np.arange(2014, 2020, 1):
         if year == 2020:
            com_trigrams[year] = set(imp_trigrams[year].index).intersection(set(imp_trigrams[year-1].index))

         else:
            com_trigrams[year] = set(imp_trigrams[year].index).intersection(set(imp_trigrams[year+1].index))
      

      st.markdown('''
      
                  ''')
      st.title('Plot of Unigrams')
      st.bar_chart(total_data_unigram.head(20))

      st.title('Plot of Bigrams')
      st.bar_chart(total_data_bigram.head(20))

      st.title('Plot of Trigrams')
      st.bar_chart(total_data_trigram.head(20))

      st.markdown('''
      
                  ''')
      st.title("Plot of Year-Wise distribution of most frequent n-grams")

      st.markdown('''
      
                  ''')
      

      
      #Initialising the slider
      value = st.sidebar.slider('Choose year for corresponding n-gram visualization', 
                                min_value = 2014, max_value = 2020, value = 2015, step = 1)

      st.markdown('Unigrams for {} year'.format(value))
      st.bar_chart(imp_unigrams[value].head(5))

      st.markdown('Bigrams for {} year'.format(value))
      st.bar_chart(imp_bigrams[value].head(5))

      st.markdown('Trigrams for {} year'.format(value))
      st.bar_chart(imp_trigrams[value].head(5))
      

      #Initialising another dropdown
      st.markdown(
          '''

          ''' 
      )
      st.title('WordCloud Visualization')
      menu_bank = ['Hdfc','Axis', 'RBI', 'Yes']
      value = st.sidebar.selectbox('Word Cloud Visualization', menu_bank)

      if value == 'Hdfc':
         index_hdfc = vis_data_sorted['Combined_Text'].str.match(r'(?=.*\bhdfc\b)(?=.*\bbank\b).*$', case=False)
         data_hdfc = vis_data_sorted.loc[index_hdfc]
         wordcloud = wordcloud_generator(data_hdfc)
         plt.imshow(wordcloud)
         plt.axis('off')
         plt.xticks([])
         plt.yticks([])
         plt.show()
         st.pyplot()

      elif value == 'Axis':
         index_axis = vis_data_sorted['Combined_Text'].str.match(r'(?=.*\bAxis\b)(?=.*\bbank\b).*$', case=False)
         data_axis = vis_data_sorted.loc[index_axis]
         wordcloud = wordcloud_generator(data_axis)
         plt.imshow(wordcloud)
         plt.axis('off')
         plt.xticks([])
         plt.yticks([])
         plt.show()
         st.pyplot()


      elif value == 'RBI':
         index_RBI = vis_data_sorted['Combined_Text'].str.match(r'.*\bRBI\b.*$', case=False)
         data_RBI = vis_data_sorted.loc[index_RBI]
         wordcloud = wordcloud_generator(data_RBI)
         plt.imshow(wordcloud)
         plt.axis('off')
         plt.xticks([])
         plt.yticks([])
         plt.show()
         st.pyplot()


      elif value == 'Yes':
         index_yes = vis_data_sorted['Combined_Text'].str.match(r'(?=.*\bYes\b)(?=.*\bbank\b).*$', case=False)
         data_yes = vis_data_sorted.loc[index_yes]
         wordcloud = wordcloud_generator(data_yes)
         plt.imshow(wordcloud)
         plt.axis('off')
         plt.xticks([])
         plt.yticks([])
         plt.show()
         st.pyplot()


      # col1, col2, col3 = st.beta_columns(3)

      # with col1:
        #  st.header("Unigrams")
        #st.bar_chart(total_data_unigram.head(20))
        #  total_data_unigram.head(20).plot(kind='bar', figsize=(25, 10), colormap='Set1')
        #  plt.xlabel('Unigrams')
        #  plt.ylabel('Frequency')
        #  plt.title('Count of Unigrams in Dataset', fontsize=20)
        #  plt.xticks(size=20)
        #  st.pyplot()

      # with col2:
        #  st.header("Bigrams")
        #st.bar_chart(total_data_unigram.head(20))
        #  total_data_bigram.head(20).plot(kind='bar', figsize=(25, 10), colormap='Set2')
        #  plt.xlabel('Unigrams')
        #  plt.ylabel('Frequency')
        #  plt.title('Count of Unigrams in Dataset', fontsize=20)
        #  plt.xticks(size=20)
        #  st.pyplot()

      # with col3:
        #  st.header("Trigrams")
        #st.bar_chart(total_data_unigram.head(20))
        #  total_data_unigram.head(20).plot(kind='bar', figsize=(25, 10), colormap='Set3')
        #  plt.xlabel('Unigrams')
        #  plt.ylabel('Frequency')
        #  plt.title('Count of Unigrams in Dataset', fontsize=20)
        #  plt.xticks(size=20)
        #  st.pyplot()
      

if __name__ == '__main__':
  main()