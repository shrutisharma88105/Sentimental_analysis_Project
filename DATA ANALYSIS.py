#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import re

def extract_text(url):
	textss=[]
	webpage = requests.get(url)
	soup_title = BeautifulSoup(webpage.content, 'html.parser')
	soup=BeautifulSoup(webpage.text, 'lxml')
	url_title = soup_title.title.get_text().split('|')[0].strip()
	paras = soup.find_all('p')
	for i in range(16,len(paras)-3):
		textss.append(paras[i].get_text())
	url_text = ' '.join(textss)
	return url_title, url_text

def extract_list_from_text_file(filepath):
	final_list=[]
	f = open(filepath, "r")
	list_temp=f.read().splitlines()
	for x in list_temp:
		final_list.extend(x.split('|'))
	return final_list

def do_lower(word_list):
    lower_list=[]
    for word in word_list:
        lower_list.append(word.lower())
    return lower_list

def count_sentences(paragraph):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
    return len(sentences)



df=pd.read_excel('Output Data Structure.xlsx')

stop_words_auditor=extract_list_from_text_file('StopWords_Auditor.txt')
stop_words_currencies=extract_list_from_text_file('StopWords_Currencies.txt')
stop_words_datesandnumbers=extract_list_from_text_file('StopWords_DatesandNumbers.txt')
stop_words_generic=extract_list_from_text_file('StopWords_Generic.txt')
stop_words_genericlong=extract_list_from_text_file('StopWords_GenericLong.txt')
stop_words_geographic=extract_list_from_text_file('StopWords_Geographic.txt')
stop_words_names=extract_list_from_text_file('StopWords_Names.txt')

stop_words=stop_words_auditor+stop_words_currencies+stop_words_datesandnumbers+stop_words_generic+stop_words_geographic+stop_words_names

positive_words=extract_list_from_text_file('positive-words.txt')
negative_words=extract_list_from_text_file('negative-words.txt')

# lowers the words of each list
stop_words = do_lower(stop_words)
positive_words = do_lower(positive_words)
negative_words = do_lower(negative_words)

vowels='AaEeIiOoUu'
personal_pronouns=['I', 'we','my','ours','us']

print('Iterating dataframe and calculating all variable')
for i, row in df.iterrows():
    print('processing index number: ',i)

#     webscraping 
    title,text=extract_text(row['URL'])
    
    words_count_cleaned=0
    positive_score=0
    negative_score=0
    complex_words_count=0
    total_syllable_count=0
    personal_pronouns_count=0
    no_of_characters=0
    
    no_of_sentences= count_sentences(title+'. '+text)
    
    for token in nltk.tokenize.word_tokenize(title+'. '+text, language='english', preserve_line=False):
        vowel_count=0
        
        if token.lower() in stop_words:
            continue
        if not token.lower().isalnum():
            continue
        
        words_count_cleaned+=1
        
        for char in token.lower():
            no_of_characters+=1
            if char in vowels:
                vowel_count+=1
        total_syllable_count+=vowel_count
        
        if vowel_count<=2 or token.lower().endswith('es') or token.lower().endswith('ed'):
            complex_words_count+=1
        if token.lower() in positive_words:
            positive_score+=1
        elif token.lower() in negative_words:
            negative_score+=1
        if token in personal_pronouns:
            personal_pronouns_count+=1
        
        
    df.loc[i,'POSITIVE SCORE']=positive_score
    df.loc[i,'NEGATIVE SCORE']=negative_score
    
    df.loc[i,'POLARITY SCORE']=(positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)
    
    df.loc[i,'SUBJECTIVITY SCORE']=(positive_score + negative_score)/ ((words_count_cleaned) + 0.000001)
    
    df.loc[i,'AVG SENTENCE LENGTH'] = words_count_cleaned/no_of_sentences
    
    df.loc[i,'PERCENTAGE OF COMPLEX WORDS']=complex_words_count/words_count_cleaned

    df.loc[i,'FOG INDEX']=0.4 * (df.loc[i,'AVG SENTENCE LENGTH'] + df.loc[i,'PERCENTAGE OF COMPLEX WORDS'])
    
    df.loc[i,'AVG NUMBER OF WORDS PER SENTENCE'] = words_count_cleaned/no_of_sentences
    
    df.loc[i,'COMPLEX WORD COUNT'] = complex_words_count
    
    df.loc[i,'WORD COUNT'] = words_count_cleaned
    
    df.loc[i,'SYLLABLE PER WORD'] = total_syllable_count/words_count_cleaned
    
    df.loc[i,'PERSONAL PRONOUNS'] = personal_pronouns_count
    
    df.loc[i,'AVG WORD LENGTH'] = no_of_characters/words_count_cleaned
    
df.to_csv('Output Data Structure.csv', index=False)

