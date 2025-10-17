import streamlit as st
from googleapiclient.discovery import build
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import string
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import emoji
import contractions
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import plotly.graph_objects as go
import requests
import time
import streamlit as st

# Load Google Font (Roboto)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    * {
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

API_KEY = "AIzaSyBSOqV5S5EyaGcNzUfGCzzDMeM0OAV5F7c"
#AIzaSyBSOqV5S5EyaGcNzUfGCzzDMeM0OAV5F7c

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        position: relative;
        overflow: hidden;
    }

    /* Rotated background image using pseudo-element */
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background-image: url("https://res.cloudinary.com/hilnmyskv/image/upload/q_auto,f_auto/v1683817866/Algolia_com_Blog_assets/Featured_images/ai/what-is-a-neural-network-and-how-many-types-are-there/fvxz7vm1h0z3ujwdk0or.jpg");
        background-size: cover;
        background-position: center;
        transform: rotate(90deg);
        opacity: 0.3;
        z-index: 0;
    }

    /* Radio container styling */
    [data-testid="stSidebar"] .stRadio {
        position: relative;
        z-index: 1;
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(0,0,0,0.4);
    }

    /* Make radio button text white */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stRadio span {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None
    
# Sidebar content
with st.sidebar:
    add_ratio = st.radio(
        "Features Available",
        (
            "Overview","Your Video", "Sentiment Analysis","Video Search","Overall Sentiment", "Top Positive Comments", "Top Negative Comments",
            "Word Cloud", "Recent Comment Summary", "Comment Search box", "About Project","About Me"
        )
    )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
if add_ratio in ["Overview"]:
    st.markdown("""
    <style>
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-8px); }
    }
    
    .floating-text {
      animation: float 3s ease-in-out infinite;
      display: inline-block;
    }
    
    .center-text {
      text-align: center;
      margin-top: 30px;
    }
    
    h2, h5 {
      margin: 0;
      text-align: center;
    }
    
    /* Outline effect using text-shadow */
    .black-outline {
      color: black;
      text-shadow: 
        -1px -1px 0 #000,  
         1px -1px 0 #000,  
        -1px  1px 0 #000,  
         1px  1px 0 #000;
    }
    
    .white-with-black-outline {
      color: white;
      text-shadow: 
        -0.5px -0.5px 0 #000,  
         0.5px -0.5px 0 #000,  
        -0.5px  0.5px 0 #000,  
         0.5px  0.5px 0 #000;
    }
    </style>
    
    <div class="center-text">
      <h2 class="floating-text black-outline" style="
          font-weight: 900;
          font-size: 3.3em;
          color:white;
      ">
          🔬 SentimentScope
      </h2>
      
      <h5 class="floating-text black-with-black-outline" style="
          font-size: 1.2em;
          font-weight: 500;
          margin-top: 10px;
          color:#b2eee6;
      ">
          <strong>An AI-powered sentiment analysis system for YouTube audience evaluation.</strong>
      </h5>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='display: inline-flex; align-items: center; gap: 10px ;color:white;'>"
                "<span style='font-size: 1.2em;'><strong>🔗 Paste YouTube Video URL here:<strong></span></div>",
                unsafe_allow_html=True)

    st.session_state.video_url = st.text_input(
        label="YouTube URL",
        placeholder="Paste the YouTube video URL here",
        label_visibility="collapsed"
    )
video_url = st.session_state.video_url
video_id = extract_video_id(video_url) if video_url else None
st.session_state.video_id = video_id

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
import base64
import streamlit as st

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ✅ Correct Absolute Path
local_image_path = "/Users/surajkumar/Documents/Data_Science/Sentiment/vecteezy_abstract-blue-background-simple-design-for-your-website_6852804.jpg"
bg_img_base64 = get_base64_of_bin_file(local_image_path)

# ✅ USE f-string HERE
page_bg_img = f'''
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.3); /* 👈 Adjust opacity here */
    z-index: 0;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

stop_words=set(stopwords.words('english'))
lemma=WordNetLemmatizer()
negation_words = {"no", "nor", "not", "ain", "aren't", "couldn't", "didn't", "doesn't","hadn't", "hasn't", "haven't", "isn't", "shouldn't", "wasn't","weren't", "won't", "wouldn't","mightn't","needn't"}
stop_words = stop_words-negation_words
correction_dict = {'bday': 'birthday', 'gr8': 'great', 'luv': 'love', 'ur': 'your', 'pls': 'please', 'thx': 'thanks', 'u': 'you', 'brb': 'be right back', 'idk': 'I don\'t know', 'omg': 'oh my god', 'lol': 'laugh out loud', 'tbh': 'to be honest', 'fyi': 'for your information', 'lmk': 'let me know', 'btw': 'by the way', 'asap': 'as soon as possible', 'smh': 'shaking my head', 'ttyl': 'talk to you later', 'ppl': 'people', 'nvm': 'never mind', 'cya': 'see you', 'rofl': 'rolling on the floor laughing', 'omw': 'on my way', 'wdym': 'what do you mean', 'fomo': 'fear of missing out', 'yolo': 'you only live once', 'lmao': 'laughing my ass off', 'gtg': 'got to go', 'wbu': 'what about you', 'bbl': 'be back later', 'bff': 'best friends forever', 'gm': 'good morning', 'gn': 'good night', 'np': 'no problem', 'gg': 'good game', 'afk': 'away from keyboard', 'yup': 'yes', 'nah': 'no', 'yass': 'yes', 'plz': 'please', 'thru': 'through', 'gr8t': 'great', 'wat': 'what', 'wht': 'what', 'howdy': 'hello', 'g2g': 'got to go', 'l8r': 'later', 'no1': 'no one', 'cuz': 'because', 'bro': 'brother', 'sis': 'sister', 'imho': 'in my humble opinion', 'ftw': 'for the win', 'tmi': 'too much information', 'jmho': 'just my humble opinion', 'tbh': 'to be honest', 'btw': 'by the way', 'jk': 'just kidding', 'afaik': 'as far as I know', 'ik': 'I know', 'wfh': 'work from home', 'lmk': 'let me know', 'swag': 'style, confidence', 'fam': 'family', 'thnx': 'thanks', 'gr8ful': 'grateful', 'wyd': 'what you doing', 'sd': 'social distancing', 'pplz': 'people', 'seeya': 'see you', 'yay': 'yes', 'hbu': 'how about you', 'tho': 'though', 'm8': 'mate', 'gr8ful': 'grateful', 'gimme': 'give me', 'fml': 'f**k my life', 'qik': 'quick', 'realy': 'really', 'yr': 'your', 'wtf': 'what the f**k', 'bffl': 'best friends for life', '2morrow': 'tomorrow', '2nite': 'tonight', 'wth': 'what the hell', 'stfu': 'shut the f**k up', 'ngl': 'not gonna lie', 'tbh': 'to be honest', 'smh': 'shaking my head', 'hbd': 'happy birthday', 'gg': 'good game', 'n00b': 'newbie', 'pmu': 'pissed me off', 'rotfl': 'rolling on the floor laughing', 'sol': 'shout out loud', 'omfg': 'oh my f**king god', 'srsly': 'seriously', 'dunno': 'don\'t know', 'bbl': 'be back later', 'lolz': 'laugh out loud', 'l8': 'late', 'fr': 'for real', 'plz': 'please', 'stoked': 'excited', 'lit': 'awesome', 'noob': 'newbie', 'h8': 'hate', 'xoxo': 'hugs and kisses', 'smh': 'shaking my head', 'yolo': 'you only live once','plz':'please','gn':'good night'}

category_mapping = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "20": "Gaming",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers"
}

#nlp=spacy.load("en_core_web_lg",disable=['ner','parse'])
tqdm.pandas()
def remove_HTML_tags(text):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', text)
def lowercasing(text):
    return text.lower()
def remove_URL(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(pattern, '', text)
def remove_punc(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))
def demojify(text):
    return emoji.demojize(text)
def expand_contractions(text):
    return contractions.fix(text)
def remove_stopwords(text):
    list_words = [word for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(list_words)
def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()
def lemmatization(text):
    l=[lemma.lemmatize(word) for word in word_tokenize(text)]
    return ' '.join(l)
def replace_slang_in_review(text):
    words = word_tokenize(text)
    corrected_text = [correction_dict[word] if word in correction_dict else word for word in words]
    return ' '.join(corrected_text)
tqdm.pandas()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def preprocessing(input_text):
  input_text=lowercasing(input_text)
  input_text=remove_extra_whitespaces(input_text)
  input_text=remove_HTML_tags(input_text)
  input_text=remove_URL(input_text)
  input_text=remove_punc(input_text)
  input_text=expand_contractions(input_text)
  input_text=replace_slang_in_review(input_text)
  input_text=demojify(input_text)
  input_text=remove_stopwords(input_text)
  input_text=lemmatization(input_text)
  return input_text

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("/Users/surajkumar/Documents/Data_Science/Sentiment/sentiment_bilstm_model.h5")

@st.cache_resource
def load_tokenizer():
    import pickle
    with open("/Users/surajkumar/Documents/Data_Science/Sentiment//tokenizer.pkl", 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define your class labels (update as per your own dataset)
label_classes = ['Negative', 'Positive']  # Modify if more than 2 classes

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_wordcloud_excluding_oov(text_list, tokenizer):
    # Get tokenizer vocab
    st.markdown(
        "<p style='color:white; font-size:24px; font-family:Roboto,sans-serif; font-weight:bold;'>Word Cloud :</p>",
        unsafe_allow_html=True
    )
    vocab = set(tokenizer.word_index.keys())
    
    all_words = []
    
    for text in text_list:
        clean_text = preprocessing(text)
        tokens = clean_text.split()
        # Filter out OOV tokens (not in vocab)
        filtered_tokens = [word for word in tokens if word in vocab]
        all_words.extend(filtered_tokens)

    # Join all filtered words
    final_text = " ".join(all_words)

    # Generate WordCloud with optimized parameters for speed
    wordcloud = WordCloud(
        width=1200,         # smaller width for faster generation
        height=800,         # smaller height
        max_words=200,      # limit number of words
        background_color='white',
        colormap='viridis',
        random_state=42     # ensures reproducibility
    ).generate(final_text)

    # Plot and display in Streamlit
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)  # smaller fig size for speed
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    # Center in Streamlit using columns
    col1, col2, col3 = st.columns([1, 25, 1])
    with col2:
        st.pyplot(fig)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
            
def get_comments_with_time(video_id, api_key=API_KEY, target_count=20000):
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key,
        'textFormat': 'plainText',
        'maxResults': 100,
        'fields': 'items(snippet/topLevelComment/snippet(textDisplay,publishedAt)),nextPageToken'
    }

    page = 0
    while len(comments) < target_count:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()

        for item in data.get("items", []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comment_text = snippet['textDisplay']
            published_time = snippet['publishedAt']
            comments.append((comment_text, published_time))  # now a tuple

            if len(comments) >= target_count:
                break

        if 'nextPageToken' in data:
            params['pageToken'] = data['nextPageToken']
        else:
            print("No more comments available.")
            break

        page += 1
        print(f"Fetched: {len(comments)} comments")
        time.sleep(0.1)  # Delay to prevent quota spikes

    return comments

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def batch_predict_sentiment(text_list, model, tokenizer, max_len=208, label_classes=['Negative', 'Positive']):
    # Step 1: Extract and preprocess only the text
    comments = [preprocessing(text) for text, _ in text_list]
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    valid_indices = []
    valid_padded = []

    for i, seq in enumerate(padded):
        seq = np.array(seq)
        zero_count = np.sum(seq == 0)
        oov_count = np.sum(seq == 1)
        rem = max_len - zero_count

        # Skip if empty or too many OOVs
        if rem == 0 or oov_count / rem > 0.6:
            continue

        valid_indices.append(i)
        valid_padded.append(seq)

    if not valid_padded:
        return []  # Nothing valid to predict

    # Predict in batch
    valid_padded = np.array(valid_padded)
    pred_probs = model.predict(valid_padded).flatten()
    pred_classes = [1 if prob >= 0.43 else 0 for prob in pred_probs]
    pred_labels = [label_classes[cls] for cls in pred_classes]

    # Step 2: Include publish time
    results = [
        (text_list[i][0], label, round(prob * 100, 2), text_list[i][1])
        for i, label, prob in zip(valid_indices, pred_labels, pred_probs)
    ]

    return results
@st.cache_data
def batch_predict_once(text_list):
    return batch_predict_sentiment(text_list, model, tokenizer)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

from datetime import datetime, timedelta
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go

def recent_comment_summary():
    # Sort comments by publish_time in IST
    sorted_recent = sorted(
        list_of_predicted_labels,
        key=lambda x: datetime.strptime(x[3], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=5, minutes=30),
        reverse=True
    )

    # Pick most recent 250 comments
    recent_250 = sorted_recent[:250]

    # Count sentiments
    recent_pos = sum(1 for _, label, *_ in recent_250 if label == "Positive")
    recent_neg = sum(1 for _, label, *_ in recent_250 if label == "Negative")

    # --- Pie Chart ---
    labels = ['Positive', 'Negative']
    values = [recent_pos, recent_neg]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(
            line=dict(color='black', width=2)  # border of 2px
        )
    )])
    fig.update_layout(
        title=dict(
            text="Sentiment Distribution of Most Recent 250 Comments",
            font=dict(
                size=17,
                color="white",
                family="Roboto,sans-serif"
            )
        ),
        legend=dict(
            font=dict(
                family="Roboto,sans-serif",
                size=16,
                color="white"
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Subheader with color #b2eee6
    st.markdown("<h3 style='color:#b2eee6;'>📊📈 Recent Comment Summary (Last 250)</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Show 10 Most Recent Comments ---
    st.markdown("<h3 style='color:#b2eee6;'>💬 10 Most Recent Comments</h3>", unsafe_allow_html=True)
    for i, (comment, label, score, publish_time) in enumerate(sorted_recent[:10], start=1):
        # Convert to IST for display
        ist_time = datetime.strptime(publish_time, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=5, minutes=30)

        st.markdown(f"""
        <p style='color:white; font-size:16px;'>
        <strong>{i}.</strong> {convert_back_to_emoji(comment)} <br>
        ➤ Sentiment: {'🟢 Positive' if label == 'Positive' else '🔴 Negative'} <br>
        ➤ Score: {score:.2f} <br>
        ➤ Published At: {ist_time.strftime("%Y-%m-%d %I:%M %p")} IST
        </p>
        """, unsafe_allow_html=True)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def search_comments(list_of_predicted_labels):
    """
    Displays a Streamlit text input for comment search and shows matching results.

    Parameters:
    - list_of_predicted_labels: list of (comment, sentiment, score, publish_time)
    """

    # Step 1: Create DataFrame
    df = pd.DataFrame(list_of_predicted_labels, columns=["comment", "sentiment", "score", "publish_time"])

    # Convert publish_time to IST (manually add 5 hours 30 minutes)
    def convert_to_ist(utc_str):
        try:
            utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
            ist_dt = utc_dt + timedelta(hours=5, minutes=30)
            return ist_dt.strftime("%Y-%m-%d %I:%M %p") + " IST"
        except:
            return utc_str  # fallback if format breaks

    df["publish_time_ist"] = df["publish_time"].apply(convert_to_ist)

    # Step 2: Input search term
    st.markdown("""
        <style>
        .stTextInput>label {
            color: white;
            font-size: 22px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # --- Text input ---
    search_term = st.text_input("🔍 Search Comments", placeholder="Type a keyword or phrase...")

    # Step 3: If term entered, filter results
    if search_term:
        filtered_df = df[df['comment'].str.contains(search_term, case=False, na=False)]

        st.markdown(f'<h3 style="color:#b2eee6;">🔎 Search Results for: *{search_term}*</h3>', unsafe_allow_html=True)
        
        if filtered_df.empty:
            st.warning("No matching comments found.")
        else:
            for i, row in filtered_df.iterrows():
                st.markdown(f"""
                <div style="font-size: 0.85rem; color:white;">
                    <b style="color:#b2eee6;">🗨️ Comment:</b> {row['comment']}<br>
                    <b style="color:#b2eee6;">🏷️ Sentiment:</b> {row['sentiment']}<br>
                    <b style="color:#b2eee6;">📊 Score:</b> {row['score']:.2f}<br>
                    <b style="color:#b2eee6;">🕒 Published:</b> {row['publish_time_ist']}
                    <hr style="margin: 8px 0; border: 1px solid #00FFFF;">
                </div>
                """, unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------
def get_top_videos(query, api_key=API_KEY, max_results=10):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",           # request only videos
        "maxResults": max_results,
        "key": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    videos = []
    for item in data.get("items", []):
        # Safely check before accessing videoId
        video_id = item.get("id", {}).get("videoId")
        title = item["snippet"]["title"]

        if video_id:  # only add if videoId exists
            videos.append((video_id, title))

    return videos

#---------------------------------------------------------------------------------------------------------------------------------------------------
def get_video_stats(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if "items" in data and len(data["items"]) > 0:
        video_info = data["items"][0]
        title = video_info["snippet"]["title"]
        stats = video_info["statistics"]
        likes = stats.get("likeCount", "N/A")
        views = stats.get("viewCount", "N/A")
        comments = stats.get("commentCount", "N/A")

        return {
            "title": title,
            "likes": likes,
            "views": views,
            "comments": comments
        }
    else:
        return "Invalid Video ID or API Limit Reached!"
        
#---------------------------------------------------------------------------------------------------------------------------------------------------
def overall_sentiment(video_stats):
    total_pos = 0
    total_neg = 0
    total_comments = 0
    
    for video in video_stats:
        total_pos += video["pos"]
        total_neg += video["neg"]
        total_comments += video["total"]
    
    if total_comments == 0:
        return 0, 0  # Avoid division by zero
    
    overall_pos = (total_pos / total_comments) * 100
    overall_neg = (total_neg / total_comments) * 100
    
    return round(overall_pos, 2), round(overall_neg, 2)

#---------------------------------------------------------------------------------------------------------------------------------------------------
def category(API_KEY,video_ID):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract categoryId
    category_id = data['items'][0]['snippet']['categoryId']
    return category_id

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def convert_back_to_emoji(sentence):

    # Remove spaces around colons
    sentence_fixed = re.sub(r":\s*(\w+)\s*:", r":\1:", sentence)
    
    # Convert emoji names to actual emojis
    converted = emoji.emojize(sentence_fixed, language='alias')
    
    return converted
#--------------------------------------------------------------------------------------------------------------------------------------------------
def plot_stacked_bar(data_list):
    x = list(range(1, 11))
    pos = [d["pos"] for d in data_list]
    neg = [d["neg"] for d in data_list]
    
    # Create stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=pos,
        name='Positive',
        marker_color='#b2eee6'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=neg,
        name='Negative',
        marker_color='#ffc0a9'
    ))
    
    # Update layout for stacked bars with transparent background
    fig.update_layout(
        barmode='stack',
        title=dict(
            text='Stacked Bar Chart of Sentiments',
            font=dict(color='white', size=20)  # Title color and size
        ),
        xaxis=dict(
            title=dict(text='Index', font=dict(color='white', size=16)),  # X-axis label white
            tickfont=dict(color='white')  # X-axis tick labels white
        ),
        yaxis=dict(
            title=dict(text='Count', font=dict(color='white', size=16)),  # Y-axis label white
            tickfont=dict(color='white')  # Y-axis tick labels white
        ),
        plot_bgcolor='rgba(0,0,0,0)',   # transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # transparent paper background
        font=dict(color='white'),       # default font color
        legend=dict(
            font=dict(
                family="Roboto,sans-serif",  # Bold + Italic
                size=16,
                color="white"  # Legend text color
            )
        )
    )

    
    # Show figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)



#--------------------------------------------------------------------------------------------------------------------------------------------------

def about_project_section():
    st.markdown("""
    <h2 style="color:#b2eee6;">📘 About the Project</h2>
    <p style="color:white;">This project performs <b>Real-Time Sentiment Analysis</b> on YouTube comments using a <b>BiLSTM deep learning model</b>.</p>

    <h3 style="color:#b2eee6;">🔍 Features:</h3>
    <p style="color:white;">🎯 Predicts Positive/Negative sentiment<br>
    📊 Pie chart and timeline of sentiment trends<br>
    🏷️ Top 10 confident comments (Positive/Negative)<br>
    🔎 Keyword-based comment search<br>
    ⏱️ Recent comment summary (last 250)</p>

    <h3 style="color:#b2eee6;">🧠 Model:</h3>
    <p style="color:white;">Trained BiLSTM using TensorFlow/Keras<br>
    Custom thresholding, padding & OOV filtering<br>
    Uses saved tokenizer for inference</p>

    <h3 style="color:#b2eee6;">⚙️ Tech Stack:</h3>
    <p style="color:white;">TensorFlow · Streamlit · Matplotlib · Pandas</p>

    <hr style="border: 1px solid #00FFFF; margin-top:30px; margin-bottom:30px;">

    <p style="color:white;">👨‍💻 Built with ❤️ by <b>Suraj Kumar</b></p>
    """, unsafe_allow_html=True)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def about_me_section():
    # --- Title and Greeting ---
    st.markdown("""
    <div style="text-align:center; margin-top:30px;">
        <h1 style="
            display: inline-block;
            padding: 10px 25px;
            font-size:3em;
            font-weight:900;
            background: linear-gradient(90deg, #00FFFF, #FF00FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-color: white;   /* White background */
            border-radius: 25px;       /* Rounded corners */
            box-shadow: 2px 2px 15px rgba(0,0,0,0.3); /* subtle shadow */
        ">
            👋 Hi, I'm Suraj Kumar
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # --- Social Links with CSS hover effect ---
    st.markdown("""
    <style>
        .social-link img {
            transition: transform 0.3s;
            border-radius: 10px;
        }
        .social-link img:hover {
            transform: scale(1.2);
        }
    </style>
    <div style="text-align:center; margin-top:20px;">
        <a href="https://www.linkedin.com/in/suraj-kumar-a22608260/" class="social-link">
            <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
        </a>
        &nbsp;
        <a href="https://github.com/surajkumar4117" class="social-link">
            <img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github" alt="GitHub">
        </a>
    </div>
    """, unsafe_allow_html=True)

    # --- Divider line ---
    st.markdown("""
    <hr style="border: 1px solid #00FFFF; margin-top:30px; margin-bottom:30px;">
    """, unsafe_allow_html=True)


#--------------------------------------------------------------------------------------------------------------------------------------------------
if 'videos_data' not in st.session_state:
    st.session_state.videos_data = []
#--------------------------------------------------------------------------------------------------------------------------------------------------

if (add_ratio=="About Me"):
    about_me_section()
if (add_ratio=="About Project"):
    about_project_section()
if (add_ratio=="Video Search"):
    st.markdown("""
        <style>
        .stTextInput>label {
            color: white !important;
            font-size: 18px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    search_term = st.text_input("🔍 Search", placeholder="Type a keyword or phrase...")
    vdo=get_top_videos(search_term,API_KEY);
    
    vdoID=[]
    vdoTitle=[]
    
    
    for i in range(len(vdo)):
        vdoID.append(vdo[i][0])
        vdoTitle.append(vdo[i][1])
    for i in range(len(vdo)):
        vdo_stats={}
        video_id=vdoID[i]
        result = get_video_stats(video_id)
        st.markdown(f'<h3 style="color:#b2eee6;">{vdoTitle[i]}</h3>', unsafe_allow_html=True)

        if 'all_comments' not in st.session_state or st.session_state.video_id != video_id:
            progress = st.progress(0)
            status_text = st.empty()
            st.session_state.video_id = video_id
            st.session_state.p = 0
            st.session_state.n = 0

            status_text.text("📥 Fetching comments...")
            all_comments = get_comments_with_time(video_id)
            st.session_state.all_comments = all_comments
            progress.progress(50)
            st.toast("✅ All Comment Fetched!")

            status_text.text("🧹 Preprocessing comments...")
            all_comments_prepro = [(preprocessing(i),j) for i,j in all_comments]
            progress.progress(63)
            status_text.text("🧠 Predicting sentiment...")
            st.session_state.list_of_predicted_labels = batch_predict_sentiment(all_comments_prepro, model, tokenizer)
            progress.progress(75)

            status_text.text("📊 Counting results...")
            for i in st.session_state.list_of_predicted_labels:
                if i[1] == 'Positive':
                    st.session_state.p += 1
                else:
                    st.session_state.n += 1
            progress.progress(100)
            status_text.text("✅ Done!")
    
        # Pull from cache
        p = st.session_state.p
        n = st.session_state.n
        
        vdo_stats["total"]=p+n
        vdo_stats["pos"]=p
        vdo_stats["neg"]=n
        st.session_state.videos_data.append(vdo_stats)
        #st.markdown(videos_data)
        list_of_predicted_labels = st.session_state.list_of_predicted_labels
            
        
            
        if (p+n==0):
            st.warning("No Comments Found")
        else:
    
            # Labels and corresponding values
            labels = ['Positive', 'Negative']
            values = [p, n]
            colors = ['00FF00', '#FF0000']  # Green and Red
            
            # Create Pie chart with custom colors and black border
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(
                    colors=colors,
                    line=dict(color='black', width=2.2)  # Black border around each slice
                ),
            )])
            
            # Set background to transparent or match Streamlit background
            fig.update_layout(
                legend=dict(
                    font=dict(
                        family="Roboto,sans-serif",  # Bold + Italic
                        size=16,
                        color="white"  # Legend text color
                    )
                ),
                title=dict(
                    text='Sentiment Distribution',
                    font=dict(
                        family="Roboto,sans-serif" ,
                        size=20,
                        color='#FFFDFA',
                        
                        
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                #legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            result = get_video_stats(video_id)
            if result:
                likes = result.get('likes', 'N/A')
                views = result.get('views', 'N/A')
                comments = result.get('comments', 'N/A')
                
                st.markdown(f"<p style='color:white; font-weight:bold;'>Likes: {likes} | Views: {views} | Comments: {comments}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:white;'>Data not available</p>", unsafe_allow_html=True)
        
            # Video category
            category_id = category(API_KEY, video_id)
            st.markdown(f"<p style='color:white; font-weight:bold;'>Category: {category_mapping.get(category_id, 'Unknown')}</p>", unsafe_allow_html=True)

    
if (add_ratio=="Overall Sentiment"):
    plot_stacked_bar(st.session_state.videos_data)
    overall_pos, overall_neg = overall_sentiment(st.session_state.videos_data)
    
    if (overall_pos + overall_neg == 0 and len(st.session_state.videos_data) != 0):
        st.markdown("No comments found")
    else:
        labels = ['Positive', 'Negative']
        values = [overall_pos, overall_neg]
        colors = ['00FF00', '#FF0000']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color='black', width=2.2))
        )])

        fig.update_layout(
            legend=dict(
                font=dict(
                    family="Roboto,sans-serif",  # Bold + Italic
                    size=16,
                    color="white"  # Legend text color
                )
            ),
            title=dict(
                text='Overall Sentiment Distribution',
                font=dict(
                    family="Roboto,sans-serif" ,
                    size=20,
                    color='#FFFDFA',
                    
                    
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #legend=dict(bgcolor='rgba(0,0,0,0)')
        )

        st.plotly_chart(fig, use_container_width=True)
            

if video_id:
    def is_valid_video_id(video_id):
        try:
            youtube = build('youtube', 'v3', developerKey=API_KEY)
            request = youtube.videos().list(part='id', id=video_id)
            response = request.execute()
            return len(response['items']) > 0
        except Exception as e:
            return False  # Fail-safe
    
    is_valid=is_valid_video_id(video_id)
    
    if (is_valid):
        
        if 'all_comments' not in st.session_state or st.session_state.video_id != video_id:
            progress = st.progress(0)
            status_text = st.empty()
            st.session_state.video_id = video_id
            st.session_state.p = 0
            st.session_state.n = 0

            status_text.text("📥 Fetching comments...")
            all_comments = get_comments_with_time(video_id)
            st.session_state.all_comments = all_comments
            progress.progress(50)
            st.toast("✅ All Comment Fetched!")

            status_text.text("🧹 Preprocessing comments...")
            all_comments_prepro = [(preprocessing(i),j) for i,j in all_comments]
            progress.progress(63)
            status_text.text("🧠 Predicting sentiment...")
            st.session_state.list_of_predicted_labels = batch_predict_sentiment(all_comments_prepro, model, tokenizer)
            progress.progress(75)

            status_text.text("📊 Counting results...")
            for i in st.session_state.list_of_predicted_labels:
                if i[1] == 'Positive':
                    st.session_state.p += 1
                else:
                    st.session_state.n += 1
            progress.progress(100)
            status_text.text("✅ Done!")
    
        # Pull from cache
        p = st.session_state.p
        n = st.session_state.n
        list_of_predicted_labels = st.session_state.list_of_predicted_labels

        
        if add_ratio == "Your Video":
            # Embed YouTube video with rounded corners and white border
            st.markdown(f"""
            <div style="
                border: 4px solid white; 
                border-radius: 15px; 
                overflow: hidden; 
                width: 100%; 
                max-width: 720px;
                margin-bottom: 10px;
            ">
                <iframe 
                    width="100%" 
                    height="405" 
                    src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" 
                    allowfullscreen
                ></iframe>
            </div>
            """, unsafe_allow_html=True)
        
            # Get video stats
            result = get_video_stats(video_id)
            if result:
                likes = result.get('likes', 'N/A')
                views = result.get('views', 'N/A')
                comments = result.get('comments', 'N/A')
                
                st.markdown(f"<p style='color:white; font-weight:bold;'>Likes: {likes} | Views: {views} | Comments: {comments}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:white;'>Data not available</p>", unsafe_allow_html=True)
        
            # Video category
            category_id = category(API_KEY, video_id)
            st.markdown(f"<p style='color:white; font-weight:bold;'>Category: {category_mapping.get(category_id, 'Unknown')}</p>", unsafe_allow_html=True)
                    
        if (add_ratio=="Sentiment Analysis"):
            
            if (p+n==0):
                st.warning("No comments found")
            else:
        
                # Labels and corresponding values
                labels = ['Positive', 'Negative']
                values = [p, n]
                colors = ['00FF00', '#FF0000']  # Green and Red
                
                # Create Pie chart with custom colors and black border
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    marker=dict(
                        colors=colors,
                        line=dict(color='black', width=2.2)  # Black border around each slice
                    ),
                )])
                
                # Set background to transparent or match Streamlit background
                fig.update_layout(
                    legend=dict(
                        font=dict(
                            family="Roboto,sans-serif",  # Bold + Italic
                            size=16,
                            color="white"  # Legend text color
                        )
                    ),
                    title=dict(
                        text='Sentiment Distribution',
                        font=dict(
                            family="Roboto,sans-serif" ,
                            size=30,
                            color='#FFFDFA',
                            
                            
                        )
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    #legend=dict(bgcolor='rgba(0,0,0,0)')
                )
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
        if (add_ratio=="Word Cloud"):
            if (p+n==0):
                st.markdown("No comments found")
            else:
            
                generate_wordcloud_excluding_oov([x[0] for x in list_of_predicted_labels], tokenizer)

        if add_ratio == "Top Positive Comments":
            positive_comments = [item for item in list_of_predicted_labels if item[1] == 'Positive']
        
            # Sort them by score in descending order
            top_positive = sorted(positive_comments, key=lambda x: x[2], reverse=True)
            
            # Take the top 10
            top_10_positive = top_positive[:10]
            
            # Display subheader with custom color
            st.markdown("<h3 style='color:#b2eee6;'>Top 10 Positive Comments</h3>", unsafe_allow_html=True)
            
            # Display comments in white
            for i, (comment, label, score, publish_time) in enumerate(top_10_positive, start=1):
                st.markdown(f"<p style='color:white; font-size:16px;'>"
                            f"<strong>{i}.</strong> {convert_back_to_emoji(comment)} <br>🟢 <em>Score:</em> {score:.3f}"
                            f"</p>", unsafe_allow_html=True)

        if add_ratio == "Top Negative Comments":
            negative_comments = [item for item in list_of_predicted_labels if item[1] == 'Negative']
        
            # Sort them by score in descending order
            top_negative = sorted(negative_comments, key=lambda x: x[2], reverse=True)
            
            # Take the top 10
            top_10_negative = top_negative[:10]
            
            # Display subheader with custom color
            st.markdown("<h3 style='color:#b2eee6;'>Top 10 Negative Comments</h3>", unsafe_allow_html=True)
            
            # Display comments in white
            for i, (comment, label, score, publish_time) in enumerate(top_10_negative, start=1):
                st.markdown(
                    f"<p style='color:white; font-size:16px;'>"
                    f"<strong>{i}.</strong> {convert_back_to_emoji(comment)} <br>🔴 <em>Score:</em> {score:.3f}"
                    f"</p>", 
                    unsafe_allow_html=True
                )

        
        if (add_ratio=="Recent Comment Summary"):
            recent_comment_summary()

        if (add_ratio=="Comment Search box"):
            search_comments(list_of_predicted_labels)
        
    else:
        st.markdown("Not a valid video ID")






