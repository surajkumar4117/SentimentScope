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

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt')
nltk.download('punkt_tab')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

API_KEY = st.secrets["YOUTUBE_API_KEY"]

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
        -0.5px -0.5px 0 #000,  
         0.5px -0.5px 0 #000,  
        -0.5px  0.5px 0 #000,  
         0.5px  0.5px 0 #000;
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
          font-size: 3em;
      ">
          🎯 SentimentScope
      </h2>
      
      <h5 class="floating-text black-with-black-outline" style="
          font-size: 1.2em;
          font-weight: 500;
          margin-top: 10px;
      ">
          Explore public sentiment on YouTube videos at a glance.
      </h5>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='display: inline-flex; align-items: center; gap: 10px;'>"
                "<span style='font-size: 1.2em;'>📺 Enter YouTube Video URL</span></div>",
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

page_bg_img = '''
<style>
.stApp {
background-image:url("https://media.istockphoto.com/id/1841743797/vector/empty-and-blank-blotched-messy-pastel-cream-or-beige-coloured-grunge-textured-horizontal-old.jpg?s=612x612&w=0&k=20&c=x3nkQnZxrZZahlnGaF8CIdKJWMR9Qf_WXGT8Jl_Gilc=");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}
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
    return load_model(r"sentiment_bilstm_model.h5")

@st.cache_resource
def load_tokenizer():
    import pickle
    with open(r"tokenizer.pkl", 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define your class labels (update as per your own dataset)
label_classes = ['Negative', 'Positive']  # Modify if more than 2 classes

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_wordcloud_excluding_oov(text_list, tokenizer):
            # Get tokenizer vocab
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
    
        # Generate WordCloud
        wordcloud = WordCloud(width=2400, height=1600, background_color='white', colormap='viridis').generate(final_text)
    
        # Plot and display in Streamlit
        fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        
        col1, col2, col3 = st.columns([1, 6, 1])  # Adjust width ratios
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

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(
        title_text="Sentiment Distribution of Most Recent 250 Comments",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.subheader("📊 Recent Comment Summary (Last 250)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Show 10 Most Recent Comments ---
    st.subheader("🕒 10 Most Recent Comments")
    for i, (comment, label, score, publish_time) in enumerate(sorted_recent[:10], start=1):
        # Convert to IST for display
        ist_time = datetime.strptime(publish_time, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=5, minutes=30)

        st.markdown(f"""
        **{i}.** {comment}  
        ➤ Sentiment: `{'🟢 Positive' if label == 'Positive' else '🔴 Negative'}`  
        ➤ Score: `{score:.2f}`  
        ➤ Published At: `{ist_time.strftime("%Y-%m-%d %I:%M %p")} IST`
        """)

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
        except Exception as e:
            return utc_str  # fallback if format breaks

    df["publish_time_ist"] = df["publish_time"].apply(convert_to_ist)

    # Step 2: Input search term
    search_term = st.text_input("🔍 Search Comments", placeholder="Type a keyword or phrase...")

    # Step 3: If term entered, filter results
    if search_term:
        filtered_df = df[df['comment'].str.contains(search_term, case=False, na=False)]

        st.markdown(f"### 🔎 Search Results for: *{search_term}*")
        if filtered_df.empty:
            st.warning("No matching comments found.")
        else:
            for i, row in filtered_df.iterrows():
                st.markdown(f"""
                <div style="font-size: 0.85rem;">
                <b>🗨️ Comment:</b> {row['comment']}<br>
                🏷️ <b>Sentiment:</b> <code>{row['sentiment']}</code><br>
                📊 <b>Score:</b> <code>{row['score']:.2f}</code><br>
                🕒 <b>Published:</b> {row['publish_time_ist']}
                <hr style="margin: 8px 0;">
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

def about_project_section():
    st.markdown("## 📘 About the Project")

    st.markdown("""
    This project performs **Real-Time Sentiment Analysis** on YouTube comments using a **BiLSTM deep learning model**.

    ### 🔍 Features:
    - 🎯 Predicts Positive/Negative sentiment  
    - 📊 Pie chart and timeline of sentiment trends  
    - 🏷️ Top 10 confident comments (Positive/Negative)  
    - 🔎 Keyword-based comment search  
    - ⏱️ Recent comment summary (last 250)  
    
    ### 🧠 Model:
    - Trained BiLSTM using TensorFlow/Keras  
    - Custom thresholding, padding & OOV filtering  
    - Uses saved tokenizer for inference  
    
    ### ⚙️ Tech Stack:
    - TensorFlow · Streamlit · Matplotlib · Pandas

    ---
    👨‍💻 Built with ❤️ by **Suraj Kumar**
    """)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def about_me_section():
    

    # --- Title and Greeting ---
    st.markdown("""
        ## 👋 Hi, I'm **Suraj Kumar**
        <h6 style='color:red; font-weight:normal;'>🚀 Passionate about Deep Learning, NLP, and Building Real-Time ML Applications</h6>
    """, unsafe_allow_html=True)

    # --- Social Links ---
    st.markdown("""
    <div style="margin-top: 10px;">
        <a href="https://www.linkedin.com/in/suraj-kumar-a22608260/">
            <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
        </a>
        &nbsp;
        <a href="https://github.com/surajkumar4117">
            <img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github" alt="GitHub">
        </a>
    </div>
    """, unsafe_allow_html=True)


#---------------------------------------------------------------------------------------------------------------------------------------------------
if 'videos_data' not in st.session_state:
    st.session_state.videos_data = []
#---------------------------------------------------------------------------------------------------------------------------------------------------


if (add_ratio=="About Me"):
    about_me_section()
if (add_ratio=="About Project"):
    about_project_section()
if (add_ratio=="Video Search"):
    
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
        st.subheader(vdoTitle[i])

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
            st.markdown("No comments found")
        else:
    
            # Labels and corresponding values
            labels = ['Positive', 'Negative']
            values = [p, n]
            colors = ['#2ecc71', '#e74c3c']  # Green and Red
            
            # Create Pie chart with custom colors and black border
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(
                    colors=colors,
                    line=dict(color='black', width=2)  # Black border around each slice
                )
            )])
            
            # Set background to transparent or match Streamlit background
            fig.update_layout(
                title_text='Sentiment Distribution',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            if result:
                likes = result.get('likes', 'N/A')
                views = result.get('views', 'N/A')
                comments = result.get('comments', 'N/A')
        
                st.markdown(f"**Likes:** {likes} | **Views:** {views} | **Comments:** {comments}")
            else:
                st.markdown("Data not available")
            category_id=category(API_KEY,video_id)
            st.markdown(f"**Category:** {category_mapping.get(category_id, 'Unknown')}")

    
if (add_ratio=="Overall Sentiment"):
    overall_pos, overall_neg = overall_sentiment(st.session_state.videos_data)
    
    if (overall_pos + overall_neg == 0 and len(st.session_state.videos_data) != 0):
        st.markdown("No comments found")
    else:
        labels = ['Positive', 'Negative']
        values = [overall_pos, overall_neg]
        colors = ['#2ecc71', '#e74c3c']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color='black', width=2))
        )])

        fig.update_layout(
            title_text='Overall Sentiment Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(0,0,0,0)')
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

        
        if (add_ratio=="Your Video"):
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            result = get_video_stats(video_id)
            if result:
                likes = result.get('likes', 'N/A')
                views = result.get('views', 'N/A')
                comments = result.get('comments', 'N/A')
        
                st.markdown(f"**Likes:** {likes} | **Views:** {views} | **Comments:** {comments}")
            else:
                st.markdown("Data not available")
            category_id=category(API_KEY,video_id)
            st.markdown(f"**Category:** {category_mapping.get(category_id, 'Unknown')}")
            
        if (add_ratio=="Sentiment Analysis"):
            
            if (p+n==0):
                st.markdown("No comments found")
            else:
        
                # Labels and corresponding values
                labels = ['Positive', 'Negative']
                values = [p, n]
                colors = ['#2ecc71', '#e74c3c']  # Green and Red
                
                # Create Pie chart with custom colors and black border
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    marker=dict(
                        colors=colors,
                        line=dict(color='black', width=2)  # Black border around each slice
                    )
                )])
                
                # Set background to transparent or match Streamlit background
                fig.update_layout(
                    title_text='Sentiment Distribution',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(bgcolor='rgba(0,0,0,0)')
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
        if (add_ratio=="Word Cloud"):
            if (p+n==0):
                st.markdown("No comments found")
            else:
            
                generate_wordcloud_excluding_oov([x[0] for x in list_of_predicted_labels], tokenizer)

        if (add_ratio=="Top Positive Comments"):
            positive_comments = [item for item in list_of_predicted_labels if item[1] == 'Positive']

            # Sort them by score in descending order
            top_positive = sorted(positive_comments, key=lambda x: x[2], reverse=True)
            
            # Take the top 10
            top_10_positive = top_positive[:10]
            
            # Display in Streamlit
            st.subheader("Top 10 Positive Comments")
            for i, (comment, label, score,publish_time) in enumerate(top_10_positive, start=1):
                st.markdown(f"**{i}.** {comment}  \n🟢 *Score:* `{score:.3f}`")

        if (add_ratio=="Top Negative Comments"):
            negative_comments = [item for item in list_of_predicted_labels if item[1] == 'Negative']

            # Step 2: Sort them by score in descending order (higher score = more confident)
            top_negative = sorted(negative_comments, key=lambda x: x[2], reverse=True)
            
            # Step 3: Pick top 10
            top_10_negative = top_negative[-10:]
            
            # Step 4: Display using Streamlit
            st.subheader("Top 10 Negative Comments")
            for i, (comment, label, score,pulish_time) in enumerate(top_10_negative, start=1):
                st.markdown(f"**{i}.** {comment}  \n🔴 *Score:* `{score:.3f}`")
        
        if (add_ratio=="Recent Comment Summary"):
            recent_comment_summary()

        if (add_ratio=="Comment Search box"):
            search_comments(list_of_predicted_labels)
        
    else:
        st.markdown("Not a valid video ID")






