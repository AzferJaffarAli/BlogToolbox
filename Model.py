import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
import pandas as pd
import datetime

# Download NLTK resources
nltk.download('wordnet')
nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS with gradient background and reduced sidebar size
custom_css = """
  <style>
    .stApp {
      display: flex;
      flex-direction: row;
      align-items: stretch;
      background: linear-gradient(to right, #f0f0f0 0%, #EFE7BC 100%);
    }
    .stSidebar {
      flex: 0.5;
      padding: 2rem;
      background-color: #EFE7BC;
    }
    .stContent {
      flex: 2.5;
      padding: 2rem;
      background-color: transparent;
    }
    .stButton>button, .stSidebar .stRadio>div>div>div>div {
      background-color: #4CAF50;
      color: white;
      padding: 12px 20px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      margin-bottom: 10px;
    }
    .stButton>button:hover, .stSidebar .stRadio>div>div>div>div:hover {
      background-color: #45a049;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
      background-color: #ffffff;
      color: #333333;
      border-radius: 0.5rem;
    }
    .stSelectbox>div>div>div>div {
      background-color: #ffffff;
      color: #333333;
      border-radius: 0.5rem;
    }
    .option-description {
      margin-bottom: 10px;
    }
  </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Disable the warning for pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(
        model='C:\\Users\\Azfer\\Downloads\\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)
    response = llm.prompt(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

def text_summarization(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3)
    return " ".join(str(sentence) for sentence in summary)

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def get_related_keywords(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    related_keywords = set()
    for word, pos in pos_tags:
        if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ') or pos.startswith('RB'):
            synonyms = wordnet.synsets(word)
            for syn in synonyms:
                for lemma in syn.lemmas():
                    related_keywords.add(lemma.name())
    return list(related_keywords)

def generate_content_ideas(topic):
    ideas = [
        f"10 Tips for {topic}",
        f"How to Improve Your {topic} Skills",
        f"The Future of {topic}: Trends to Watch",
        f"Beginner's Guide to {topic}",
        f"Top 5 Tools for {topic}",
        f"Common Mistakes in {topic} and How to Avoid Them",
        f"Expert Interviews: Insights on {topic}",
        f"Case Studies in {topic}",
        f"The Benefits of {topic}",
        f"Advanced Techniques in {topic}"
    ]
    return ideas

# Initialize session state for content calendar
if "content_calendar" not in st.session_state:
    st.session_state["content_calendar"] = pd.DataFrame(columns=["Content Idea", "Publication Date"])

# Add headers and subheaders
st.title("Blog Toolbox")
st.subheader("Enhance Your Content Creation Experience")

# Sidebar
st.sidebar.title("Blog Toolbox")

# Descriptions and corresponding options for toolbox
tool_descriptions = {
    "Generate Blogs": "Generate blogs based on specific topics and job profiles.",
    "Text Summarization": "Summarize lengthy texts into shorter, concise versions.",
    "Sentiment Analysis": "Analyze the sentiment of texts to determine positivity, negativity, or neutrality.",
    "Content Idea Generator": "Get creative ideas for generating content.",
    "Content Calendar": "Organize and manage your content publishing schedule."
}

# Titles for each option
tool_titles = {
    "Generate Blogs": "Generate Blogs",
    "Text Summarization": "Text Summarization",
    "Sentiment Analysis": "Sentiment Analysis",
    "Content Idea Generator": "Content Idea Generator",
    "Content Calendar": "Content Calendar"
}

# Display descriptions and radio buttons
tool_options = list(tool_descriptions.keys())
selected_tool = st.sidebar.radio("Select Tool", tool_options)

# Display selected tool description
st.sidebar.markdown(f"**{tool_titles[selected_tool]}**")
st.sidebar.markdown("### Description")  # Added description title
st.sidebar.markdown(tool_descriptions[selected_tool])

# Add additional elements below the description
st.sidebar.markdown("### About")
st.sidebar.markdown("This application helps you generate blogs, summarize text, analyze sentiment, and manage your content calendar.")

st.sidebar.markdown("### Contact")
st.sidebar.markdown("For more information, contact us at: info@blogtoolbox.com")

# Main content area
if selected_tool == "Generate Blogs":
    st.subheader("Generate Content For Blogs")
    input_text = st.text_input("Enter the Blog Topic", max_chars=100)
    no_words = st.text_input("Number of Words", max_chars=5)
    blog_styles = ('Researchers', 'Data Scientist', 'Common People')
    blog_style = st.selectbox("Writing the blog for", blog_styles, format_func=lambda x: x, index=0)
    if st.button("Generate"):
        with st.spinner("Generating blog..."):
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.write(response)
        if input_text:
            st.subheader("Related Keywords")
            keywords = get_related_keywords(input_text)
            st.write(", ".join(keywords))

elif selected_tool == "Text Summarization":
    st.subheader("Text Summarization")
    text_to_summarize = st.text_area("Enter text to summarize:", height=200, max_chars=1000)
    if st.button("Summarize"):
        with st.spinner("Summarizing text..."):
            summary = text_summarization(text_to_summarize)
            st.write("Summary:", summary)

elif selected_tool == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    text_for_sentiment = st.text_area("Enter text for sentiment analysis:", height=200, max_chars=1000)
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(text_for_sentiment)
            st.write("Sentiment:", sentiment)

elif selected_tool == "Content Idea Generator":
    st.subheader("Content Idea Generator")
    topic = st.text_input("Enter the topic for content ideas", max_chars=50)
    if st.button("Generate Ideas"):
        with st.spinner("Generating ideas..."):
            ideas = generate_content_ideas(topic)
            st.write("Here are some content ideas:")
            for idea in ideas:
                st.write(f"- {idea}")

elif selected_tool == "Content Calendar":
    st.subheader("Content Calendar")
    st.write("Plan your content with a publication schedule.")
    
    new_content_idea = st.text_input("New Content Idea", max_chars=50)
    publication_date = st.date_input("Publication Date", datetime.date.today())
    
    if st.button("Add to Calendar"):
        new_entry = pd.DataFrame([[new_content_idea, publication_date]], columns=["Content Idea", "Publication Date"])
        st.session_state["content_calendar"] = pd.concat([st.session_state["content_calendar"], new_entry], ignore_index=True)
    
    st.write("### Scheduled Content")
    st.dataframe(st.session_state["content_calendar"])

    if st.button("Clear Calendar"):
        st.session_state["content_calendar"] = st.session_state["content_calendar"].iloc[0:0]
        st.write("Calendar cleared.")
