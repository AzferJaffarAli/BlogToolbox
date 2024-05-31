import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob

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
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
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

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='wide')

# Sidebar
st.sidebar.title("Blog Toolbox")

# Function for generating blogs
def generate_blog():
    st.subheader("Generate Content For Blogs")
    input_text = st.text_input("Enter the Blog Topic")
    no_words = st.text_input("Number of Words")
    blog_style = st.selectbox("Writing the blog for", ('Researchers', 'Data Scientist', 'Common People'), index=0)
    if st.button("Generate"):
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.write(response)

# Function for text summarization
def summarize_text():
    st.subheader("Text Summarization")
    text_to_summarize = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        summary = text_summarization(text_to_summarize)
        st.write("Summary:", summary)

# Function for sentiment analysis
def analyze_sentiment_func():
    st.subheader("Sentiment Analysis")
    text_for_sentiment = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(text_for_sentiment)
        st.write("Sentiment:", sentiment)

# Function for Content Idea Generator
def generate_content_ideas():
    st.subheader("Content Idea Generator")
    st.write("Coming soon...")

# Function for Content Calendar
def manage_content_calendar():
    st.subheader("Content Calendar")
    st.write("Coming soon...")

# Main content area
st.title("Blog Toolbox")
selected_tool = st.sidebar.radio("Select Tool", ("Generate Blogs", "Text Summarization", "Sentiment Analysis", "Content Idea Generator", "Content Calendar"))

if selected_tool == "Generate Blogs":
    generate_blog()
elif selected_tool == "Text Summarization":
    summarize_text()
elif selected_tool == "Sentiment Analysis":
    analyze_sentiment_func()
elif selected_tool == "Content Idea Generator":
    generate_content_ideas()
elif selected_tool == "Content Calendar":
    manage_content_calendar()
