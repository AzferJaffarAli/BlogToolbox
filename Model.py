import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import datetime

# Download NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set page configuration
st.set_page_config(
    page_title="Blog Toolbox",
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
    .content-header {
      font-size: 1.5rem;
      font-weight: bold;
      margin-bottom: 1rem;
    }
    .input-section {
      margin-bottom: 1.5rem;
    }
    .related-keywords {
      margin-top: 1.5rem;
      font-style: italic;
      color: #666;
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
        config={'max_new_tokens': 512, 'temperature': 0.01}
    )
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = llm(formatted_prompt)
    return response

def text_summarization(text, reduction_ratio):
    original_length = len(text.split())
    target_length = int(original_length * (reduction_ratio / 100))
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=target_length)
    return " ".join(str(sentence) for sentence in summary)

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0:
        sentiment = "Positive"
        comments = "Great! Your text has a positive sentiment."
    elif sentiment_score < 0:
        sentiment = "Negative"
        comments = "Uh-oh! Your text has a negative sentiment."
    else:
        sentiment = "Neutral"
        comments = "Hmm. Your text has a neutral sentiment."
    
    return sentiment, sentiment_score, comments

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
    llm = CTransformers(
        model='C:\\Users\\Azfer\\Downloads\\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 512, 'temperature': 0.01}
    )
    template = """
        Generate 10 unique and creative content ideas for a blog topic on {topic}.
    """
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    formatted_prompt = prompt.format(topic=topic)
    response = llm(formatted_prompt)
    ideas = response.split('\n')
    return [idea.strip() for idea in ideas if idea.strip()]

def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    return bleu_score

# Initialize session state for content calendar
if "content_calendar" not in st.session_state:
    st.session_state["content_calendar"] = pd.DataFrame(columns=["Content Idea", "Publication Date"])

# Initialize session state for sentiment analysis results
if "sentiment_results" not in st.session_state:
    st.session_state["sentiment_results"] = pd.DataFrame(columns=["Text", "Sentiment", "Sentiment Score"])

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
    "Content Calendar": "Organize and manage your content publishing schedule.",
    "Model Evaluation": "Evaluate the model's accuracy using BLEU score."
}

# Titles for each option
tool_titles = {
    "Generate Blogs": "Generate Blogs",
    "Text Summarization": "Text Summarization",
    "Sentiment Analysis": "Sentiment Analysis",
    "Content Idea Generator": "Content Idea Generator",
    "Content Calendar": "Content Calendar",
    "Model Evaluation": "Model Evaluation"
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
    st.markdown("<div class='content-header'>Generate a Blog Post</div>", unsafe_allow_html=True)
    
    input_col, words_col = st.columns([3, 1])
    with input_col:
        input_text = st.text_input("Enter the Blog Topic", max_chars=100, placeholder="e.g., Data Science Trends")
    with words_col:
        no_words = st.text_input("Number of Words", max_chars=5, placeholder="e.g., 500")
    
    blog_style = st.selectbox("Writing the blog for", ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
    if st.button("Generate"):
        with st.spinner("Generating blog..."):
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.text_area("Generated Blog", response, height=200)
            st.markdown("</div>", unsafe_allow_html=True)
        
        if input_text:
            st.markdown("<div class='related-keywords'>Related Keywords:</div>", unsafe_allow_html=True)
            keywords = get_related_keywords(input_text)
            st.write(", ".join(keywords))

elif selected_tool == "Text Summarization":
    st.subheader("Text Summarization")
    text_to_summarize = st.text_area("Enter text to summarize:", height=200)
    reduction_ratio = st.slider("Reduction Ratio (%)", min_value=10, max_value=90, value=30, step=10)
    
    if st.button("Summarize"):
        with st.spinner("Summarizing text..."):
            summary = text_summarization(text_to_summarize, reduction_ratio)
            st.write("Summary:", summary)

elif selected_tool == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    text_for_sentiment = st.text_area("Enter text for sentiment analysis:", height=200)
    
    if st.button("Analyze Sentiment"):
        if text_for_sentiment.strip() == "":
            st.error("Please enter some text for sentiment analysis.")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, sentiment_score, comments = analyze_sentiment(text_for_sentiment)
                st.markdown("### Sentiment Analysis Results")
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown(f"**Sentiment Score:** {sentiment_score:.2f}")
                st.markdown(f"**Comments:** {comments}")
                
                # Update session state with results
                new_entry = pd.DataFrame([[text_for_sentiment, sentiment, sentiment_score]], columns=["Text", "Sentiment", "Sentiment Score"])
                st.session_state["sentiment_results"] = pd.concat([st.session_state["sentiment_results"], new_entry], ignore_index=True)
                
                # Display historical sentiment analysis results
                st.markdown("### Historical Sentiment Analysis Results")
                st.dataframe(st.session_state["sentiment_results"])

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

elif selected_tool == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write("Evaluate the model's accuracy using BLEU score.")
    
    # Assuming you have a test dataset
    test_data = pd.DataFrame({
        'input': ['Data Science Trends', 'AI in Healthcare', 'Future of Robotics'],
        'expected_output': [
            'Expected blog text about Data Science Trends...',
            'Expected blog text about AI in Healthcare...',
            'Expected blog text about the Future of Robotics...'
        ]
    })
    
    if st.button("Evaluate Model"):
        with st.spinner("Evaluating model..."):
            test_data['generated_output'] = test_data['input'].apply(lambda x: getLLamaresponse(x, 500, 'Researchers'))
            test_data['bleu_score'] = test_data.apply(lambda row: calculate_bleu(row['expected_output'], row['generated_output']), axis=1)
            average_bleu_score = test_data['bleu_score'].mean()
            st.write(f"Average BLEU Score: {average_bleu_score:.4f}")

