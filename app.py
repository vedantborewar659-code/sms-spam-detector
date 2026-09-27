import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="SMS Shield",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    background-color: white;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    text-align: center;
}

.safe {
    border-left: 7px solid #16a34a;
}

.spam {
    border-left: 7px solid #dc2626;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.example-box {
    padding: 12px;
    border-radius: 10px;
    background-color: #eef2ff;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():

    url = (
        "https://raw.githubusercontent.com/justmarkham/"
        "pycon-2016-tutorial/master/data/sms.tsv"
    )

    data = pd.read_csv(
        url,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    data["label"] = data["label"].astype(str).str.lower()
    data["message"] = data["message"].astype(str)

    return data


# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam"
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam"
    )
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )

    return (
        vectorizer,
        model,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        y_test,
        predictions
    )


# ---------------- LOAD ----------------
try:

    data = load_data()

    (
        vectorizer,
        model,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        y_test,
        predictions
    ) = train_model(data)

except Exception as e:

    st.error("Unable to load the SMS dataset.")
    st.info(
        "Please make sure your internet connection is active "
        "and all required packages are installed."
    )

    # TEMPORARY: show the real error so we can debug it
    st.exception(e)

    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">

<h1>🛡️ SMS Shield</h1>

<p>
Machine Learning Based SMS Spam Detection System
</p>

</div>
""", unsafe_allow_html=True)


# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Detect SMS", "📊 Model Performance", "ℹ️ About Project"]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Analyze a Message")

    st.write(
        "Enter an SMS message below and the machine learning model "
        "will determine whether it is Spam or Safe."
    )

    message = st.text_area(
        "Enter SMS message",
        placeholder="Example: Congratulations! You have won a prize...",
        height=150
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🚨 Analyze Message",
            use_container_width=True
        ):

            if message.strip() == "":
                st.warning("Please enter an SMS message first.")

            else:

                transformed = vectorizer.transform([message])

                prediction = model.predict(transformed)[0]

                probabilities = model.predict_proba(transformed)[0]

                confidence = max(probabilities) * 100

                if prediction == "spam":

                    st.markdown(
                        f"""
                        <div class="result-card spam">

                        <h1>🚨 SPAM DETECTED</h1>

                        <h3>Confidence: {confidence:.2f}%</h3>

                        <p>
                        This message has characteristics commonly
                        associated with spam messages.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="result-card safe">

                        <h1>✅ SAFE MESSAGE</h1>

                        <h3>Confidence: {confidence:.2f}%</h3>

                        <p>
                        This message appears to be a legitimate SMS.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    st.markdown("---")

    st.subheader("💡 Try Example Messages")

    example1, example2, example3 = st.columns(3)

    with example1:

        st.markdown(
            """
            <div class="example-box">
            🎁 Congratulations! You have won a free prize.
            </div>
            """,
            unsafe_allow_html=True
        )

    with example2:

        st.markdown(
            """
            <div class="example-box">
            📱 Hey, are we meeting at college today?
            </div>
            """,
            unsafe_allow_html=True
        )

    with example3:

        st.markdown(
            """
            <div class="example-box">
            💰 Claim your FREE reward now by clicking this link!
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("📊 Model Performance")

    st.write(
        "The model was trained using the SMS Spam Collection dataset "
        "and evaluated on unseen test data."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("📈 Dataset Distribution")

        counts = data["label"].value_counts()

        fig1, ax1 = plt.subplots()

        ax1.bar(
            ["Safe", "Spam"],
            [
                counts.get("ham", 0),
                counts.get("spam", 0)
            ]
        )

        ax1.set_ylabel("Number of Messages")
        ax1.set_title("SMS Dataset Distribution")

        st.pyplot(fig1)


    with right:

        st.subheader("🎯 Confusion Matrix")

        fig2, ax2 = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["Safe", "Spam"],
            yticklabels=["Safe", "Spam"],
            ax=ax2
        )

        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        st.pyplot(fig2)


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("ℹ️ About SMS Shield")

    st.write("""
    *SMS Shield* is a machine learning based application that
    automatically identifies whether an SMS message is legitimate
    or spam.
    """)

    st.markdown("### 🔄 Machine Learning Pipeline")

    st.write("""
    SMS Message
    ↓
    Text Preprocessing
    ↓
    TF-IDF Feature Extraction
    ↓
    Multinomial Naive Bayes
    ↓
    Spam / Safe Prediction
    """)

    st.markdown("### 🧠 Technologies Used")

    st.write("""
    • Python  
    • Streamlit  
    • Pandas  
    • Scikit-learn  
    • TF-IDF Vectorization  
    • Multinomial Naive Bayes  
    • Matplotlib  
    • Seaborn
    """)

    st.markdown("### 📚 Dataset")

    st.write("""
    The project uses the SMS Spam Collection dataset containing
    SMS messages classified as either ham (legitimate) or spam.
    """)

    st.success(
        "The system helps users identify suspicious SMS messages "
        "using machine learning."
    )


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "SMS Shield • Machine Learning Mini Project • "
    "Spam Message Detection"
)
