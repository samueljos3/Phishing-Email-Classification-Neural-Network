import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
import pickle
import app.data_processing as data_processing

df_emails = pd.DataFrame({
    "texto": [
        # 1 - Phishing
        """Subject: Urgent: Account Verification Required

Dear Customer,

Our system has detected an issue with your account. To prevent it from being suspended, please verify your information through the link below:

[Verify Account Now]

Failure to do so within 24 hours will result in a permanent lock.

Thank you for your prompt attention to this matter.

Account Security Team
""",
        # 2 - Phishing
        """Subject: Immediate Action Needed: Suspicious Login Activity

Dear User,

We noticed a suspicious login attempt from a new device. If this wasn’t you, please verify your account immediately by clicking on the secure link below:

[Secure Your Account]

Regards,
Security Team
""",
        # 3 - Phishing
        """Subject: You Have A Pending Payment Request

Dear Customer,

There is an outstanding payment of $225.48 on your account that needs to be cleared immediately. Click the link below to make your payment and avoid penalties:

[Pay Now]

Failure to act within 48 hours may result in account restrictions.

Best regards,
Billing Department
""",
        # 4 - Phishing
        """Subject: Your Shipment Is On Hold: Immediate Action Needed

Dear Shopper,

We attempted to deliver your recent order, but there was an issue with the shipping address. To avoid delays, please confirm your address by clicking on the link below:

[Confirm Address]

We appreciate your cooperation.

Shipping Department
""",
        # 5 - Phishing
        """Subject: You Won $500! Confirm Your Prize Now

Dear Winner,

Congratulations! You’ve won a $500 Amazon gift card. Please confirm your details by clicking the link below to claim your prize:

[Claim Your Prize]

Don’t miss out – offer expires soon!

Best,
Prize Notification Team
""",
        # 6 - Não Phishing
        """Subject: Meeting Agenda for Upcoming Strategy Session

Hello Team,

I’ve attached the agenda for our upcoming strategy session scheduled for next Tuesday at 9 AM. Please take a moment to review the topics before the meeting.

Looking forward to our discussion!

Best regards,
Strategy Team
""",
        # 7 - Não Phishing
        """Subject: Reminder: Time-Off Requests for Upcoming Holidays

Dear Employee,

This is a friendly reminder to submit your time-off requests for the upcoming holiday season. Ensure your requests are submitted by the deadline to allow for scheduling adjustments.

Let HR know if you need assistance.

Best regards,
Human Resources
""",
        # 8 - Não Phishing
        """Subject: Invitation to the Charity Fundraiser Event

Dear Team,

We are hosting our annual charity fundraiser on May 15th, and we would love for you to join us in supporting a good cause. The event will be held at City Hall, starting at 6 PM.

Please RSVP if you plan to attend.

Best,
Event Coordination Team
""",
        # 9 - Não Phishing
        """Subject: New Employee Onboarding Materials

Dear New Hire,

Welcome to the team! Attached, you’ll find all the necessary onboarding documents and instructions for your first day. Please review everything before your start date.

We’re excited to have you onboard!

Best regards,
HR Team
""",
        # 10 - Não Phishing
        """Subject: Invitation to Participate in Wellness Program

Dear Staff,

We are excited to announce the launch of our new wellness program designed to promote healthy living. You’re invited to join us for an introductory session on Monday at 3 PM.

We look forward to your participation!

Best,
Wellness Program Team
"""
    ]
})


preprocessor = data_processing.TextPreprocessor()
df_emails["texto_proc"] = df_emails["texto"].apply(preprocessor.preprocess)

vectorizer = data_processing.Word2VecVectorizer(model_path="models/models_w2v/vmodelo_w2v.model")
X = vectorizer.vectorize_texts(df_emails["texto_proc"])

with open("models/models_phishing/model_phishing.pkl", "rb") as f:
    modelo = pickle.load(f)

probs = modelo.predict(X) 
preds = (probs > 0.5).astype(int).flatten()  

df_resultado = df_emails.copy()
df_resultado["phishing"] = preds
df_resultado["prob_phishing"] = probs.flatten()  

for i, row in df_resultado.iterrows():
    status = "Phishing" if row["phishing"] == 1 else "Não Phishing"
    print(f"[{i+1}] {status} (Confiança: {row['prob_phishing']:.2f})")