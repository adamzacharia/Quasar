# 🔭 Quasar Beta Testing Guide

Welcome to the **Quasar** beta! Quasar is an AI-powered assistant for radio astronomy, designed to help you search archives, find research papers, and answer technical questions using the ALMA Science Archive and NASA ADS.

This guide will help you set up the system, test its core features, and provide valuable feedback to help us improve.

## 🚀 Getting Started

### 1. Prerequisites
Ensure the system is running (ask the administrator for the URL or run locally).
If running locally, you need:
- `OPENAI_API_KEY` (for the AI brain)
- `NASA_ADS_API_KEY` (for paper searches)

### 2. Registration & Login
Quasar has a built-in user management system.
1. On the landing page, click the **"Register"** tab.
2. Enter a username and password.
3. Click "Register".
4. Switch back to the **"Login"** tab and log in with your new credentials.

---

## 🎮 How to Use Quasar

Quasar understands natural language, but you can use **Command Tags** to force specific behaviors.

### 📡 Archive Search (`@archive`)
Use this to find real astronomical datasets in the ALMA/NRAO archives.
- **Goal**: Find observations, check sensitivity, or get band info.
- **Example**: 
  > `@archive Find Band 6 observations of Sz65 with high sensitivity.`
  > `@archive Show me CO(2-1) data for the galaxy M83.`
- **What to check**: Does it return a data table? Are the columns (Beam, Sensitivity) correct? Can you generate a plot?

### 🧠 Knowledge Search (`@search`)
Use this to ask technical questions about radio astronomy, facilities, or documentation.
- **Goal**: detailed explanations with citations.
- **Example**:
  > `@search What is the proprietary period for ALMA PI data?`
  > `@search How do I calibrate Cycle 10 data?`
- **What to check**: Does the answer cite a source (e.g., `[ALMA Manual, Page 42]`)? Is it accurate?

### 📚 Paper Search (`@paper`)
Use this to find relevant research papers from NASA ADS and arXiv.
- **Goal**: Literature review and finding properties.
- **Example**:
  > `@paper Find papers on the spectral index of 3C 273.`
  > `@paper What are the most recent papers about protoplanetary disks?`
- **What to check**: Does it show a list of papers? are the links clickable? Is the summary useful?

---

## 📝 Feedback Collection

We need your help to make Quasar smarter! Please use the table below to record your testing sessions.

### Key Metrics We Are Tracking
1. **Accuracy**: Did the agent understand *exactly* what you asked?
2. **Latency**: Was the response fast enough?
3. **Robustness**: Did it crash or give an empty error?
4. **"Magic"**: Did it offer to do something helpful you didn't explicitly ask for (e.g., "Shall I plot this for you?")?

### Feedback Form Template

| Feature Tested | Query Used | Successful? (Y/N) | Issues / Weird Behavior | Wishlist / Improvement Idea |
| :--- | :--- | :---: | :--- | :--- |
| **@archive** | *`@archive Find data on Sz65`* | Y | *None* | *I wish I could filter by date directly in the chat.* |
| **@search** | *`@search ALMA Cycle 9 dates`* | N | *Cited Cycle 8 instead.* | *Update the internal knowledge base.* |
| **@paper** | *`@paper Sz65 spectral analysis`* | Y | *Worked perfectly.* | *Show the abstract directly in the chat bubble.* |
| **Registration** | *Creating new user* | Y | *UI didn't refresh immediately.* | *Add a "Success" confetti animation.* |
| **General** | *`Hello, who are you?`* | Y | *Response was too robotic.* | *Give it more personality.* |

### 🐛 Reporting Bugs
If you encounter a crash or red error box:
1. Take a screenshot.
2. Copy the last command you sent.
3. Send it to the development team along with this feedback form.

Thank you for helping build the future of Radio Astronomy AI! 🌌
