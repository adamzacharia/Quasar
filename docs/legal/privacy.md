# Privacy Policy
**Effective Date: May 24, 2026**
**Version: 2.0**

Quasar Observatory ("we," "us," or "our") operates the QUASAR Astronomical Assistant platform. We are committed to protecting the privacy, confidentiality, and security of our users—primarily astronomers, academic investigators, and institutional researchers. 

This Privacy Policy explains how we collect, use, store, disclose, and protect your personal information, telemetry, and scientific metadata when you access the Platform, and details your compliance rights under the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA).

---

## 1. Information We Collect & Process

To provide highly specialized astronomical agent planning and thread continuity, we collect and process several categories of data:

### 1.1 Personal Account Data
* **Verified Credentials:** Your email address, encrypted password hash, and display name provided during registration.
* **Authentication Profiles:** Federated Single Sign-On (SSO) metadata provided via Google SSO authorization.

### 1.2 Scientific and Research Data
* **Celestial Coordinates:** Target designations (e.g., HL Tau), Right Ascension (RA), Declination (Dec), and observational bands/filters searched.
* **Literature Outlines:** Abstract parameters and DOI queries sent to external registries like the NASA Astrophysics Data System (ADS).
* **FITS Metadata:** Visual dimensions, coordinate grids, and FITS header parameters extracted when you upload observation files to be analyzed or plotted.
* **Personal Document RAG Store:** Observational proposals, scripts, or PDF papers that you voluntarily upload to your personal vector collection for Retrieval-Augmented Generation context.

### 1.3 Telemetry, Tracing, and Connection Data
* **Observability Spans (Langfuse):** To diagnose system crashes and trace multi-step Conductor loops, we securely capture LLM prompt latencies, subtask DAG pathways, and tool execution times.
* **Session Metadata:** IP addresses, browser footprints (User Agent, operating system), and geo-location details (Vercel/Cloudflare headers) strictly to maintain session tokens and prevent platform abuse.

---

## 2. Databases & Core Integrations

We process and store your information using a highly resilient, modern database stack:

### 2.1 Turso Distributed Database
All active user profiles, conversation thread buffers, and account setups are securely stored inside a globally distributed SQLite database engine powered by Turso (hosted with enterprise-grade LibSQL engines), ensuring instant query delivery and thread continuity across multiple locations.

### 2.2 Langfuse Observability Suite
To monitor runtime agent planning, all celestial tool executions, web searches, and reasoning LLM streams areParent-nested under secure generation spans inside our Langfuse pipeline. This data is kept strictly private and is utilized exclusively by system operators to improve response accuracy, fix tool failures, and validate LLM performance.

### 2.3 Temporary Container Cache
Uploaded raw files and visual plots are stored in isolated, temporary container directories. These files are kept completely private, separate from other users, and are cleared out upon session exit.

---

## 3. How We Use Your Information

We use the collected information for the following specific purposes:
* **Core Functionality:** To authenticate your session, stream real-time chat responses, compose Observational Conductor DAGs, and generate downloadable Jupyter notebooks.
* **Observability & Bug Fixes:** To audit tool execution logs, review failed subtasks, and track cost/latency analytics.
* **Personalization:** To build preference memory models that learn your specific observational bands and academic areas of interest over time.
* **Training & Model Improvement:** Your queries, conversation histories, uploaded documents, and search parameters may be used to train, fine-tune, evaluate, or otherwise improve Quasar's systems, models, and associated Large Language Models (LLMs). This helps us build better astronomical reasoning and improve tool accuracy for all users.
* **Security & Fair Use:** To monitor access rates, verify JWT tokens, detect prompt injection attacks, and block automated scrapers.

---

## 4. How We Share Your Information

We do not sell, rent, or trade your personal or scientific data to third parties. We only share data with trusted third-party providers necessary to execute astronomical tools and query archives:
* **Observatory Archives:** We transmit target coordinates and query parameters to public astronomical archives (ALMA TAP, CADC, VizieR, SIMBAD, MAST, ESO) to fetch observation lists.
* **Literature Databases:** We query NASA ADS and OpenAlex registries to look up citations and researcher bibliometrics.
* **AI Model Providers:** We pass anonymized prompt contexts to OpenAI or secure LLM failovers to generate conversation replies. These LLM providers do not use your inputs to train their baseline models.
* **Cloud Infrastructure:** Your database registers reside securely inside Turso and Qdrant Cloud nodes.

---

## 5. GDPR & CCPA Compliance and User Rights

We act as both a Data Controller and Data Processor. If you are located in the European Economic Area (EEA) or California, USA, you hold specific legal rights:

### 5.1 Your Legal Rights
* **Right of Access:** You have the right to request a full export of all your chat histories, uploaded personal documents, and profile metrics.
* **Right to Rectification:** You can update your display name, email address, and passwords at any time through the User Dashboard.
* **Right to Erasure (Right to Be Forgotten):** You can request the complete and permanent deletion of your account and thread histories.
* **Right to Object/Restrict:** You can opt-out of sharing anonymous telemetry or cost metrics.

### 5.2 Deletion & Purging Controls
At any time, you can manually delete individual conversation threads in the sidebar or trigger a complete account wipe inside your Settings. Once confirmed, all referenced user rows, vector indexes, and trace correlations are permanently deleted from our active Turso and Qdrant instances.

---

## 6. Security Covenants

We implement robust administrative, technical, and physical security measures to protect your credentials and research metadata:
* **Encryption:** All transit data is encrypted via TLS/SSL connections, and active sessions utilize secure, signed JSON Web Tokens (JWT).
* **Isolation:** User uploads, sandboxed REPL environments, and visual FITS caches are completely isolated to prevent cross-account data leakage.
* **Secrets Management:** Environment API keys, database credentials, and JWT signing keys are managed securely with production vaults.

---

## 7. Changes to This Privacy Policy

We may update this Privacy Policy from time to time. When changes are made, we will update the "Effective Date" at the top of this document. We encourage you to review this policy periodically to stay informed about how we protect your data.

---

## 8. Contact Our Data Protection Officer

For GDPR/CCPA data export requests, account erasures, or privacy compliance questions, please reach out to our team at:
**Email:** `privacy@quasarassistant.com`  
**Address:** [Quasar Observatory DPO, Placeholder Wilmington, DE, USA]
