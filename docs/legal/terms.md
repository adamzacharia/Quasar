# Terms and Conditions
**Effective Date: May 24, 2026**
**Version: 2.0**

Welcome to QUASAR (the "Platform" or "Service"), operated by Quasar Observatory ("Company," "we," "us," or "our"). 

Please read these Terms and Conditions ("Terms") carefully. These Terms constitute a binding legal agreement between you ("User," "Investigator," "you," or "your") and Quasar Observatory. By creating an authenticated account, accessing the Platform, or using its associated features, you explicitly agree to be bound by these Terms, our Privacy Policy, and our Acceptable Use Policy. If you do not agree, you must immediately cease all access and use of the Service.

---

## 1. Description of Service & Account Registration

### 1.1 Scope of Service
Quasar is an artificial intelligence-powered astronomical research assistant designed to facilitate the search, retrieval, and analysis of multi-archive space observation data (including ALMA, NASA ADS, SIMBAD, CADC, and MAST). The Service includes natural language query processing, automated workflow composition (Conductor DAG engine), a sandboxed execution environment (Python REPL), and visual data extraction tools (FITS cube analysis).

### 1.2 Access & Mandatory Authentication
To keep our platform secure, fast, and free from abuse, guest or anonymous access is strictly prohibited.
* **Credentials:** You must create and authenticate through verified credentials (either a unique email/password combination or Google Single Sign-On).
* **Information Accuracy:** You agree to provide accurate, current, and complete information during registration.
* **Security Responsibility:** You are solely responsible for maintaining the confidentiality of your session tokens, passwords, and account credentials. Any activities that occur under your authenticated account are deemed your responsibility. You must notify us immediately at `security@quasarassistant.com` of any unauthorized use of your account.

### 1.3 Age Restrictions & COPPA Compliance
The Service is intended solely for users who are thirteen (13) years of age or older. By registering an account or using the Platform, you represent, warrant, and certify that you are at least 13 years of age. 
We do not knowingly collect, store, or solicit personal information from children under the age of 13. In accordance with the Children's Online Privacy Protection Act (COPPA) and global minor data protection regulations:
* **Detection & Wiping:** If we detect, learn, or receive a credible notification that we have inadvertently collected personal information from a child under the age of 13, we will immediately and without notice terminate the associated user account and take all necessary measures to permanently purge such information from our databases, logs, and server cache systems.
* **Inquiries:** If you have reason to believe that we might have any information from or about a child under 13, please contact our Legal Department immediately at `legal@quasarassistant.com`.

---

## 2. Platform Usage & License Grant

### 2.1 License Grant
Subject to your compliance with these Terms, we grant you a limited, non-exclusive, non-transferable, revocable, and personal license to access and use Quasar and its associated features for your scientific, academic, and other **noncommercial** research purposes. This grant is consistent with the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) under which the Quasar software is licensed (see the `LICENSE` file): noncommercial use — including personal use, research, education, and use by nonprofits, public research organizations, and government institutions — is permitted, while commercial use of any kind is prohibited without a separate written commercial license from us.

### 2.2 Sandboxed Execution Environment
Quasar provides a Sandboxed Python REPL to run data reduction scripts and mathematical calculations.
* **Limited Environment:** The sandbox contains standard scientific dependencies (e.g., `numpy`, `astropy`, `pandas`).
* **Security Controls:** Any attempt to escape the sandboxed environment, access the underlying host system, run denial-of-service scripts, scrape system resources, or extract active runtime environment variables will result in immediate account termination and potential legal action.

---

## 3. Artificial Intelligence, Outputs, & Scientific Reliability

### 3.1 AI-Generated Outputs
The Platform utilizes advanced Large Language Models (LLMs) and custom automated agents to decompose scientific queries and summarize research papers. 
* **The Nature of LLMs:** You acknowledge that generative AI model outputs are probabilistic in nature.
* **Hallucinations & Disclaimers:** AI-generated scripts (including CASA and CARTA recipes), mathematical calculations, and consensus summaries may contain errors, inaccuracies, or "hallucinations."
* **Verification Duty:** You are solely responsible for verifying, validating, and testing all code, coordinates, lines, and literature facts generated by the Platform before using them in observational proposals, physical telescope runs, or peer-reviewed publications. Quasar is not liable for lost telescope time, ruined observations, or academic errors resulting from unverified AI outputs.

### 3.2 Uptime & Beta Features
Quasar is constantly evolving. 
* **Beta Features:** Certain services (such as automated proposal critiquing, 3D cube rendering, or CARTA integrations) may be marked as "Beta" or "Experimental." These features are provided "AS IS" and may contain bugs, experience sudden changes, or be removed without notice.
* **Uptime Interruptions:** We do not guarantee uninterrupted availability of the Platform. Maintenance, high server demand, external observatory API downtime, or cloud container rebuilds may cause temporary outages.

---

## 4. Proprietary Data & Scientific Sovereignty

### 4.1 Your Data and Discoveries
We respect the intellectual property of the scientific community.
* **Ownership:** You retain 100% ownership, intellectual property rights, and publication claims over all files, observational images, scientific parameters, and raw FITS data you upload to the Platform.
* **Platform Rights:** Quasar acts strictly as an automated processing assistant. We do not claim any rights, titles, or licensing options over your discoveries.

### 4.2 Temporary Isolated Processing
Any FITS files or observation logs uploaded to the Platform are processed inside isolated, temporary containers. These files are securely cached and isolated so that other platform users cannot access, view, or scrape your proprietary observational findings.

### 4.3 Training & Model Improvement
By using the Platform, you acknowledge and agree that any data you share with Quasar—including but not limited to queries, conversation histories, uploaded documents, and search parameters—may be used to train, fine-tune, evaluate, or otherwise improve Quasar's systems, models, and associated Large Language Models (LLMs). This data helps us build better astronomical reasoning, improve tool accuracy, and enhance the overall quality of the Service for all users.

---

## 5. Subscriptions, Payments, & Fair Use

### 5.1 Pricing & Payment Structures
Quasar is currently offered as a service for the astronomical community. 
* **Future Tiers:** We reserve the right to introduce premium paid subscription tiers or institutional licensing frameworks. 
* **Notice:** Should paid billing plans be established, existing users will be provided clear, advanced notice and the option to opt-in or migrate their account metrics.

### 5.2 Fair Use & API Restrictions
To keep our platform accessible to all astronomers, you agree to utilize Quasar through our standard graphical interface. Mass scraping, building unofficial automated wrapper APIs around our endpoints, or sending automated high-frequency queries that exceed typical human speeds without explicit API keys is strictly prohibited.

---

## 6. Term & Account Termination

### 6.1 Termination by User
You may stop using the Service and request the complete deletion of your account and thread history at any time through the dashboard Settings.

### 6.2 Termination by Quasar
We reserve the right to suspend or terminate your account and block your access to the Service immediately, without prior notice, if:
* You violate any provision of these Terms or the Acceptable Use Policy.
* Your account exhibits behavior indicating scraping, API abuse, or security escape attempts.
* We are required to do so by governing law or observatory mandates.

---

## 7. Limitation of Liability & Warranty Disclaimer

### 7.1 Warranty Disclaimer
THE PLATFORM IS PROVIDED "AS IS" AND "AS AVAILABLE," WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.

### 7.2 Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL QUASAR OBSERVATORY, ITS DIRECTORS, EMPLOYEES, OR PARTNERS BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF DATA, LOSS OF TELESCOPE OBSERVING TIME, PUBLICATION RETRACTIONS, ACADEMIC DISPUTES, OR FINANCIAL LOSSES ARISING OUT OF OR IN CONNECTION WITH YOUR ACCESS TO OR USE OF THE SERVICE.

---

## 8. Governing Law & Dispute Resolution

These Terms shall be governed by and construed in accordance with the laws of the State of Delaware, United States, without regard to its conflict of law principles. Any legal actions or proceedings arising under these Terms shall be brought exclusively in the state or federal courts located in Wilmington, Delaware, and you hereby consent to the personal jurisdiction and venue therein.

---

## 9. Contact Us

For any legal notices, violations of these terms, or support queries, please contact us at:
**Email:** `legal@quasarassistant.com`  
**Mailing Address:** [Quasar Observatory Legal Department, Placeholder Wilmington, DE, USA]
