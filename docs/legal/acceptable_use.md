# Acceptable Use Policy
**Effective Date: May 24, 2026**
**Version: 2.0**

This Acceptable Use Policy ("AUP") defines the acceptable parameters of conduct and usage governing your access to the QUASAR Astronomical Assistant platform. 

Our goal is to keep Quasar fast, secure, and helpful for the global astronomical research community. By using the Platform, you agree to comply with this AUP at all times.

---

## 1. Sandbox Safety & Code Execution Restrictions

Quasar provides a Sandboxed Python REPL to allow sub-agents and researchers to execute computations and plot spectra in real-time. To maintain the security of our host servers, you agree to the following strict boundaries:

* **No Sandbox Escapes:** You must not attempt to escape the Python sandboxed container. Any actions designed to gain root access to the host server, modify underlying container configurations, or execute unauthorized low-level system commands are strictly prohibited.
* **No Network Socket Abuse:** You must not utilize the sandboxed Python REPL to run port scans, open unauthorized reverse shells, coordinate distributed denial-of-service (DDoS) attacks, or transmit spam.
* **Resource Limits:** You must not execute infinite loops, run resource-exhausting memory leaks, or spin up unauthorized high-core parallel calculations that degrade system performance for other researchers.

---

## 2. API Abuse & Scraping Restrictions

Quasar connects to highly competitive, public astronomical archives and literature databases. To prevent database blocks and maintain fair use:

* **No Automated Scraping:** You must not use automated scraping scripts, bots, spiders, or head-less browsers to mass-extract our chat threads, processed FITS figures, or literature summaries without an authorized API key.
* **No Unofficial wrappers:** You must not build commercial wrappers or public proxy APIs around Quasar's backend endpoints (`/api/chat`, `/api/conversations`) to redistribute our LLM outputs, traced results, or conductor planning networks.
* **Respect Rate Limits:** You must respect the rate-limits implemented on our public endpoints. Excessive high-frequency queries that mimic automated scraper traffic are subject to automatic IP throttling and account suspension.

---

## 3. Input Safety, Prompt Injection, & AI Alignment

* **No Prompt Injection:** You agree not to input prompt injections or malicious instruction payloads designed to bypass the safety alignment of Quasar's backbone models, hijack sub-agents, or reveal underlying system prompt instructions.
* **No Malicious Payloads:** You must not upload files, code scripts, or documents containing malware, viruses, trojans, ransomware, or any other destructive software designed to compromise our cloud infrastructure or third-party databases.

---

## 4. Content Moderation & Academic Integrity

Quasar allows you to upload personal research proposals, papers, and FITS files to your personal RAG store.
* **No Offensive Content:** You must not upload documents containing hate speech, harassment, graphic violence, adult content, or any material that violates local laws.
* **Academic Verification:** You must not use Quasar to generate plagiarized content, fabricate research results, or engage in academic fraud. 

---

## 5. Violation Enforcement & Penalties

We maintain sole discretion in determining whether this Acceptable Use Policy has been violated. Violations of this AUP may result in:
1. **Warning:** A formal warning issued to your registered email address.
2. **Suspension:** Temporary suspension of your authenticated Quasar account and access tokens.
3. **Termination:** Permanent closure of your account, deletion of your database records, and permanent blocking of your IP address.
4. **Legal Action:** Reporting malicious hacking attempts, sandbox escapes, or DDoS attacks to relevant cybersecurity and academic law enforcement bodies.

---

## 6. Report Violations

If you observe any violations of this policy or suspect a security vulnerability in our sandboxed execution environment, please report it immediately to:
**Email:** `security@quasarassistant.com`
