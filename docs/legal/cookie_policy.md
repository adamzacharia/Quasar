# Cookie Policy
**Effective Date: May 24, 2026**
**Version: 2.0**

Quasar Observatory ("we," "us," or "our") uses cookies and similar tracking technologies on the QUASAR Astronomical Assistant platform. This Cookie Policy explains what cookies are, how we use them, and how you can manage your preferences.

---

## 1. What Are Cookies?

Cookies are small text files stored on your computer or mobile device by your web browser when you visit a website. They help the website recognize your device, maintain your login state, and remember your visual preferences.
* **First-Party Cookies:** Set directly by our domain (`quasarassistant.com`) to manage system sessions and theme preferences.
* **Third-Party Cookies:** Set by partner services, such as Google SSO for federated login.

---

## 2. How We Use Cookies

We use cookies strictly to keep Quasar functioning, secure, and user-friendly. We do **not** use cookies for third-party advertising, retargeting, or profile-brokering.

We classify our cookies into the following categories:

### 2.1 Essential Cookies (Strictly Necessary)
These cookies are required to authenticate your user session and securely route API requests. Without these cookies, the system cannot function:
* **JWT Token Cache:** We store your encrypted session JSON Web Token (JWT) in local storage or secure HTTP-only cookies to keep you logged in to the Quasar workspace.
* **CSRF Protection:** Secure tokens used to prevent Cross-Site Request Forgery attacks on our FastAPI backend.

### 2.2 Functional Preferences (UX Optimization)
These cookies help us personalize the workspace visual styling:
* **Theme Preference:** Remembers whether you use Dark Mode or Light Mode styling (`quasar_theme` stored in `localStorage` to prevent screen flashes during page hydrations).
* **Onboarding Tutorial Progress:** Remembers whether you have already completed or dismissed our 5-step interactive workspace walkthrough so it doesn't pop up repeatedly on every visit.

### 2.3 Observability and Telemetry Metrics
We use tracking tokens to monitor query latency and debug system bottlenecks in real-time:
* **Langfuse Sessions:** Secure tracing identifiers linked to your active session to correlate multi-stage Conductor subtask DAG paths and generate latency dashboards.
* **Analytics Page Hit Telemetry:** Captures anonymous screen sizes, browser versions, and regional hit distributions strictly to optimize our front-end assets.

---

## 3. Managing Your Cookie Settings

You have full control over how your browser handles cookies. You can configure your settings through your browser's options menu:
* **Block All Cookies:** You can disable all cookies, but doing so will prevent you from authenticating, logging in, or accessing the Quasar chat workspace.
* **Clear Storage:** You can clear your browser's cookies, session storage, and `localStorage` at any time to reset theme states and log out of the active platform.

---

## 4. Contact Us

If you have any questions about our use of cookies or tracking technologies, please contact us at:
**Email:** `security@quasarassistant.com`
