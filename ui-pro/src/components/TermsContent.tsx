/** Terms of use and privacy notice. Shared by /terms (what people accept at
 * sign-up) and the Terms tab on /help, so the two can never disagree.
 *
 * Keep every statement true to the code. When data handling changes (what is
 * stored, who receives it, what deletion removes), update this file and
 * TERMS_UPDATED in the same change. */

import type { LucideIcon } from "lucide-react";
import {
    Shield, Sparkles, Eye, Share2, Activity, Brain, Trash2, FileText, Lock, Scale, RefreshCw, HelpCircle,
} from "lucide-react";

export const TERMS_UPDATED = "October 4, 2026";

const REPO_URL = "https://github.com/adamzacharia/Quasar";

export interface TermsSection {
    id: string;
    title: string;
    icon: LucideIcon;
    body: React.ReactNode;
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <li>
            <strong className="font-medium text-[var(--q-text)]">{label}:</strong> {children}
        </li>
    );
}

const LIST = "list-disc space-y-2 pl-5";

export const TERMS_SECTIONS: TermsSection[] = [
    {
        id: "using",
        title: "1. Using Quasar",
        icon: Shield,
        body: (
            <>
                <p>
                    Quasar is a free AI research assistant for astronomy. You need to sign in to use it (there is no
                    guest access), and you must be at least 13 years old. If we learn an account belongs to someone
                    under 13, we will close it and delete its data.
                </p>
                <p>Use Quasar for lawful research and education. Please do not:</p>
                <ul className={LIST}>
                    <li>try to get around the access controls of any archive Quasar connects to, including proprietary data periods;</li>
                    <li>send automated or bulk traffic that slows the service down for other people;</li>
                    <li>use data in ways the source archive&apos;s own policies do not allow;</li>
                    <li>present Quasar&apos;s output as peer-reviewed or independently verified work.</li>
                </ul>
                <p>
                    Each account gets a weekly token allowance on the models Quasar provides. Models you run on your
                    own API key are limited only by that key and any limit you set in Settings.
                </p>
            </>
        ),
    },
    {
        id: "ai",
        title: "2. AI-generated answers",
        icon: Sparkles,
        body: (
            <>
                <p>
                    Quasar&apos;s answers are written by large language models. They can be wrong, out of date or
                    incomplete, even when they cite a source. Check anything important against the original archive
                    record or paper before you rely on it, and especially before you publish it.
                </p>
                <p>
                    Quasar does not replace your scientific judgment. The Research timeline under each answer shows
                    which tools ran and what they returned, so you can see where a result came from.
                </p>
            </>
        ),
    },
    {
        id: "collect",
        title: "3. What we collect",
        icon: Eye,
        body: (
            <>
                <p>We collect what Quasar needs to work, to keep your history, and to fix problems:</p>
                <ul className={LIST}>
                    <Item label="Account">your name, email address and sign-in method (password or Google).</Item>
                    <Item label="Chats">your full prompts and Quasar&apos;s responses, including the tools each answer used.</Item>
                    <Item label="Files you attach">images, PDFs, FITS, tables and other documents you upload to a chat.</Item>
                    <Item label="Things you save">saved papers, Data Lab tables and jobs, knowledge-base documents, and your memory profile.</Item>
                    <Item label="Keys and connections">API keys you add and MCP servers you connect. Keys and server secrets are stored encrypted, and we only ever show you the last four characters of a key.</Item>
                    <Item label="Feedback">ratings, thumbs up or down, and issue reports. If you choose to attach the conversation to a report, a copy of it is kept with the report.</Item>
                    <Item label="Usage and technical data">your IP address, browser type, approximate country, token usage, response times and errors.</Item>
                </ul>
            </>
        ),
    },
    {
        id: "share",
        title: "4. Services that receive your data",
        icon: Share2,
        body: (
            <>
                <p>To answer you, Quasar sends parts of your request to other services:</p>
                <ul className={LIST}>
                    <Item label="AI model providers">your prompt and conversation go to the provider of the model you pick. The default model runs at the Texas Advanced Computing Center (TACC). Other models go to DeepSeek, OpenAI, Anthropic or Google. Documents you attach may be uploaded to that provider&apos;s file storage, and images sent to a model that cannot read images are first described by an OpenAI model.</Item>
                    <Item label="Your knowledge base">documents you add in Settings are indexed with OpenAI embeddings and stored in a private, per-user search index.</Item>
                    <Item label="Astronomy archives">target names, coordinates and query parameters go to the archives Quasar searches, such as ALMA, NRAO, NASA ADS, SIMBAD, NOIRLab Astro Data Lab, MAST, IRSA and others.</Item>
                    <Item label="Web search">when web search is on, search queries go to Brave Search, Tavily or Exa.</Item>
                    <Item label="MCP servers you connect">these receive whatever Quasar sends their tools during your chats. Only connect servers you trust.</Item>
                    <Item label="Performance tracing">we use Langfuse to trace requests so we can find and fix slow or failing steps.</Item>
                </ul>
                <p>Each of these services handles data under its own terms. We do not sell your data.</p>
            </>
        ),
    },
    {
        id: "use",
        title: "5. How we use your data",
        icon: Activity,
        body: (
            <>
                <p>
                    We use your data to run Quasar, keep your chat history, personalize answers, fix bugs and measure
                    performance. The project team can see usage analytics and the feedback reports you send.
                </p>
                <p>
                    <strong className="font-medium text-[var(--q-text)]">Improving Quasar:</strong> anything you share
                    with Quasar, including queries, conversations, uploaded documents and search parameters, may be used
                    to evaluate, train, fine-tune or otherwise improve Quasar and its AI models.
                </p>
            </>
        ),
    },
    {
        id: "memory",
        title: "6. Memory and personalization",
        icon: Brain,
        body: (
            <>
                <p>
                    Memory is on by default. Quasar learns research details from your chats (for example your field,
                    favorite facilities, bands and targets) and uses them to tailor later answers. Earlier versions of
                    a memory are kept until you delete them.
                </p>
                <p>
                    In Settings &gt; Memory you can see, edit, export, pause or clear everything Quasar remembers about
                    you.
                </p>
            </>
        ),
    },
    {
        id: "control",
        title: "7. Deleting your data",
        icon: Trash2,
        body: (
            <>
                <p>
                    You can delete chats from the sidebar, and remove saved papers, Data Lab tables, knowledge-base
                    documents, memory, API keys and MCP servers at any time.
                </p>
                <p>
                    Deleting a chat removes it and its messages from your history. Some copies stay behind: usage
                    analytics and performance traces, any feedback report you attached the chat to, and files already
                    sent to a model provider, which that provider keeps under its own retention rules.
                </p>
                <p>
                    To delete your whole account and the data tied to it, contact the project team through the{" "}
                    <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-[var(--color-primary)] underline underline-offset-2">GitHub repository</a>.
                </p>
            </>
        ),
    },
    {
        id: "yours",
        title: "8. Your research is yours",
        icon: FileText,
        body: (
            <>
                <p>
                    We claim no ownership of your questions, files or findings. Other users cannot see your chats,
                    files or saved items.
                </p>
                <p>
                    Data you retrieve through Quasar still belongs to the archives it came from and is covered by their
                    data-use and citation policies. When you publish, acknowledge the facilities and archives you used
                    (ALMA, NRAO, NOIRLab, MAST and so on) the way each one asks.
                </p>
            </>
        ),
    },
    {
        id: "security",
        title: "9. Security",
        icon: Lock,
        body: (
            <p>
                We encrypt stored API keys and MCP secrets and take reasonable care to protect your data. No online
                service is perfectly secure, so please do not upload anything you cannot afford to have exposed, such
                as passwords or confidential proposals you are not permitted to share.
            </p>
        ),
    },
    {
        id: "liability",
        title: "10. No warranty",
        icon: Scale,
        body: (
            <p>
                Quasar is a research project provided &quot;as is&quot;, without warranties of any kind. It may be
                unavailable at times, and features may change or be removed. We are not liable for any direct or
                indirect loss arising from your use of Quasar, including lost data or wrong results.
            </p>
        ),
    },
    {
        id: "changes",
        title: "11. Changes to these terms",
        icon: RefreshCw,
        body: (
            <p>
                We may update these terms. The date at the top shows when they last changed, and we will flag
                material changes in the app. Continuing to use Quasar after a change means you accept the updated
                terms.
            </p>
        ),
    },
    {
        id: "contact",
        title: "12. Contact",
        icon: HelpCircle,
        body: (
            <p>
                Questions, requests or problems: open an issue on the{" "}
                <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-[var(--color-primary)] underline underline-offset-2">GitHub repository</a>{" "}
                or use the thumbs-down &quot;Report a problem&quot; button on any answer.
            </p>
        ),
    },
];

/** The full terms as a list of cards (no page chrome). */
export function TermsContent() {
    return (
        <div className="space-y-8">
            {TERMS_SECTIONS.map(({ id, title, icon: Icon, body }) => (
                <section key={id} id={id} className="scroll-mt-24 space-y-3">
                    <div className="flex items-center gap-3">
                        <div className="flex size-8 items-center justify-center rounded-lg bg-[var(--q-canvas)] text-[var(--q-text-muted)]">
                            <Icon className="h-4 w-4" strokeWidth={1.75} />
                        </div>
                        <h2 className="text-[17px] font-medium text-[var(--q-text)]">{title}</h2>
                    </div>
                    <div className="glass-card space-y-3 rounded-2xl p-5 text-[13.5px] leading-relaxed text-[var(--q-text-secondary)]">
                        {body}
                    </div>
                </section>
            ))}
        </div>
    );
}
