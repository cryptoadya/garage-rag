import { Fragment, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ask } from './api';
import type { Answer, Citation, Language, Mode } from './api';
import { AnswerText } from './AnswerText';
import { Documents } from './Documents';
import { PdfViewer } from './PdfViewer';

type Result =
  | { status: 'idle' | 'loading' }
  | { status: 'success'; answer: Answer; mode: Mode }
  | { status: 'error'; message: string };

const languageNames = { en: 'English', de: 'Deutsch', ru: 'Русский' };

export default function App() {
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<Mode>('grounded');
  const [language, setLanguage] = useState<Language>('auto');
  const [result, setResult] = useState<Result>({ status: 'idle' });
  const inFlight = useRef(false);
  const [uploading, setUploading] = useState(false);
  const [citation, setCitation] = useState<Citation | null>(null);
  const loading = result.status === 'loading';

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inFlight.current || uploading || !question.trim() || question.length > 4000) return;
    inFlight.current = true;
    setResult({ status: 'loading' });
    try {
      const answer = await ask({ question, mode, language });
      setResult({ status: 'success', answer, mode });
    } catch (error) {
      setResult({ status: 'error', message: error instanceof Error ? error.message : 'Request failed.' });
    } finally {
      inFlight.current = false;
    }
  }

  return (
    <Fragment>
      <header className="site-header">
        <a className="brand" href="/" aria-label="GarageRAG home">
          <svg className="brand-icon" viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <rect width="40" height="40" rx="11" fill="currentColor" />
            <path d="M11 29V18l9-7 9 7v11H11m6 0v-9h6v9" stroke="#e4eddf" strokeWidth="1.8" strokeLinejoin="round" />
          </svg>
          Garage<span>RAG</span>
        </a>
        <span className="local-label">Local workspace</span>
      </header>

      <main>
        <div className="intro">
          <p className="eyebrow">YOUR AUTOMOTIVE DOCUMENTATION</p>
          <h1>A little clarity.<br className="mobile-break" /> From your manual.</h1>
          <p>Ask a question. Get an answer with evidence from your documents.</p>
        </div>

        <Documents disabled={loading} onBusyChange={setUploading} />
        <div className="workspace">
          <section className="question-panel" aria-labelledby="question-heading">
            <div className="panel-heading"><span className="step">01</span><h2 id="question-heading">Ask your manual</h2></div>
            <form aria-label="Ask a question" onSubmit={submit}>
              <fieldset disabled={loading || uploading}>
                <label htmlFor="question">Your question</label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="How do I check the engine oil level?"
                  maxLength={4000}
                  required
                  aria-describedby="question-hint"
                />
                <div className="input-hint" id="question-hint"><span>English, Deutsch or Русский</span><span>{question.length} / 4,000</span></div>

                <div className="options">
                  <div>
                    <label htmlFor="mode">Answer mode</label>
                    <select id="mode" value={mode} onChange={(event) => setMode(event.target.value as Mode)} aria-describedby="mode-hint">
                      <option value="grounded">Grounded</option>
                      <option value="direct">Direct</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="language">Answer language</label>
                    <select id="language" value={language} onChange={(event) => setLanguage(event.target.value as Language)}>
                      <option value="auto">Auto · match question</option>
                      <option value="en">English</option>
                      <option value="de">Deutsch</option>
                      <option value="ru">Русский</option>
                    </select>
                  </div>
                </div>
                <p className="mode-hint" id="mode-hint">{mode === 'grounded' ? 'Drafts an answer, then checks it against the evidence.' : 'Answers directly from retrieved evidence.'}</p>
                <button className="submit-button" type="submit" disabled={loading || uploading || !question.trim()}>
                  {loading ? 'Preparing answer…' : 'Ask GarageRAG'}<span aria-hidden="true">↗</span>
                </button>
              </fieldset>
            </form>
            <p className="privacy-note">Retrieval stays local. Your question and selected passages go to Groq.</p>
          </section>

          <section className="answer-panel" aria-labelledby="answer-heading" aria-busy={loading}>
            <div className="panel-heading"><span className="step">02</span><h2 id="answer-heading">The answer</h2><span className="panel-tag">WITH SOURCES</span></div>

            {result.status === 'idle' ? (
              <div className="empty-state">
                <svg viewBox="0 0 100 100" fill="none" aria-hidden="true">
                  <rect x="12" y="12" width="76" height="76" rx="24" fill="#edf1e8" />
                  <path d="M50 34c-8-5-16-6-25-3v34c9-3 17-2 25 3 8-5 16-6 25-3V31c-9-3-17-2-25 3Zm0 0v34M33 41l10 2m-10 7 10 2m14-9 10-2m-10 11 10-2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <h3>Your manual, a question away.</h3>
                <p>Ask about a procedure, a warning light,<br />or everyday care. The evidence appears here.</p>
              </div>
            ) : null}

            <div aria-live="polite" aria-atomic="true">
              {loading ? <div className="loading-state" role="status"><span className="spinner" aria-hidden="true" /><h3>Preparing your answer</h3><p>Retrieving evidence and generating a response.<br />This can take a little time.</p></div> : null}
              {result.status === 'error' ? <div className="error-state" role="alert"><p className="eyebrow">REQUEST FAILED</p><h3>Couldn’t get an answer.</h3><p>{result.message}</p><p>Your question is still here. You can submit it again.</p></div> : null}
              {result.status === 'success' ? (
                <section className="result" aria-label="Answer">
                  <div className="result-meta"><span className={result.answer.supported ? 'answer-badge' : 'refusal-badge'}>{result.answer.supported ? 'Answer with evidence' : 'No supported answer'}</span><span>{result.mode === 'grounded' ? 'Grounded' : 'Direct'} · {languageNames[result.answer.language]}</span></div>
                  <div lang={result.answer.language}><AnswerText text={result.answer.text} claims={result.answer.supported ? result.answer.claims : []} onOpenPdf={setCitation} /></div>
                  {!result.answer.supported ? <p className="refusal-reason">Reason: <code>{result.answer.reason}</code></p> : null}
                </section>
              ) : null}
            </div>
          </section>
        </div>
        <footer className="site-footer"><span>Built around your documents.</span><span>Check the original manual for your vehicle and equipment.</span></footer>
      </main>
      {citation ? <PdfViewer citation={citation} onClose={() => setCitation(null)} /> : null}
    </Fragment>
  );
}
