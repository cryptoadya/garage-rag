import type { Citation, Claim } from './api';

export function AnswerText({ text, claims, onOpenPdf }: { text: string; claims: Claim[]; onOpenPdf: (citation: Citation) => void }) {
  return (
    <>
      <div className="answer-text">{text}</div>
      {claims.length ? (
        <section className="sources" aria-labelledby="sources-heading">
          <h3 id="sources-heading">Sources by claim <span>{claims.length}</span></h3>
          <ol>
            {claims.map((claim, index) => (
              <li key={index}>
                <span className="source-number">{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <p>{claim.text}</p>
                  {claim.citations.map((citation, citationIndex) => (
                    <div className="citation" key={citationIndex}>
                      <p>{citation.document}</p>
                      <button className="citation-link" onClick={() => onOpenPdf(citation)} aria-label={`Open ${citation.document} at PDF page ${citation.pdf_page}`}>PDF page {citation.pdf_page} ↗</button>
                      <blockquote>{citation.quote}</blockquote>
                    </div>
                  ))}
                </div>
              </li>
            ))}
          </ol>
          <p className="source-note">Physical PDF pages, counted from 1. Select a page to view the saved original PDF.</p>
        </section>
      ) : null}
    </>
  );
}
