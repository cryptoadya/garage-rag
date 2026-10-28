import type { Claim } from './api';

export function AnswerText({ text, claims }: { text: string; claims: Claim[] }) {
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
                      <span>PDF page {citation.pdf_page}</span>
                      <blockquote>{citation.quote}</blockquote>
                    </div>
                  ))}
                </div>
              </li>
            ))}
          </ol>
          <p className="source-note">Physical PDF pages, counted from 1. Open your own PDF to read the full passage.</p>
        </section>
      ) : null}
    </>
  );
}
