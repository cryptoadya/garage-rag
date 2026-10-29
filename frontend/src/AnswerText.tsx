import type { Citation, Claim } from './api';

function pageKey(citation: Citation) {
  return JSON.stringify([citation.document, citation.pdf_sha256 ?? null, citation.pdf_page]);
}

export function AnswerText({ text, claims, onOpenPdf }: { text: string; claims: Claim[]; onOpenPdf: (citation: Citation) => void }) {
  if (!claims.length) return <div className="answer-text">{text}</div>;

  const pages = new Map<string, { citation: Citation; quotes: Set<string> }>();
  const documents = new Map<string, number>();
  for (const claim of claims) {
    for (const citation of claim.citations) {
      const document = JSON.stringify([citation.document, citation.pdf_sha256 ?? null]);
      if (!documents.has(document)) documents.set(document, documents.size + 1);
      const key = pageKey(citation);
      const page = pages.get(key) ?? { citation, quotes: new Set<string>() };
      page.quotes.add(citation.quote);
      pages.set(key, page);
    }
  }

  return (
    <>
      <div className="answer-text">
        {claims.map((claim, index) => (
          <div className="answer-claim" key={index}>
            <p>{claim.text}</p>
            <span className="claim-references">
              {[...new Set(claim.citations.map(pageKey))].map((key) => {
                const { citation } = pages.get(key)!;
                return <button className="citation-link" key={key}
                  onClick={() => onOpenPdf(citation)}
                  title={`${citation.document} · PDF page ${citation.pdf_page}`}
                  aria-label={`Open ${citation.document} at PDF page ${citation.pdf_page}`}>
                  PDF{documents.size > 1 ? ` ${documents.get(JSON.stringify([citation.document, citation.pdf_sha256 ?? null]))}` : ''} · {citation.pdf_page} <span aria-hidden="true">↗</span>
                </button>;
              })}
            </span>
          </div>
        ))}
      </div>
      <section className="sources" aria-label="Sources">
        <details>
          <summary>Sources <span>{pages.size}</span></summary>
          <div className="source-pages">
            {[...pages.entries()].map(([key, { citation, quotes }]) => (
              <div className="source-page" key={key}>
                <div className="source-page-heading">
                  <span>PDF page {citation.pdf_page}</span>
                  <span className="source-document" title={citation.document}>{citation.document}</span>
                </div>
                {[...quotes].map((quote) => <blockquote key={quote}>{quote}</blockquote>)}
              </div>
            ))}
          </div>
          <p className="source-note">Page numbers refer to physical PDF pages. Select a PDF reference above to open the original.</p>
        </details>
      </section>
    </>
  );
}
