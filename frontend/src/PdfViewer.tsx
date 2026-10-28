import { useEffect, useRef, useState } from 'react';
import type { Citation } from './api';

const unavailable = 'Original PDF unavailable for this citation. Upload the same PDF with the same filename to make it available. Earlier CLI documents may have no saved original.';

export function PdfViewer({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [pdf, setPdf] = useState<{ url: string; pages: number } | null>(null);
  const [error, setError] = useState('');
  const [page, setPage] = useState(citation.pdf_page);

  useEffect(() => { dialog.current?.showModal(); }, []);
  useEffect(() => {
    const controller = new AbortController();
    let url: string | undefined;
    async function load() {
      try {
        if (!citation.pdf_sha256) throw new Error(unavailable);
        const query = new URLSearchParams({ document: citation.document, sha256: citation.pdf_sha256 });
        const response = await fetch(`/api/documents/pdf?${query}`, { signal: controller.signal });
        if (!response.ok) throw new Error(response.status === 404 ? unavailable
          : `Could not open PDF (HTTP ${response.status}). Close the viewer and try again.`);
        const pages = Number(response.headers.get('X-PDF-Pages'));
        if (!Number.isInteger(pages) || pages < citation.pdf_page) throw new Error(unavailable);
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        url = URL.createObjectURL(blob);
        setPdf({ url, pages });
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Could not load PDF.');
      }
    }
    void load();
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [citation]);

  return (
    <dialog ref={dialog} className="pdf-viewer" aria-labelledby="pdf-heading" onCancel={onClose} onClose={onClose}>
      <div className="pdf-heading">
        <h2 id="pdf-heading">{citation.document}</h2>
        <button className="secondary-button" onClick={onClose}>Close PDF</button>
      </div>
      {error ? <p role="alert">{error}</p> : !pdf ? <p role="status">Loading original PDF…</p> : (
        <>
          <nav className="pdf-controls" aria-label="PDF pages">
            <button className="secondary-button" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous page</button>
            <label>PDF page <input type="number" min="1" max={pdf.pages} value={page} onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isInteger(value) && value >= 1 && value <= pdf.pages) setPage(value);
            }} /></label>
            <span>of {pdf.pages}</span>
            <button className="secondary-button" disabled={page >= pdf.pages} onClick={() => setPage(page + 1)}>Next page</button>
            <a href={`${pdf.url}#page=${page}`} target="_blank" rel="noreferrer">Open PDF in new tab</a>
          </nav>
          <iframe key={page} title="Original PDF" src={`${pdf.url}#page=${page}`} />
          <p className="source-note">Physical PDF pages, counted from 1. If the preview is unavailable in your browser, open the PDF in a new tab.</p>
        </>
      )}
    </dialog>
  );
}
