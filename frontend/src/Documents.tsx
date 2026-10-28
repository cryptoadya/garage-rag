import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { listDocuments, uploadPdf } from './api';
import type { IndexedDocument, UploadPhase } from './api';

type UploadState =
  | { phase: 'idle' }
  | { phase: UploadPhase; percent?: number }
  | { phase: 'ready'; reused: boolean }
  | { phase: 'error'; message: string };

export function Documents({ disabled, onBusyChange }: {
  disabled: boolean; onBusyChange: (busy: boolean) => void;
}) {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadState>({ phase: 'idle' });
  const input = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false);
  const busy = upload.phase === 'uploading' || upload.phase === 'indexing';

  useEffect(() => {
    let active = true;
    listDocuments().then((items) => { if (active) setDocuments(items); })
      .catch((error: Error) => { if (active) setListError(error.message); })
      .finally(() => { if (active) setListLoading(false); });
    return () => { active = false; };
  }, []);

  async function refresh() {
    if (listLoading || busy || disabled) return;
    setListLoading(true);
    setListError('');
    try { setDocuments(await listDocuments()); }
    catch (error) { setListError(error instanceof Error ? error.message : 'Could not load documents.'); }
    finally { setListLoading(false); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || inFlight.current || disabled || listLoading) return;
    inFlight.current = true;
    onBusyChange(true);
    setUpload({ phase: 'uploading', percent: 0 });
    try {
      const result = await uploadPdf(file, (phase, percent) => setUpload({ phase, percent }));
      setDocuments((previous) => [...previous.filter((item) => item.document !== result.document.document),
        result.document].sort((a, b) => a.document.localeCompare(b.document)));
      setUpload({ phase: 'ready', reused: result.status === 'unchanged' });
      setFile(null);
      if (input.current) input.current.value = '';
    } catch (error) {
      setUpload({ phase: 'error', message: error instanceof Error ? error.message : 'PDF upload failed.' });
    } finally {
      inFlight.current = false;
      onBusyChange(false);
    }
  }

  return (
    <section className="documents-panel" aria-labelledby="documents-heading">
      <div className="documents-heading">
        <h2 id="documents-heading">Your PDFs</h2>
        <button className="secondary-button" type="button" onClick={refresh} disabled={disabled || busy || listLoading}>Refresh documents</button>
      </div>
      <p className="document-hint">Upload a PDF with selectable text, up to 50 MiB. It stays in private local storage.</p>
      <form aria-label="Upload PDF" onSubmit={submit}>
        <fieldset disabled={disabled || busy || listLoading} className="upload-controls">
          <div><label htmlFor="pdf-file">Choose PDF</label><input ref={input} id="pdf-file" type="file" accept=".pdf,application/pdf" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setUpload({ phase: 'idle' }); }} /></div>
          <button className="submit-button" type="submit" disabled={disabled || busy || listLoading || !file}>Upload and index PDF</button>
        </fieldset>
      </form>
      {busy ? <p className="document-hint" role="status">{upload.phase === 'indexing' ? 'Indexing PDF… You can ask questions when it is ready.' : `Uploading PDF…${upload.percent === undefined ? '' : ` ${upload.percent}%`}`}</p> : null}
      {upload.phase === 'ready' ? <p className="document-hint" role="status">{upload.reused ? 'Already indexed. Ready for questions.' : 'Ready for questions. PDF indexed.'}</p> : null}
      {upload.phase === 'error' ? <p className="document-error" role="alert">{upload.message}</p> : null}
      {listError ? <p className="document-error" role="alert">{listError}</p> : null}
      {listLoading ? <p className="document-hint">Loading documents…</p> : documents.length ? (
        <ul className="document-list">{documents.map((item) => (
          <li key={item.document}><span>{item.document}</span><span>{item.pages} PDF pages · {item.chunks} chunks · Indexed</span></li>
        ))}</ul>
      ) : !listError ? <p className="document-hint">No indexed documents yet. Upload a PDF to start.</p> : null}
    </section>
  );
}
