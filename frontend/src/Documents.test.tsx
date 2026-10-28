import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

class UploadRequest extends EventTarget {
  static requests: UploadRequest[] = [];
  upload = new EventTarget();
  status = 0;
  responseText = '';
  method = '';
  url = '';
  headers: Record<string, string> = {};
  body: unknown;
  constructor() { super(); UploadRequest.requests.push(this); }
  open(method: string, url: string) { this.method = method; this.url = url; }
  setRequestHeader(name: string, value: string) { this.headers[name] = value; }
  send(body: unknown) { this.body = body; }
  finish(body: unknown, status = 200) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.dispatchEvent(new Event('load'));
  }
}

const document = { document: 'manual [EN].pdf', pages: 3, chunks: 8 };

beforeEach(() => {
  UploadRequest.requests = [];
  vi.stubGlobal('XMLHttpRequest', UploadRequest);
  vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => Promise.resolve(
    new Response(JSON.stringify(url === '/api/documents' ? { documents: [] } : {
      supported: true, text: 'Check the tires when they are cold.', reason: 'verified_citations',
      language: 'en', workflow: {}, claims: [{ text: 'Check the tires when they are cold.',
        citations: [{ source_id: 'S1', document: document.document, pdf_page: 2,
          quote: 'Check tire pressure when the tires are cold.' }] }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
  )));
});

async function selectAndUpload() {
  const user = userEvent.setup();
  await screen.findByText('No indexed documents yet. Upload a PDF to start.');
  const file = new File(['%PDF-1.7\nfictional PDF'], document.document, { type: 'application/pdf' });
  await user.upload(screen.getByLabelText('Choose PDF'), file);
  await user.click(screen.getByRole('button', { name: 'Upload and index PDF' }));
  return { file, user, request: UploadRequest.requests[0] };
}

describe('PDF management', () => {
  it('lists active documents from the backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ documents: [document] }))));
    render(<App />);
    const panel = screen.getByRole('region', { name: 'Your PDFs' });
    expect(await within(panel).findByText(document.document)).toBeVisible();
    expect(panel).toHaveTextContent('3 PDF pages');
    expect(panel).toHaveTextContent('8 chunks');
    expect(panel).toHaveTextContent('Indexed');
  });

  it('uploads once, shows transfer/indexing, then immediately allows questions with citations', async () => {
    render(<App />);
    const { file, user, request } = await selectAndUpload();
    expect(request.method).toBe('POST');
    expect(request.url).toBe('/api/documents?filename=manual%20%5BEN%5D.pdf');
    expect(request.headers['Content-Type']).toBe('application/pdf');
    expect(request.body).toBe(file);
    expect(screen.getByRole('status')).toHaveTextContent('Uploading');
    expect(screen.getByLabelText('Your question')).toBeDisabled();
    fireEvent.submit(screen.getByRole('form', { name: 'Upload PDF' }));
    expect(UploadRequest.requests).toHaveLength(1);
    act(() => request.upload.dispatchEvent(new ProgressEvent('progress', {
      loaded: 5, total: 10, lengthComputable: true,
    })));
    expect(screen.getByRole('status')).toHaveTextContent('50%');
    act(() => request.upload.dispatchEvent(new Event('load')));
    expect(screen.getByRole('status')).toHaveTextContent('Indexing');
    expect(screen.getByRole('button', { name: 'Ask GarageRAG' })).toBeDisabled();
    await act(async () => request.finish({ status: 'indexed', document }));
    expect(screen.getByRole('status')).toHaveTextContent('Ready');
    expect(within(screen.getByRole('region', { name: 'Your PDFs' })).getByText(document.document)).toBeVisible();
    await user.type(screen.getByLabelText('Your question'), 'How do I check tires?');
    await user.click(screen.getByRole('button', { name: 'Ask GarageRAG' }));
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result).toHaveTextContent(document.document);
    expect(result).toHaveTextContent('PDF page 2');
  });

  it.each([422, 500, 503])('shows upload HTTP %s and allows retry', async (status) => {
    render(<App />);
    const { user, request } = await selectAndUpload();
    await act(async () => request.finish({ detail: 'Could not index PDF.' }, status));
    expect(screen.getByRole('alert')).toHaveTextContent(`HTTP ${status}`);
    expect(screen.getByRole('alert')).toHaveTextContent('Could not index PDF.');
    expect(screen.getByLabelText('Your question')).toBeEnabled();
    await user.click(screen.getByRole('button', { name: 'Upload and index PDF' }));
    await act(async () => UploadRequest.requests[1].finish({ status: 'unchanged', document }));
    expect(screen.getByRole('status')).toHaveTextContent('Already indexed');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('reports network failure without a retry or a false success', async () => {
    render(<App />);
    const { request } = await selectAndUpload();
    await act(async () => request.dispatchEvent(new Event('error')));
    expect(screen.getByRole('alert')).toHaveTextContent('Refresh documents');
    expect(UploadRequest.requests).toHaveLength(1);
    expect(screen.getByLabelText('Your question')).toBeEnabled();
  });

  it.each([
    new File(['plain'], 'notes.txt', { type: 'text/plain' }),
    new File([], 'empty.pdf', { type: 'application/pdf' }),
  ])('rejects invalid selections before uploading', async (file) => {
    render(<App />);
    await screen.findByText('No indexed documents yet. Upload a PDF to start.');
    const user = userEvent.setup({ applyAccept: false });
    await user.upload(screen.getByLabelText('Choose PDF'), file);
    await user.click(screen.getByRole('button', { name: 'Upload and index PDF' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('PDF');
    expect(UploadRequest.requests).toHaveLength(0);
  });

  it('reports a list error and refreshes it on request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValueOnce(new TypeError('offline'))
      .mockResolvedValue(new Response(JSON.stringify({ documents: [document] }))));
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('documents');
    await userEvent.click(screen.getByRole('button', { name: 'Refresh documents' }));
    expect(await screen.findByText(document.document)).toBeVisible();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
