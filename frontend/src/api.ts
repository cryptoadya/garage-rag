export type Mode = 'direct' | 'grounded';
export type Language = 'auto' | 'en' | 'de' | 'ru';

export interface Citation {
  source_id: string;
  document: string;
  pdf_page: number;
  quote: string;
}

export interface Claim {
  text: string;
  citations: Citation[];
}

export interface Answer {
  supported: boolean;
  text: string;
  reason: string;
  language: Exclude<Language, 'auto'>;
  workflow: Record<string, unknown>;
  claims: Claim[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

export interface IndexedDocument {
  document: string;
  pages: number;
  chunks: number;
}

export type UploadPhase = 'uploading' | 'indexing';

function isDocument(value: unknown): value is IndexedDocument {
  return isRecord(value) && typeof value.document === 'string' &&
    typeof value.pages === 'number' && Number.isInteger(value.pages) && value.pages > 0 &&
    typeof value.chunks === 'number' && Number.isInteger(value.chunks) && value.chunks > 0;
}

export async function listDocuments(): Promise<IndexedDocument[]> {
  try {
    const response = await fetch('/api/documents');
    const body: unknown = await response.json();
    if (response.ok && isRecord(body) && Array.isArray(body.documents) && body.documents.every(isDocument)) {
      return body.documents;
    }
  } catch { /* Show a fixed message for network and malformed response errors. */ }
  throw new Error('Could not load documents. Check the backend and refresh documents.');
}

export function uploadPdf(
  file: File, onProgress: (phase: UploadPhase, percent?: number) => void,
): Promise<{ status: 'indexed' | 'unchanged'; document: IndexedDocument }> {
  return new Promise((resolve, reject) => {
    if (!file.name.toLowerCase().endsWith('.pdf') || file.size === 0 || file.size > 50 * 1024 * 1024) {
      reject(new Error('Choose a nonempty PDF up to 50 MiB.'));
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/documents?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Content-Type', 'application/pdf');
    xhr.upload.addEventListener('progress', (event) => {
      onProgress('uploading', event.lengthComputable ? Math.round(event.loaded / event.total * 100) : undefined);
    });
    xhr.upload.addEventListener('load', () => onProgress('indexing'));
    const disconnected = () => reject(new Error(
      'Upload connection lost. Refresh documents to check whether indexing completed before retrying.',
    ));
    xhr.addEventListener('error', disconnected);
    xhr.addEventListener('abort', disconnected);
    xhr.addEventListener('load', () => {
      let body: unknown;
      try { body = JSON.parse(xhr.responseText); } catch { body = null; }
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = isRecord(body) && typeof body.detail === 'string' ? body.detail : 'PDF upload failed.';
        reject(new Error(`HTTP ${xhr.status} · ${detail}`));
      } else if (isRecord(body) && (body.status === 'indexed' || body.status === 'unchanged') && isDocument(body.document)) {
        resolve({ status: body.status, document: body.document });
      } else {
        reject(new Error('Unexpected upload response. Refresh documents before retrying.'));
      }
    });
    onProgress('uploading', 0);
    xhr.send(file);
  });
}

function isCitation(value: unknown): value is Citation {
  return isRecord(value) && typeof value.source_id === 'string' &&
    typeof value.document === 'string' && typeof value.quote === 'string' &&
    typeof value.pdf_page === 'number' && Number.isInteger(value.pdf_page) && value.pdf_page >= 1;
}

function isClaim(value: unknown): value is Claim {
  return isRecord(value) && typeof value.text === 'string' &&
    Array.isArray(value.citations) && value.citations.length > 0 && value.citations.every(isCitation);
}

export async function ask(request: { question: string; mode: Mode; language: Language }): Promise<Answer> {
  let response: Response;
  try {
    response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error('Check that the local backend is running, then try again.');
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string'
      ? body.detail : 'Check that the local backend is running, then try again.';
    throw new Error(`HTTP ${response.status} · ${detail}`);
  }
  if (
    !body || typeof body !== 'object' ||
    !('supported' in body) || typeof body.supported !== 'boolean' ||
    !('text' in body) || typeof body.text !== 'string' ||
    !('reason' in body) || typeof body.reason !== 'string' ||
    !('language' in body) || !['en', 'de', 'ru'].includes(String(body.language)) ||
    !('workflow' in body) || !isRecord(body.workflow) ||
    !('claims' in body) || !Array.isArray(body.claims) || !body.claims.every(isClaim)
  ) {
    throw new Error('Unexpected response from the local backend. Check the backend and try again.');
  }
  return body as Answer;
}
