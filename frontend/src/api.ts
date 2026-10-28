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
