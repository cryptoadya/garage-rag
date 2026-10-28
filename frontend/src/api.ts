export type Mode = 'direct' | 'grounded';
export type Language = 'auto' | 'en' | 'de' | 'ru';

export interface Answer {
  supported: boolean;
  text: string;
  reason: string;
  language: Exclude<Language, 'auto'>;
  workflow: Record<string, unknown>;
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
    !('workflow' in body) || !body.workflow || typeof body.workflow !== 'object' || Array.isArray(body.workflow)
  ) {
    throw new Error('Unexpected response from the local backend. Check the backend and try again.');
  }
  return body as Answer;
}
