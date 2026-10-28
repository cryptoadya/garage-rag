import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App';

const answer = {
  supported: true,
  text: 'Check the tires when they are cold.\n[aurora.pdf, PDF page 1] "Check tire pressure when the tires are cold."',
  reason: 'verified_citations',
  language: 'en',
  workflow: { mode: 'grounded', events: [], elapsed_ms: 1 },
  claims: [{
    text: 'Check the tires when they are cold.',
    citations: [{ source_id: 'S1', document: 'aurora.pdf', pdf_page: 1,
      quote: 'Check tire pressure when the tires are cold.' }],
  }],
};

function respond(body: unknown = answer, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function submit(question = 'How do I check tires?') {
  const user = userEvent.setup();
  await user.type(screen.getByRole('textbox', { name: 'Your question' }), question);
  await user.click(screen.getByRole('button', { name: 'Ask GarageRAG' }));
}

describe('Ask your manual', () => {
  it('does not request on load or submit blank input', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    expect(screen.getByRole('button', { name: 'Ask GarageRAG' })).toBeDisabled();
    await userEvent.type(screen.getByRole('textbox', { name: 'Your question' }), '   ');
    fireEvent.submit(screen.getByRole('form', { name: 'Ask a question' }));
    expect(fetch).not.toHaveBeenCalled();
  });

  it('sends only the API fields and displays structured claim citations', async () => {
    const fetch = vi.fn().mockResolvedValue(respond());
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    await submit();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: 'How do I check tires?', mode: 'grounded', language: 'auto' }),
    });
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result).toHaveTextContent('Check the tires when they are cold.');
    expect(result).toHaveTextContent('"Check tire pressure when the tires are cold."');
    const sources = within(result).getByRole('region', { name: /Sources/ });
    expect(sources).toHaveTextContent('Check the tires when they are cold.');
    expect(sources).toHaveTextContent('aurora.pdf');
    expect(sources).toHaveTextContent('PDF page 1');
    expect(sources).toHaveTextContent('Check tire pressure when the tires are cold.');
    expect(result.querySelector('.answer-text')?.textContent).toBe(answer.text);
  });

  it.each(['auto', 'en', 'de', 'ru'])('forwards Direct mode and %s language', async (language) => {
    const fetch = vi.fn().mockResolvedValue(respond());
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    await userEvent.selectOptions(screen.getByLabelText('Answer mode'), 'direct');
    await userEvent.selectOptions(screen.getByLabelText('Answer language'), language);
    await submit('Как проверить шины?');
    await screen.findByRole('region', { name: 'Answer' });
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      question: 'Как проверить шины?', mode: 'direct', language,
    });
  });

  it('keeps one request in flight and exposes loading state', async () => {
    let resolve!: (response: Response) => void;
    const fetch = vi.fn(() => new Promise<Response>((done) => { resolve = done; }));
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    await submit();
    expect(screen.getByRole('status')).toHaveTextContent('Preparing your answer');
    expect(screen.getByRole('textbox', { name: 'Your question' })).toBeDisabled();
    expect(screen.getByLabelText('Answer mode')).toBeDisabled();
    expect(screen.getByLabelText('Answer language')).toBeDisabled();
    fireEvent.submit(screen.getByRole('form', { name: 'Ask a question' }));
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => resolve(respond()));
    expect(await screen.findByRole('region', { name: 'Answer' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Ask GarageRAG' })).toBeEnabled();
  });

  it('shows a valid refusal as an answer, with the returned language and reason', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(respond({
      ...answer, supported: false, text: 'Недостаточно данных.', reason: 'no_context', language: 'ru',
    })));
    render(<App />);
    await submit();
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result).toHaveTextContent('No supported answer');
    expect(result).toHaveTextContent('Недостаточно данных.');
    expect(result).toHaveTextContent('no_context');
    expect(result.querySelector('[lang="ru"]')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(within(result).queryByRole('region', { name: /Sources/ })).not.toBeInTheDocument();
  });

  it.each([422, 500, 502, 503])('shows HTTP %s errors without losing the question, then recovers', async (status) => {
    const fetch = vi.fn().mockResolvedValueOnce(respond({ detail: 'Server is busy.' }, status))
      .mockResolvedValueOnce(respond());
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    await submit();
    expect(await screen.findByRole('alert')).toHaveTextContent(`HTTP ${status}`);
    expect(screen.getByRole('alert')).toHaveTextContent('Server is busy.');
    expect(screen.getByRole('textbox', { name: 'Your question' })).toHaveValue('How do I check tires?');
    await userEvent.click(screen.getByRole('button', { name: 'Ask GarageRAG' }));
    expect(await screen.findByRole('region', { name: 'Answer' })).toBeVisible();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('reports an unavailable backend without an automatic retry', async () => {
    const fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetch);
    render(<App />);
    await submit();
    expect(await screen.findByRole('alert')).toHaveTextContent('Check that the local backend is running');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it.each([
    new Response('<html>proxy error</html>', { status: 502 }),
    respond({ ...answer, text: null }),
    respond({ ...answer, claims: null }),
    respond({ ...answer, claims: [{ text: 'Claim', citations: [{ document: 'a.pdf' }] }] }),
    respond({ ...answer, claims: [{ ...answer.claims[0], citations: [
      { ...answer.claims[0].citations[0], pdf_page: 0 },
    ] }] }),
  ])('handles a non-JSON or malformed response without crashing', async (response) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    render(<App />);
    await submit();
    expect(await screen.findByRole('alert')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Ask GarageRAG' })).toBeEnabled();
  });

  it('keeps response markup inert and preserves multiline text and repeated citations', async () => {
    const text = '<script>alert(1)</script>\n[aurora.pdf, PDF page 1] "First line\nsecond line"\n\nAnother claim.\n[aurora.pdf, PDF page 1] "Another quote"';
    const citation = { source_id: 'S1', document: 'manual [EN], <img src=x>.pdf',
      pdf_page: 7, quote: 'First line\nsecond line <script>alert(1)</script>' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(respond({ ...answer, text, claims: [
      { text: 'First claim.', citations: [citation, { ...citation, source_id: 'S2',
        document: 'other.pdf', pdf_page: 2, quote: 'Another quote' }] },
      { text: 'Another claim.', citations: [citation] },
    ] })));
    render(<App />);
    await submit();
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result).toHaveTextContent('<script>alert(1)</script>');
    expect(result.querySelector('script')).toBeNull();
    expect(result.querySelector('img')).toBeNull();
    expect(result.querySelector('.answer-text')?.textContent).toBe(text);
    const sources = within(result).getByRole('region', { name: /Sources/ });
    const claims = within(sources).getAllByRole('listitem');
    expect(claims).toHaveLength(2);
    expect(claims[0]).toHaveTextContent('First claim.');
    expect(claims[0]).toHaveTextContent('other.pdf');
    expect(claims[0]).toHaveTextContent('PDF page 2');
    expect(claims[1]).toHaveTextContent('Another claim.');
    expect(claims[1]).not.toHaveTextContent('other.pdf');
    expect(claims[1]).toHaveTextContent('PDF page 7');
    expect(claims[1].querySelector('blockquote')?.textContent).toBe(citation.quote);
    expect(sources).not.toHaveTextContent('aurora.pdf');
  });

  it('does not infer sources from answer text when structured claims are empty', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(respond({ ...answer, claims: [] })));
    render(<App />);
    await submit();
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result.querySelector('.answer-text')?.textContent).toBe(answer.text);
    expect(within(result).queryByRole('region', { name: /Sources/ })).not.toBeInTheDocument();
  });
});
