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

  it('sends only the API fields and turns source references into local links', async () => {
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
    const link = within(result).getByRole('link', { name: '[aurora.pdf, PDF page 1]' });
    expect(link).toHaveAttribute('href', '#source-1');
    expect(document.querySelector('#source-1')).toHaveTextContent('aurora.pdf');
    expect(document.querySelector('#source-1')).toHaveTextContent('PDF page 1');
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
  ])('handles a non-JSON or malformed response without crashing', async (response) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    render(<App />);
    await submit();
    expect(await screen.findByRole('alert')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Ask GarageRAG' })).toBeEnabled();
  });

  it('keeps response markup inert and preserves multiline text and repeated citations', async () => {
    const text = '<script>alert(1)</script>\n[aurora.pdf, PDF page 1] "First line\nsecond line"\n\nAnother claim.\n[aurora.pdf, PDF page 1] "Another quote"';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(respond({ ...answer, text })));
    render(<App />);
    await submit();
    const result = await screen.findByRole('region', { name: 'Answer' });
    expect(result).toHaveTextContent('<script>alert(1)</script>');
    expect(result.querySelector('script')).toBeNull();
    expect(within(result).getAllByRole('link')).toHaveLength(2);
    expect(document.querySelectorAll('#source-1')).toHaveLength(1);
    expect(document.querySelector('#source-2')).toBeNull();
    expect(result.textContent).toContain('First line\nsecond line');
  });
});
