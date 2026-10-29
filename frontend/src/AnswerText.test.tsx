import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AnswerText } from './AnswerText';
import type { Citation } from './api';

const claim = 'Чтобы открыть капот, потяните рычаг под приборной панелью.';
const citation: Citation = {
  source_id: 'S1', document: 'bmw_e39_2002_manual.pdf', pdf_page: 137,
  pdf_sha256: 'a'.repeat(64), quote: 'Pull the lever under the instrument panel.',
};

describe('Answer presentation', () => {
  it('shows the claim once with one compact link per PDF page and collapsed evidence', () => {
    const open = vi.fn();
    render(<AnswerText text={`${claim}\n[${citation.document}, PDF page 137] "${citation.quote}"`}
      claims={[{ text: claim, citations: [citation, { ...citation, quote: 'Pull the release handle.' }] }]}
      onOpenPdf={open} />);

    expect(screen.getAllByText(claim)).toHaveLength(1);
    const link = screen.getByRole('button', { name: `Open ${citation.document} at PDF page 137` });
    expect(link).toHaveTextContent('PDF · 137');
    expect(link).not.toHaveTextContent(citation.document);
    expect(screen.getByText(citation.document)).not.toBeVisible();
    expect(screen.getByText(citation.quote)).not.toBeVisible();
    fireEvent.click(link);
    expect(open).toHaveBeenCalledWith(citation);

    fireEvent.click(screen.getByText('Sources', { exact: false, selector: 'summary' }));
    expect(screen.getByText(citation.quote)).toBeVisible();
    expect(screen.getByText('Pull the release handle.')).toBeVisible();
    expect(screen.getAllByText(citation.document)).toHaveLength(1);
    expect(screen.getAllByText(claim)).toHaveLength(1);
  });

  it('shares evidence across claims without merging different documents or revisions', () => {
    render(<AnswerText text="Legacy text" claims={[
      { text: claim, citations: [citation, { ...citation, document: 'other.pdf' }] },
      { text: 'Second claim.', citations: [citation, { ...citation, pdf_sha256: 'b'.repeat(64) }] },
    ]} onOpenPdf={vi.fn()} />);

    const paragraphs = document.querySelectorAll('.answer-claim');
    expect(paragraphs).toHaveLength(2);
    expect(within(paragraphs[0] as HTMLElement).getAllByRole('button')).toHaveLength(2);
    expect(within(paragraphs[1] as HTMLElement).getAllByRole('button')).toHaveLength(2);
    expect(screen.getByText('Sources', { exact: false, selector: 'summary' })).toHaveTextContent('3');
    fireEvent.click(screen.getByText('Sources', { exact: false, selector: 'summary' }));
    expect(screen.getAllByText(citation.quote)).toHaveLength(3);
    expect(screen.getByText('other.pdf')).toBeVisible();
    expect(screen.queryByText('Legacy text')).not.toBeInTheDocument();
  });

  it('keeps plain text intact and inert when structured claims are unavailable', () => {
    const text = '<script>alert(1)</script>\n[manual.pdf, PDF page 7]';
    render(<AnswerText text={text} claims={[]} onOpenPdf={vi.fn()} />);
    expect(document.querySelector('.answer-text')?.textContent).toBe(text);
    expect(document.querySelector('script')).toBeNull();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
