import { Fragment } from 'react';

export function AnswerText({ text }: { text: string }) {
  // /ask embeds verified references in plain text; it does not expose PDF URLs.
  // Add only local anchors. Keep the complete answer and quotes as inert text.
  const references = [...text.matchAll(/^\[(.+), PDF page (\d+)\](?= ")/gm)];
  const sources = [...new Map(references.map((match) => [match[0], {
    reference: match[0], document: match[1], page: match[2],
  }])).values()];
  let position = 0;
  return (
    <>
      <div className="answer-text">
        {references.map((match) => {
          const preceding = text.slice(position, match.index);
          position = match.index + match[0].length;
          const number = sources.findIndex((source) => source.reference === match[0]) + 1;
          return <Fragment key={match.index}>{preceding}<a href={`#source-${number}`}>{match[0]}</a></Fragment>;
        })}
        {text.slice(position)}
      </div>
      {sources.length ? (
        <section className="sources" aria-labelledby="sources-heading">
          <h3 id="sources-heading">Sources <span>{sources.length}</span></h3>
          <ol>
            {sources.map((source, index) => (
              <li id={`source-${index + 1}`} key={source.reference} tabIndex={-1}>
                <span className="source-number">{String(index + 1).padStart(2, '0')}</span>
                <div><p>{source.document}</p><span>PDF page {source.page}</span></div>
              </li>
            ))}
          </ol>
          <p className="source-note">Page references from the answer. Open your own PDF to read the full passage.</p>
        </section>
      ) : null}
    </>
  );
}
