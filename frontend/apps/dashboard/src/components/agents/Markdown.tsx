import React from 'react';

/** Tiny markdown renderer for agent output: ### headings, bullet lists, **bold**, `code`, [T1]/[F3] citations. */
function inline(text: string, keyBase: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\[(?:[FTC]\d+)(?:\s*,\s*[FTC]\d+)*\])/g);
  return parts.filter(Boolean).map((p, i) => {
    const key = `${keyBase}-${i}`;
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={key}>{p.slice(2, -2)}</strong>;
    if (p.startsWith('`') && p.endsWith('`')) return <code key={key}>{p.slice(1, -1)}</code>;
    if (/^\[(?:[FTC]\d+)/.test(p)) return <span key={key} className="cite" title="Backed by a computed result">{p.slice(1, -1)}</span>;
    return <React.Fragment key={key}>{p}</React.Fragment>;
  });
}

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r/g, '').split('\n');
  const out: React.ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) {
      const items = list;
      out.push(<ul key={`ul-${out.length}`}>{items.map((l, i) => <li key={i}>{inline(l, `li-${out.length}-${i}`)}</li>)}</ul>);
      list = [];
    }
  };
  lines.forEach((raw, idx) => {
    const line = raw.trimEnd();
    const bullet = /^\s*(?:[-*]|\d+[.)])\s+(.*)$/.exec(line);
    if (bullet) {
      list.push(bullet[1]);
      return;
    }
    flush();
    if (!line.trim()) return;
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) out.push(<h4 key={idx}>{inline(h[2], `h-${idx}`)}</h4>);
    else out.push(<p key={idx}>{inline(line, `p-${idx}`)}</p>);
  });
  flush();
  return <div className="md">{out}</div>;
}
