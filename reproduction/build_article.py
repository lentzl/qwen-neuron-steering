"""Build a standalone HTML reading copy from ARTICLE.md; no model calls or remote assets."""
from pathlib import Path
import html, re
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
md=MarkdownIt('commonmark',{'html':False}).enable('table')
env={};tokens=md.parse((ROOT/'ARTICLE.md').read_text(),env);toc=[];slugs=set()
for i,t in enumerate(tokens):
    if t.type=='heading_open':
        label=tokens[i+1].content;slug=re.sub(r'[^a-z0-9]+','-',label.lower()).strip('-')
        assert slug not in slugs;slugs.add(slug);t.attrSet('id',slug)
        if t.tag=='h2':toc.append((slug,label))
for ref in env['references'].values():
    target=ref['href']
    if not target.startswith(('https://','http://')):assert (ROOT/target).is_file(),target
body=md.renderer.render(tokens,md.options,env).replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
css='''
:root{color-scheme:light;--ink:#192a30;--muted:#586c70;--accent:#146758;--line:#cedbd6;--paper:#fffef9;--wash:#edf4ef}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:2rem}
body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.7 Georgia,serif}
a{color:var(--accent);text-underline-offset:.2em}.topbar{padding:14px 5vw;border-bottom:1px solid var(--line);font:13px system-ui;display:flex;justify-content:space-between;gap:20px}
.layout{display:grid;grid-template-columns:220px minmax(0,790px);gap:58px;max-width:1235px;margin:auto;padding:48px 32px 100px}
aside{font:13px/1.5 system-ui;position:sticky;top:28px;align-self:start}aside li{margin:0 0 14px}aside a{text-decoration:none;color:var(--muted)}
main{min-width:0}main img{display:block;width:100%;height:auto;margin:28px 0 32px}
h1{font-size:clamp(38px,4.3vw,56px);line-height:1.08;letter-spacing:-.035em;font-weight:500;margin:0 0 24px}
h1+p{font:13px/1.6 system-ui;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:24px}
h2{font-size:29px;line-height:1.25;font-weight:500;margin:64px 0 22px;border-top:1px solid var(--line);padding-top:26px}h3{font-size:23px;font-weight:500}
p{margin:0 0 22px}blockquote{margin:28px 0;padding:6px 24px;border-left:3px solid var(--accent);background:var(--wash)}blockquote p{margin:10px 0}
.table-scroll{overflow-x:auto;margin:26px 0}table{width:100%;border-collapse:collapse;font:14px/1.5 system-ui}th,td{text-align:left;padding:13px 14px;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--wash)}
pre{overflow-x:auto;background:#f0f1ec;padding:16px}code{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}footer{border-top:1px solid var(--line);margin-top:50px;padding-top:20px;color:var(--muted);font:12px/1.6 system-ui}
:focus-visible{outline:3px solid #c98637;outline-offset:3px}
@media(max-width:850px){.layout{display:block;padding:30px 20px}aside{position:static;margin-bottom:36px}body{font-size:17px}}
@media print{aside,.topbar{display:none}.layout{display:block;padding:0}body{font-size:11pt}h2{break-after:avoid}img{max-height:80vh;object-fit:contain}}
'''
nav=''.join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s,t in toc)
page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Making Qwen’s “crimethink” public</title><style>{css}</style></head>
<body><div class="topbar"><span>An interpretability investigation</span><a href="README.md">Repository guide</a></div><div class="layout"><aside><nav aria-label="Contents"><ol>{nav}</ol></nav></aside><main>{body}<footer>Research direction: lentzl. Research, analysis and writing assistance: GPT-6 Astra through Codex. <a href="ACKNOWLEDGMENTS.md">Acknowledgments</a> · <a href="REPRODUCING.md">Reproduction</a> · <a href="LICENSE-ARTICLE">CC BY 4.0</a></footer></main></div></body></html>'''
(ROOT/'index.html').write_text(page)
print('Built index.html; all article evidence links resolve.')
