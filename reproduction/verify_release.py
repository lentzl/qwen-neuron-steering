"""Verify file integrity and portable reading-copy links without network or model access."""
from pathlib import Path
import hashlib,json,re,zipfile
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
ROOT=Path(__file__).resolve().parents[1]
def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in {'.git','.venv','__pycache__','outputs','hf-cache'} for x in p.relative_to(ROOT).parts) and p.name not in {'.DS_Store','release-manifest.json'})
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:assert d['id'] not in self.ids;self.ids.add(d['id'])
        for k in ['href','src']:
            if k in d:self.links.append(d[k])
        assert tag not in ['script','iframe']
def main():
    expected=json.loads((ROOT/'release-manifest.json').read_text())['files']
    actual={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files()}
    assert actual==expected,'Release files differ from manifest'
    for p in files():
        assert not p.is_symlink()
        if p.suffix in {'.json','.py','.md','.html','.txt','.jsonl'}:
            s=p.read_text()
            assert not re.search(r'/(?:Users|home/ubuntu)/|file'+r'://',s),p.relative_to(ROOT)
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY)',s),p.relative_to(ROOT)
    for r in json.loads((ROOT/'provenance/exports.json').read_text())['files']:
        assert actual[r['path']]['sha256']==r['public_sha256'],r['path']
    parser=Links();parser.feed((ROOT/'index.html').read_text())
    for link in parser.links:
        if link.startswith('#'):assert link[1:] in parser.ids,link
        elif not urlparse(link).scheme:assert (ROOT/unquote(urlparse(link).path)).is_file(),link
    for target in re.findall(r'^\[[^\]]+\]: (.+)$',(ROOT/'ARTICLE.md').read_text(),re.M):
        if not urlparse(target).scheme:assert (ROOT/target).is_file(),target
    print(f'PASS: {len(actual)} files; export hashes; article links; no private absolute paths or credential patterns.')
if __name__=='__main__':main()
