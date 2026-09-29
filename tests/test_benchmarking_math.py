"""GitHub math delimiters must shield TeX from Markdown without changing it."""
import hashlib
import json
from pathlib import Path
import re

REPORT = Path(__file__).resolve().parents[1] / 'docs/benchmarking.md'
DISPLAY = re.compile(r'(?ms)^\$\$\n(.*?)^\$\$$')
PROTECTED_INLINE = re.compile(r'\$`([^`\n]+)`\$')
FENCED_DISPLAY = re.compile(r'(?ms)^```math\n(.*?)^```$')


def unprotect(text):
    text = PROTECTED_INLINE.sub(lambda m: '$' + m[1] + '$', text)
    return FENCED_DISPLAY.sub(lambda m: '$$\n' + m[1] + '$$', text)


def test_display_equations_are_separate_markdown_blocks():
    text = REPORT.read_text()
    lines = text.splitlines()
    assert '$$' not in text, 'Dollar-delimited displays let Markdown consume TeX escapes'
    assert r'\operatorname' not in text, 'GitHub rejects this macro; use the upright Alg label'
    assert r'\tag' not in text, 'Use explicit visible labels, not optional MathML tag support'
    for number in (1, 2):
        assert any(f'\\qquad ({number})' in body for body in FENCED_DISPLAY.findall(text))
    for i, line in enumerate(lines):
        if line == '```math':
            assert i and not lines[i - 1].strip(), f'No blank line before equation on line {i + 1}'
        elif line == '```':
            assert i + 1 < len(lines) and not lines[i + 1].strip(), f'No blank line after equation on line {i + 1}'
    assert len(FENCED_DISPLAY.findall(text)) == 18
    assert not re.search(r'(?m)(?<!\\)\\$', text), 'GitHub doubles a trailing TeX spacing backslash'


def test_inline_math_is_protected_from_markdown_emphasis_and_escapes():
    prose = FENCED_DISPLAY.sub('', REPORT.read_text())
    assert len(PROTECTED_INLINE.findall(prose)) == 126
    assert '$' not in PROTECTED_INLINE.sub('', prose)


def test_tex_bodies_and_tags_unchanged_by_presentation_fix():
    # These two spellings denote the same upright algorithm label.
    text = unprotect(REPORT.read_text()).replace(r'\mathrm{Alg}', r'\operatorname{Alg}')
    for number in (1, 2):
        text = text.replace(f'\\qquad ({number})', f'\\tag{{{number}}}')
    display = DISPLAY.findall(text)
    inline = re.findall(r'\$([^\$\n]+)\$', DISPLAY.sub('', text))
    bodies = {kind: [' '.join(s.split()) for s in values]
              for kind, values in [('display', display), ('inline', inline)]}
    digest = hashlib.sha256(json.dumps(bodies,
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    # Frozen from the published report before changing only its delimiters.
    assert digest == 'eb9ec44e0863c513e8a6a37a120552e44f8f054cf4a3d9b40590aa5eeec18378'
    assert r'\tag{1}' in text and r'\tag{2}' in text
    for name in ('mathcal-e-definition', 'fourth-order-expansion-definition'):
        assert f'<a id="{name}"></a>' in text and f'](#{name})' in text


def test_all_table_cells_times_and_evidence_links_are_unchanged():
    lines = [line for line in unprotect(REPORT.read_text()).splitlines() if line.startswith('|')]
    assert len(lines) == 54
    assert hashlib.sha256('\n'.join(lines).encode()).hexdigest() == '4e9736fad4c31d75f751791dc90421a2f9edbd76bd7189d43f2dcc275dd1b7fc'
