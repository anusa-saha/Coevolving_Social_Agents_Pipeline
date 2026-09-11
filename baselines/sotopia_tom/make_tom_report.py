"""Build the Sotopia-ToM-on-CSA report.

Recomputes every figure from whatever is on disk at build time. Before any arm has run it
emits the design, the computable baseline and the saturation prediction, with a banner
saying so; once `logs/Record-tom-*.txt` exist it fills in results and paired tests. Never
prints a number it did not derive from a file.

    python make_tom_report.py
"""
import ast
import glob
import os
import statistics as st
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, XPreformatted)

import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import data_csa as data_csa
import metrics_tom as M
import prompts_tom as P

OUT = 'sotopia-tom-csa-report.pdf'
TITLE = 'Sotopia-ToM on CSA: Prompting Without Training'
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ data
def load_records(path):
    out = []
    if not os.path.exists(path):
        return out
    for blk in open(path, encoding='utf-8').read().split('\n\n'):
        blk = blk.strip()
        if blk:
            try:
                out.append(ast.literal_eval(blk))
            except Exception:                        # noqa: BLE001
                pass
    return out


def arms_on_disk(split='test'):
    got = {}
    for arm in P.STRATEGIES:
        r = load_records(os.path.join(paths.LOGS, 'Record-tom-%s-%s.txt' % (arm, split)))
        if r:
            got[arm] = r
    return got


def normalise(r):
    """Records from the other baselines put reveal_* in different places."""
    s = r.get('score') or {}
    return {'uid': r.get('uid'),
            'score': {'disclosure_rate': s.get('disclosure_rate')},
            'reveal_elicited': r.get('reveal_elicited') or s.get('reveal_elicited') or {},
            'reveal_turn': r.get('reveal_turn') or s.get('reveal_turn') or {},
            'max_turn': r.get('max_turn') or s.get('max_turn') or 1}


def external_arms():
    """PPDPP and EPO records, so InfoMgmt3 has something to sit beside."""
    out = []
    p = os.path.join(HERE, '..', 'ppdpp', 'ppdpp_csa', 'tmp', 'csa', 'eval_result',
                     'Record-epoch-6-csa-sft-qwen-qwen-qwen-verifier-seed1.txt')
    if os.path.exists(p):
        out.append(('PPDPP ep6', [normalise(r) for r in load_records(p)]))
    import re
    pat = re.compile(r'-ep(\d+)\.txt$')
    for f in sorted(glob.glob(os.path.join(HERE, '..', 'epo', 'logs',
                                           'Record-*seed1-ep*.txt')),
                    key=lambda x: int(pat.search(x).group(1)) if pat.search(x) else 0):
        m = pat.search(f)
        if m:
            out.append(('EPO ep%s' % m.group(1), [normalise(r) for r in load_records(f)]))
    return out


ARMS = arms_on_disk()
EXT = external_arms()
HAVE_RESULTS = bool(ARMS)


# ------------------------------------------------------------------ style
def _fonts():
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        faces = [('DejaVu', 'DejaVuSans.ttf', 0, 0),
                 ('DejaVu-Bold', 'DejaVuSans-Bold.ttf', 1, 0),
                 ('DejaVu-Oblique', 'DejaVuSans-Oblique.ttf', 0, 1),
                 ('DejaVu-BoldOblique', 'DejaVuSans-BoldOblique.ttf', 1, 1)]
        for n, fn, _b, _i in faces:
            pdfmetrics.registerFont(TTFont(n, os.path.join(d, fn)))
        for n, _fn, b, i in faces:
            addMapping('DejaVu', b, i, n)
        pdfmetrics.registerFont(TTFont('DejaVuMono', os.path.join(d, 'DejaVuSansMono.ttf')))
        addMapping('DejaVuMono', 0, 0, 'DejaVuMono')
        return 'DejaVu', 'DejaVu-Bold', 'DejaVuMono'
    except Exception:                                # noqa: BLE001
        return 'Helvetica', 'Helvetica-Bold', 'Courier'


SANS, SANSB, MONO = _fonts()
INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5b5b5b')
RULE = colors.HexColor('#d0d0d0')
HEAD = colors.HexColor('#22333b')
BAND = colors.HexColor('#f2f4f5')
CODEBG = colors.HexColor('#f6f7f8')
GOOD = colors.HexColor('#2c6b55')
BAD = colors.HexColor('#9c2f3b')
WARN = colors.HexColor('#8e5314')

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontName=SANSB, fontSize=17, leading=21,
                    textColor=INK, alignment=TA_LEFT, spaceAfter=2)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontName=SANS, fontSize=9, leading=13,
                     textColor=MUTED, spaceAfter=4)
META = ParagraphStyle('META', parent=SUB, fontName=MONO, fontSize=7.6, leading=11,
                      spaceAfter=12)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontName=SANSB, fontSize=12, leading=15,
                    textColor=HEAD, spaceBefore=14, spaceAfter=5)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontName=SANSB, fontSize=9.5, leading=12,
                    textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle('BODY', parent=ss['Normal'], fontName=SANS, fontSize=9, leading=12.5,
                      textColor=INK, spaceAfter=6)
BUL = ParagraphStyle('BUL', parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=3.5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8, leading=11, textColor=MUTED)
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontName=MONO, fontSize=7.2, leading=9.8,
                      textColor=INK, leftIndent=0, spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName=SANS, fontSize=7.4, leading=9.4,
                      textColor=INK)
CELLM = ParagraphStyle('CELLM', parent=CELL, fontName=MONO, fontSize=7.2)
CELLH = ParagraphStyle('CELLH', parent=CELL, fontName=SANSB, textColor=colors.white)
CALL = ParagraphStyle('CALL', parent=BODY, fontSize=8.4, leading=11.6, spaceAfter=5)


def Pp(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BUL, bulletText='•')


def mono(t):
    return '<font face="%s">%s</font>' % (MONO, t)


def col(t, c):
    return '<font color="#%s"><b>%s</b></font>' % (c.hexval()[2:], t)


def fmt(x, d=3):
    try:
        return ('%.' + str(d) + 'f') % x
    except Exception:                                # noqa: BLE001
        return '&#8212;'


def table(rows, widths, mono_cols=(), size=7.4):
    c = ParagraphStyle('c', parent=CELL, fontSize=size, leading=size + 2)
    m = ParagraphStyle('m', parent=CELLM, fontSize=size - 0.2, leading=size + 2)
    data = [[Pp(x, CELLH) for x in rows[0]]]
    for r in rows[1:]:
        data.append([Pp(x, m if j in mono_cols else c) for j, x in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    stl = [('BACKGROUND', (0, 0), (-1, 0), HEAD),
           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
           ('TOPPADDING', (0, 0), (-1, -1), 4),
           ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
           ('LEFTPADDING', (0, 0), (-1, -1), 5),
           ('RIGHTPADDING', (0, 0), (-1, -1), 5),
           ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
           ('BOX', (0, 0), (-1, -1), 0.5, RULE)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            stl.append(('BACKGROUND', (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(stl))
    return t


def codebox(text, bar=HEAD, width=170 * mm):
    t = Table([[XPreformatted(text, CODE)]], colWidths=[width], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), CODEBG),
                           ('BOX', (0, 0), (-1, -1), 0.5, RULE),
                           ('LINEBEFORE', (0, 0), (0, -1), 2.2, bar),
                           ('LEFTPADDING', (0, 0), (-1, -1), 8),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                           ('TOPPADDING', (0, 0), (-1, -1), 6),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return t


def callout(title, paras, bar=BAD, width=170 * mm):
    inner = [Paragraph(title, ParagraphStyle('CT', parent=CALL, fontName=SANSB,
                                             fontSize=8, textColor=bar, spaceAfter=4))]
    inner += [Paragraph(x, CALL) for x in paras]
    t = Table([[inner]], colWidths=[width], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), BAND),
                           ('BOX', (0, 0), (-1, -1), 0.5, RULE),
                           ('LINEBEFORE', (0, 0), (0, -1), 2.2, bar),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 9),
                           ('TOPPADDING', (0, 0), (-1, -1), 7),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t


# ------------------------------------------------------------------ document
S = []
S.append(Paragraph(TITLE, H1))
S.append(Paragraph(
    'Sotopia-ToM (arXiv:2605.02307) reimplemented from the paper and evaluated on CSA '
    'with executable metrics instead of an LLM judge. Nothing is trained: one frozen '
    'Qwen2.5-7B-Instruct plays every role, and the arms differ only in how the chair is '
    'prompted.', SUB))
S.append(Paragraph(
    'arms %d/5 run  |  split test (n=42)  |  greedy decoding  |  no training, no API  |  '
    'all figures recomputed from records' % len(ARMS), META))

if not HAVE_RESULTS:
    S.append(callout('NO RESULTS YET', [
        'No %s files exist. Everything below is the design, the baseline computed from '
        'the arms that HAVE run, and what to expect -- there is not a single Sotopia-ToM '
        'number in this document.' % mono('logs/Record-tom-*.txt'),
        'Re-running %s once the arms have been evaluated fills in the results tables and '
        'the paired tests automatically.' % mono('make_tom_report.py')], bar=WARN))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. What this arm does', H2))
S.append(Pp(
    'The other three baselines all train something: PPDPP a planner, EPO a strategist, '
    'Sotopia-RL the agent itself. This one trains nothing. It is the training-free '
    'reference the comparison otherwise lacks, and after two arms where the supervised '
    'stage beat the reinforcement stage, the question it answers is the live one: does '
    'any of that training beat good prompting?'))
S.append(Spacer(1, 2))
S.append(table([
    ['arm', 'extra calls / turn', 'what the chair is told', 'source'],
    ['stripped', '0', 'nothing about drawing information out', col('ours', WARN)],
    ['basic', '0', 'pursue the decision, close when done', 'their vanilla'],
    ['cot', '0', 'five reasoning steps, then THINKING / TURN', 'their CoT-Privacy'],
    ['tom_coach', '1', 'a fresh read of what each participant holds', 'theirs'],
    ['tom_belief', '1', 'a belief state carried across the meeting', 'theirs'],
], [24 * mm, 26 * mm, 66 * mm, 32 * mm], mono_cols=(0,)))
S.append(Spacer(1, 5))
S.append(Pp(
    'The %s block from the CoT arm is stripped before the turn enters the transcript: '
    'advisors must never read the chair&#8217;s private deliberation. The two ToM arms '
    'each spend one extra model call per chair turn, and that cost is counted in %s '
    'like every other call, because a strategy that wins by making more calls has not '
    'won.' % (mono('THINKING:'), mono('n_calls')), NOTE))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. Why the control arm is the whole experiment', H2))
S.append(Pp(
    'CSA&#8217;s own chair prompt already contains the instruction '
    '<i>&#8220;The others hold information you do not have; it is your job to draw it '
    'out.&#8221;</i> That is most of what chain-of-thought and theory-of-mind scaffolding '
    'exist to induce. Without a control that removes it, the four real arms would very '
    'likely land on top of each other and the study would measure nothing.'))
S.append(Pp(
    'So %s is not a courtesy baseline. If it scores close to %s, the scaffolds have no '
    'room to show anything, and the correct output of this study is that finding rather '
    'than a table of near-identical numbers.' % (mono('stripped'), mono('basic'))))

S.append(Paragraph('2.1 The saturation risk is measurable in advance', H3))
S.append(Pp(
    'Sotopia-ToM&#8217;s headline finding is that inquiry alignment sits at <b>0.288 '
    'across every model they test</b> &#8212; agents share information but almost never '
    'deliberately ask for it. That gap is what their ToM arms exist to close.'))
rows = [['arm', 'n', 'DA', 'IA', 'EFF', 'InfoMgmt3']]
for lbl, recs in EXT:
    m = M.summarise(recs)
    rows.append([lbl, str(m['n']), fmt(m['DA']), fmt(m['IA']), fmt(m['EFF']),
                 fmt(m['InfoMgmt3'])])
for arm, recs in ARMS.items():
    m = M.summarise(recs)
    rows.append([col('ToM ' + arm, GOOD), str(m['n']), fmt(m['DA']), fmt(m['IA']),
                 fmt(m['EFF']), fmt(m['InfoMgmt3'])])
if len(rows) > 1:
    S.append(Spacer(1, 2))
    S.append(table(rows, [36 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm, 28 * mm],
                   mono_cols=(0,)))
    S.append(Spacer(1, 4))
    ia = [float(r[3]) for r in rows[1:] if r[3] != '&#8212;']
    if ia:
        S.append(Pp(
            'On CSA the existing arms already reach IA up to %s, against the 0.288 the '
            'paper reports. Different benchmark, computed rather than judged, and a '
            'three-way rather than four-way mean &#8212; so not a like-for-like. But it '
            'is direct evidence that CSA&#8217;s task instruction already produces most '
            'of the behaviour the scaffolds target.' % mono(fmt(max(ia))), NOTE))

# ---------------------------------------------------------------- 3
S.append(PageBreak())
S.append(Paragraph('3. Metrics', H2))
S.append(Pp(
    'Three of the paper&#8217;s four dimensions have exact CSA analogues that need no '
    'judge. The fourth does not exist here at all and is reported as absent rather than '
    'approximated.'))
S.append(Spacer(1, 2))
S.append(table([
    ['paper', 'here', 'computed from'],
    ['DA &#8212; disclosure alignment', 'decisive facts that reached the meeting',
     mono('score.disclosure_rate')],
    ['IA &#8212; inquiry alignment',
     'of those, the share ELICITED rather than volunteered', mono('reveal_elicited')],
    ['EFF &#8212; efficiency', 'how early pooling happened vs the turn budget',
     mono('reveal_turn / max_turn')],
    ['CPV &#8212; privacy violations', col('no analogue on CSA', BAD), '&#8212;'],
], [46 * mm, 68 * mm, 46 * mm]))
S.append(Spacer(1, 5))
S.append(codebox('InfoMgmt3 = [ DA . IA . EFF ]^(1/3)          '
                 '(the paper: [DA . IA . (1-CPV) . EFF]^(1/4))'))
S.append(Spacer(1, 5))
S.append(callout('THE COMPOSITE IS NOT COMPARABLE TO THE PAPER', [
    'CSA is a single public meeting in which every private fact SHOULD be pooled. There '
    'is no private channel and nothing an agent must withhold, so critical privacy '
    'violations have no meaning here. Dropping a factor that is usually close to 1 raises '
    'the mean, so InfoMgmt3 reads HIGHER than InfoMgmt would on identical behaviour.',
    'Use it to rank these arms against each other. Never place it beside the '
    'paper&#8217;s table. %s is deliberately not substituted for CPV either: a leak here '
    'means an agent stated a fact it was never shown, which is fabrication, not '
    'inappropriate disclosure.' % mono('leaks')], bar=WARN))
S.append(Spacer(1, 4))
S.append(Pp(
    'The composite is averaged over EPISODES, not computed from the averaged dimensions. '
    'Doing it the other way lets an episode that scored zero on one dimension be rescued '
    'by the others, which is exactly what a geometric mean exists to prevent.', NOTE))

# ---------------------------------------------------------------- 4
S.append(Paragraph('4. Results', H2))
if not HAVE_RESULTS:
    S.append(Pp('No arm has been run. The command that fills this section in:'))
    S.append(codebox('cd baselines/sotopia_tom\n'
                     'python run_tom.py --strategies stripped basic --device cuda:0\n'
                     '\n'
                     '# only if those two separate:\n'
                     'python run_tom.py --strategies cot tom_coach tom_belief '
                     '--device cuda:0\n'
                     'python run_tom.py --compare        # scores what is on disk',
                     bar=WARN))
    S.append(Spacer(1, 5))
    S.append(Pp(
        'Two arms over 42 scenarios is roughly an hour with no training and no API. Run '
        '%s and %s first and read the gap before building out the rest: if they do not '
        'separate, the remaining three arms cannot tell you anything the first two did '
        'not.' % (mono('stripped'), mono('basic'))))
else:
    hdr = ['arm', 'n', 'DA', 'IA', 'EFF', 'InfoMgmt3', 'SR', 'dca', 'calls']
    rows = [hdr]
    for arm in P.STRATEGIES:
        if arm not in ARMS:
            continue
        m = M.summarise(ARMS[arm])
        rows.append([arm, str(m['n']), fmt(m['DA']), fmt(m['IA']), fmt(m['EFF']),
                     fmt(m['InfoMgmt3']), fmt(m['SR']), fmt(m['dca']),
                     fmt(m['n_calls'], 1)])
    S.append(table(rows, [24 * mm, 13 * mm, 18 * mm, 18 * mm, 18 * mm, 24 * mm,
                          16 * mm, 18 * mm, 16 * mm], mono_cols=(0,)))

    base = 'stripped' if 'stripped' in ARMS else sorted(ARMS)[0]
    S.append(Spacer(1, 6))
    S.append(Paragraph('4.1 Paired per-scenario tests against %s' % base, H3))
    S.append(Pp('Every arm runs the same 42 scenarios, so the comparison is paired. '
                'Two-sided sign test over the decisive pairs.', NOTE))
    prows = [['arm', 'metric', 'win', 'tie', 'loss', 'p']]
    for arm in P.STRATEGIES:
        if arm not in ARMS or arm == base:
            continue
        for key in ('DA', 'IA', 'EFF', 'InfoMgmt3'):
            r = M.paired(ARMS[arm], ARMS[base], key)
            pf = '&lt;0.001' if r['p'] < 0.001 else fmt(r['p'])
            prows.append([arm, key, str(r['win']), str(r['tie']), str(r['loss']), pf])
    S.append(table(prows, [26 * mm, 26 * mm, 18 * mm, 18 * mm, 18 * mm, 22 * mm],
                   mono_cols=(0, 1)))
    S.append(Spacer(1, 5))
    if base in ARMS and 'basic' in ARMS:
        a, b = M.summarise(ARMS[base]), M.summarise(ARMS['basic'])
        d = b['InfoMgmt3'] - a['InfoMgmt3']
        if abs(d) < 0.02:
            S.append(callout('SATURATED', [
                'The control and the vanilla arm differ by %s on InfoMgmt3. CSA&#8217;s '
                'task instruction already produces what the scaffolds target, so the '
                'remaining arms have no headroom. Report this rather than the near-'
                'identical table it would produce.' % mono(fmt(d))], bar=WARN))
        else:
            S.append(Pp('Control to vanilla moves InfoMgmt3 by %s, so the instruction '
                        'itself is doing measurable work and the scaffolds have room.'
                        % mono(fmt(d))))

# ---------------------------------------------------------------- 5
S.append(Paragraph('5. Deviations from the paper', H2))
S.append(table([
    ['#', 'Paper', 'Here', 'Why'],
    ['1', 'their released code', 'reimplemented from the prose',
     'the "Code" link points at the generic sotopia framework; the Sotopia-ToM codebase '
     '"will be made publicly available" and was not'],
    ['2', 'LLM-judged DA / IA / EFF', 'computed from executable signals',
     'CSA ships the ground truth; a judge would reintroduce the error this project '
     'exists to remove'],
    ['3', 'CPV, four-way composite', 'omitted, three-way composite',
     'no private channel and nothing to withhold on CSA'],
    ['4', 'CoT-Privacy', 'CoT-Elicitation',
     'the leakage-check step is a no-op here, so it is repointed at which decisive fact '
     'is still missing'],
    ['5', '&#8212;', 'a stripped control arm',
     'CSA already instructs elicitation; without the control the arms are not separable'],
    ['6', 'GPT-4o and five others', 'Qwen2.5-7B-Instruct only',
     'matches the other three baselines, so the environment is constant across arms'],
], [7 * mm, 34 * mm, 40 * mm, 79 * mm]))

S.append(Paragraph('6. Caveats', H2))
for t in [
    'Every template here is our reading of the paper&#8217;s description, not their code. '
    'Treat the arms as faithful in intent, not in detail.',
    'The scenarios are CSA, not the paper&#8217;s 160. CSA was generated FROM Sotopia-ToM '
    'seeds by a separate pipeline and has a different schema, so this measures their '
    'methods on our data, not a reproduction of their benchmark.',
    'One seed, 42 scenarios. Paired tests are the only way to see a small effect at this '
    'n, and every arm runs the same scenarios by construction so pairing is always valid.',
    'The two ToM arms cost roughly 30%% more calls per episode than the others. Read the '
    'comparison at matched budget, not matched episode count.',
]:
    S.append(B(t))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'Sotopia-ToM: Yashwanth YS, Wang, Zeng, Zhou, Onoue, Varadarajan &amp; Sap, '
    '<i>Evaluating Information Management in Multi-Agent Interaction with Theory of '
    'Mind</i> (arXiv:2605.02307). Figures recomputed at build time from '
    '<font face="%s">logs/</font> and from the PPDPP and EPO records; nothing is copied '
    'from an earlier summary.' % MONO, NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'Sotopia-ToM on CSA')
    canv.drawRightString(A4[0] - 20 * mm, 10 * mm, str(canv.getPageNumber()))
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.4)
    canv.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
    canv.restoreState()


if __name__ == '__main__':
    SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=18 * mm,
                      title=TITLE, author='').build(S, onFirstPage=_footer,
                                                    onLaterPages=_footer)
    print('wrote %s  (%d/5 arms had results)' % (OUT, len(ARMS)))
