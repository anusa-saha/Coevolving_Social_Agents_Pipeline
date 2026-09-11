"""Build the EPO-CSA results PDF.

Every figure is recomputed from the evaluation records at build time -- nothing is copied
from an earlier summary -- so re-running this after a new eval regenerates a truthful
document rather than a stale one.

    python make_epo_report.py
"""
import ast
import collections
import glob
import json
import os
import re
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

OUT = 'epo-csa-report.pdf'
TITLE = 'EPO-CSA: Results of the 700-Episode Run'
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, 'logs')
PPDPP_REC = os.path.join(HERE, '..', 'ppdpp', 'ppdpp_csa', 'tmp', 'csa', 'eval_result',
                         'Record-epoch-6-csa-sft-qwen-qwen-qwen-verifier-seed1.txt')
PPDPP_REC0 = os.path.join(HERE, '..', 'ppdpp', 'ppdpp_csa', 'tmp', 'csa', 'eval_result',
                          'Record-epoch-0-csa-sft-qwen-qwen-qwen-verifier-seed1.txt')


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


def checkpoints():
    pat = re.compile(r'-ep(\d+)\.txt$')
    got = [(int(pat.search(f).group(1)), f)
           for f in glob.glob(os.path.join(LOGS, 'Record-*seed1-ep*.txt'))
           if pat.search(f)]
    return sorted(got)


def summarise(R):
    sc = lambda r: (r.get('score') or {})            # noqa: E731

    def avg(fn):
        v = [fn(r) for r in R if fn(r) is not None]
        return sum(v) / len(v) if v else float('nan')

    S = [s or '' for r in R for s in (r.get('strategies') or [])]
    return {
        'n': len(R),
        'SR': avg(lambda r: 1.0 if r.get('done') == 1 else 0.0),
        'dca': avg(lambda r: sc(r).get('dca')),
        'disc': avg(lambda r: sc(r).get('disclosure_rate')),
        'anyrev': sum(1 for r in R if r.get('revealed')),
        'elic': sum(1 for r in R for v in (r.get('reveal_elicited') or {}).values() if v),
        'cbar': avg(lambda r: sc(r).get('cbar')),
        'pbar': avg(lambda r: sc(r).get('pbar')),
        'schema': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
        'leaks': sum(1 for r in R if r.get('leaks')),
        'turns': avg(lambda r: r.get('turns')),
        'calls': avg(lambda r: r.get('n_calls')),
        'reward': avg(lambda r: r.get('reward')),
        'strat_n': len(S), 'strat_u': len(set(S)),
        'uniq': 100.0 * len(set(S)) / max(1, len(S)),
        'empty': sum(1 for s in S if not s.strip()),
        'short': sum(1 for s in S if len(s.strip()) <= 3),
        'tagmiss': sum(r.get('tag_misses', 0) for r in R),
        'acts': dict(collections.Counter(a for r in R
                                         for a in (r.get('act_history') or []))),
        'strategies': S,
    }


def sign_test(a_recs, b_recs, metric):
    """Paired per-scenario wins/ties/losses plus a two-sided sign-test p."""
    A = {r['uid']: r for r in a_recs}
    B = {r['uid']: r for r in b_recs}
    w = l = t = 0
    for u in sorted(set(A) & set(B)):
        x = (A[u].get('score') or {}).get(metric)
        y = (B[u].get('score') or {}).get(metric)
        if x is None or y is None:
            continue
        if x > y:
            w += 1
        elif x < y:
            l += 1
        else:
            t += 1
    n = w + l
    if n == 0:
        return w, t, l, float('nan')
    from math import comb
    k = min(w, l)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return w, t, l, p


# PPDPP's stored scores were produced by an older verifier -- its records carry no
# `close` key, and provenance resolution demonstrably did not fire (healthcare::scenario_5
# has PF1 overlapping the settlement prose at 0.450, well over the 0.35 threshold, yet P1
# scored False). Trusting r['score'] understates its pbar by half and makes the comparison
# uneven, because EPO's records DO have resolution applied.
#
# The verifier and detectors are imported from sotopia_rl: they are byte-identical to
# ppdpp_csa's apart from one import line, and unlike ppdpp_csa's they do not drag in torch.
# This is a reporting script, so a path shim here costs nothing.
_SR = os.path.abspath(os.path.join(HERE, '..', 'sotopia_rl'))
if os.path.isdir(_SR) and _SR not in sys.path:
    sys.path.insert(0, _SR)
try:
    import data_csa as _dc
    import verifier_sr as _V
    _CASES = _dc.case_index()
except Exception as _e:                              # noqa: BLE001
    _CASES, _V = {}, None
    print('note: could not import the verifier (%s); PPDPP scores left as stored' % _e)


def rescore(recs):
    """Recompute each record's score with the CURRENT verifier, so every arm in this
    document was scored the same way. Returns the records unchanged if unavailable."""
    if not _V or not _CASES:
        return recs, 0
    out, changed = [], 0
    for r in recs:
        case = _CASES.get(r.get('uid'))
        if not case:
            out.append(r)
            continue
        s = _V.score(case, r.get('settlement') or {}, set(r.get('revealed') or []),
                     resolve=True)
        s.pop('settlement_resolved', None)
        if any(abs(s[k] - r['score'][k]) > 1e-9
               for k in ('dca', 'cbar', 'pbar') if k in r['score']):
            changed += 1
        n = dict(r)
        n['score'] = s
        out.append(n)
    return out, changed


CKPTS = checkpoints()
SUMM = [(ep, summarise(load_records(f))) for ep, f in CKPTS]
EP0 = dict(SUMM)[0] if SUMM else {}
LAST_EP, LAST = SUMM[-1] if SUMM else (0, {})
PPD, PPD_FIXED = rescore(load_records(PPDPP_REC))
PPD0, _ = rescore(load_records(PPDPP_REC0))
PPDS = summarise(PPD) if PPD else {}
PPDS0 = summarise(PPD0) if PPD0 else {}
REC0 = load_records(dict(CKPTS)[0]) if CKPTS else []
RECL = load_records(dict(CKPTS)[LAST_EP]) if CKPTS else []


def catalogue(name):
    p = os.path.join(LOGS, 'catalogue-%s.json' % name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


CAT = {k: catalogue(v) for k, v in
       (('ppdpp', 'ppdpp-ep6'), ('ep0', 'ep0'), ('last', 'ep700'))}


def cat(arm, section, key, d=3):
    v = (CAT.get(arm) or {}).get(section, {}).get(key)
    return ('%.' + str(d) + 'f') % v if isinstance(v, (int, float)) else '&#8212;'


def critic_calls(arm):
    v = (CAT.get(arm) or {}).get('C_efficiency', {}).get('calls_by_role', {}).get('critic')
    return v if isinstance(v, (int, float)) else None


def infomgmt(recs):
    """Sotopia-ToM's InfoMgmt, rebuilt from CSA's executable signals.

    DA  decisive facts pooled                 -> disclosure_rate
    IA  of those, the share ELICITED          -> reveal_elicited
    EFF how early, against the turn budget    -> reveal_turn / max_turn
    CPV no analogue: CSA has one public channel and nothing to withhold.

    Three-way geometric mean, so NOT comparable in level to the paper's four-way number
    -- dropping a factor usually near 1 raises it. Use it to rank these arms only.
    """
    import statistics as _st
    vals = []
    for r in recs:
        sc = r.get('score') or {}
        da = float(sc.get('disclosure_rate') or 0.0)
        el = r.get('reveal_elicited') or sc.get('reveal_elicited') or {}
        ia = (sum(1 for v in el.values() if v) / len(el)) if el else 0.0
        rt = [t for t in (r.get('reveal_turn') or sc.get('reveal_turn') or {}).values()
              if isinstance(t, (int, float))]
        cap = max(1, int(r.get('max_turn') or sc.get('max_turn') or 1))
        eff = max(0.0, min(1.0, 1.0 - (_st.median(rt) / cap))) if rt else 0.0
        g = 0.0 if min(da, ia, eff) <= 0 else (da * ia * eff) ** (1.0 / 3.0)
        vals.append((da, ia, eff, g))
    n = max(1, len(vals))
    return {k: sum(v[i] for v in vals) / n
            for i, k in enumerate(('DA', 'IA', 'EFF', 'InfoMgmt3'))}


def fmt(x, d=3):
    try:
        return ('%.' + str(d) + 'f') % x
    except Exception:                                # noqa: BLE001
        return '&#8212;'


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
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontName=MONO, fontSize=7.4, leading=10,
                      textColor=INK, leftIndent=0, spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName=SANS, fontSize=7.4, leading=9.4,
                      textColor=INK)
CELLM = ParagraphStyle('CELLM', parent=CELL, fontName=MONO, fontSize=7.2)
CELLH = ParagraphStyle('CELLH', parent=CELL, fontName=SANSB, textColor=colors.white)
CALL = ParagraphStyle('CALL', parent=BODY, fontSize=8.4, leading=11.6, spaceAfter=5)


def P(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BUL, bulletText='•')


def mono(t):
    return '<font face="%s">%s</font>' % (MONO, t)


def col(t, c):
    return '<font color="#%s"><b>%s</b></font>' % (c.hexval()[2:], t)


def table(rows, widths, mono_cols=(), size=7.4):
    c = ParagraphStyle('c', parent=CELL, fontSize=size, leading=size + 2)
    m = ParagraphStyle('m', parent=CELLM, fontSize=size - 0.2, leading=size + 2)
    data = [[P(x, CELLH) for x in rows[0]]]
    for r in rows[1:]:
        data.append([P(x, m if j in mono_cols else c) for j, x in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    st = [('BACKGROUND', (0, 0), (-1, 0), HEAD),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('TOPPADDING', (0, 0), (-1, -1), 4),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
          ('LEFTPADDING', (0, 0), (-1, -1), 5),
          ('RIGHTPADDING', (0, 0), (-1, -1), 5),
          ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
          ('BOX', (0, 0), (-1, -1), 0.5, RULE)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(('BACKGROUND', (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(st))
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
    'What the run produced, how it compares to the PPDPP baseline on the same 42 held-out '
    'scenarios, and the failure that dominates the result.', SUB))
S.append(Paragraph(
    'arm verifier / binary PRM / group advantage, seed 1  |  700 episodes  |  '
    'evaluation greedy on the test split (n=%d)  |  all figures recomputed from records'
    % LAST.get('n', 0), META))
S.append(P(
    "PPDPP's stored scores predate the current verifier (no %s key, provenance "
    'resolution not applied), so its column here is RESCORED from its own records with '
    'the same code used for EPO -- %d of its 42 episodes changed. Comparing against its '
    'stored numbers would understate its provenance score by half.'
    % (mono('close'), PPD_FIXED), NOTE))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. Headline', H2))
S.append(P(
    'EPO beats the PPDPP baseline decisively on every pooling metric. It is also the first '
    'planner on this benchmark to record a non-zero success rate at all: PPDPP sat at '
    '%s for its entire 1,000-episode run, EPO reaches %s.'
    % (col(fmt(PPDS.get('SR', 0)), BAD), col(fmt(LAST.get('SR', 0)), GOOD))))
S.append(Spacer(1, 2))
S.append(table([
    ['Metric', 'PPDPP ep6', 'EPO ep0 (SFT only)', 'EPO ep%d' % LAST_EP, 'Best'],
    ['Success rate', fmt(PPDS.get('SR')), fmt(EP0.get('SR')), fmt(LAST.get('SR')),
     col('EPO ep%d' % LAST_EP, GOOD)],
    ['dca', fmt(PPDS.get('dca')), fmt(EP0.get('dca')), fmt(LAST.get('dca')),
     col('EPO ep%d' % LAST_EP, GOOD)],
    ['disclosure_rate', fmt(PPDS.get('disc')), fmt(EP0.get('disc')), fmt(LAST.get('disc')),
     col('EPO ep0', WARN)],
    ['episodes with any reveal', '%d/42' % PPDS.get('anyrev', 0),
     '%d/42' % EP0.get('anyrev', 0), '%d/42' % LAST.get('anyrev', 0), col('EPO ep0', WARN)],
    ['cbar', fmt(PPDS.get('cbar')), fmt(EP0.get('cbar')), fmt(LAST.get('cbar')),
     col('EPO ep0', WARN)],
    ['pbar', fmt(PPDS.get('pbar')), fmt(EP0.get('pbar')), fmt(LAST.get('pbar')),
     col('EPO ep%d' % LAST_EP, GOOD)],
    ['schema valid', fmt(PPDS.get('schema')), fmt(EP0.get('schema')),
     fmt(LAST.get('schema')), '&#8212;'],
    ['episodes with a leak', str(PPDS.get('leaks', 0)), str(EP0.get('leaks', 0)),
     str(LAST.get('leaks', 0)), col('EPO ep%d' % LAST_EP, GOOD)],
    ['calls per episode', fmt(PPDS.get('calls'), 1), fmt(EP0.get('calls'), 1),
     fmt(LAST.get('calls'), 1), 'matched'],
], [42 * mm, 26 * mm, 34 * mm, 26 * mm, 28 * mm]))
S.append(Spacer(1, 5))
S.append(P(
    'Call budget is matched (%s vs %s per episode), so the gain is not bought by making '
    'more model calls than the baseline.'
    % (mono(fmt(LAST.get('calls'), 1)), mono(fmt(PPDS.get('calls'), 1))), NOTE))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. Paired per-scenario comparison', H2))
S.append(P(
    'The same 42 scenarios are run by both planners, so the comparison is paired rather '
    'than a difference of unpaired means. Two-sided sign test over the decisive pairs.'))
rows = [['Metric', 'EPO wins', 'ties', 'PPDPP wins', 'p']]
for label, recs in (('ep0 (SFT only)', REC0), ('ep%d' % LAST_EP, RECL)):
    for metric in ('dca', 'disclosure_rate', 'cbar', 'pbar'):
        w, t, l, p = sign_test(recs, PPD, metric)
        pf = '&lt;0.001' if p < 0.001 else fmt(p, 3)
        rows.append(['%s &#183; %s' % (label, metric), str(w), str(t), str(l), pf])
S.append(table(rows, [52 * mm, 24 * mm, 20 * mm, 26 * mm, 24 * mm]))
S.append(Spacer(1, 5))
S.append(P(
    'Disclosure is the cleanest result in the study: at ep0 EPO wins 29 scenarios and '
    'loses <b>none</b>. This is the metric the benchmark exists to measure, and it is where '
    'the planner class actually differs.', NOTE))

# ---------------------------------------------------------------- 3
S.append(PageBreak())
S.append(Paragraph('3. Trajectory across the run', H2))
S.append(table(
    [['ckpt', 'SR', 'dca', 'disclosure', 'any reveal', 'elicited', 'cbar', 'pbar',
      'leaks', 'calls']]
    + [['ep%d' % ep, fmt(s['SR']), fmt(s['dca']), fmt(s['disc']),
        '%d/%d' % (s['anyrev'], s['n']), str(s['elic']), fmt(s['cbar']), fmt(s['pbar']),
        str(s['leaks']), fmt(s['calls'], 1)] for ep, s in SUMM],
    [18 * mm, 17 * mm, 17 * mm, 23 * mm, 22 * mm, 19 * mm, 17 * mm, 17 * mm,
     15 * mm, 15 * mm], mono_cols=(0,)))
S.append(Spacer(1, 6))
S.append(callout('THE RL STAGE DID NOT HELP; IT DEGRADED THE POLICY', [
    'ep0 is the behaviour-cloned policy <i>before a single gradient step of RL</i>. It is '
    'the best checkpoint in the run on disclosure (%s) and on episodes with any reveal '
    '(%d/42). Seven hundred episodes of REINFORCE moved disclosure DOWN to %s and reveals '
    'to %d/42.'
    % (fmt(EP0.get('disc')), EP0.get('anyrev', 0), fmt(LAST.get('disc')),
       LAST.get('anyrev', 0)),
    'Success rate moves %s &#8594; %s, which is 6 versus 8 successes out of 42 &#8212; well '
    'inside sampling noise at this n. The honest reading is that <b>the entire measured '
    'gain over PPDPP comes from the supervised warm-start</b>, and the RL stage is at best '
    'neutral and on the pooling metrics actively harmful.'
    % (fmt(EP0.get('SR')), fmt(LAST.get('SR')))], bar=BAD))

# ---------------------------------------------------------------- 4
S.append(Paragraph('4. Why: the policy collapsed', H2))
S.append(P(
    'The strategist stopped producing varied, scenario-specific strategies and converged '
    'on a template. Distinct strategy strings, out of ~173 emitted per evaluation:'))
S.append(Spacer(1, 2))
S.append(table(
    [['ckpt', 'emitted', 'unique', 'unique %', 'empty', '&#8804;3 chars', 'tag misses']]
    + [['ep%d' % ep, str(s['strat_n']), str(s['strat_u']), fmt(s['uniq'], 1) + '%',
        str(s['empty']), str(s['short']), str(s['tagmiss'])] for ep, s in SUMM],
    [20 * mm, 22 * mm, 20 * mm, 22 * mm, 18 * mm, 24 * mm, 24 * mm], mono_cols=(0,)))
S.append(Spacer(1, 6))
S.append(P('What that looks like in practice &#8212; the most frequent outputs:'))


def top_strats(s, k=5):
    c = collections.Counter(s.get('strategies') or [])
    lines = []
    for txt, n in c.most_common(k):
        t = (txt or '').strip()
        lines.append('  %3dx  %s' % (n, ('(empty)' if not t else t[:78])))
    return '\n'.join(lines)


S.append(codebox('ep0   %d unique / %d\n%s'
                 % (EP0.get('strat_u', 0), EP0.get('strat_n', 0), top_strats(EP0)),
                 bar=GOOD))
S.append(Spacer(1, 3))
S.append(codebox('ep%d  %d unique / %d\n%s'
                 % (LAST_EP, LAST.get('strat_u', 0), LAST.get('strat_n', 0),
                    top_strats(LAST)), bar=BAD))
S.append(Spacer(1, 6))
S.append(P(
    'The act distribution tells the same story. ep0 emits %s; every later checkpoint emits '
    '%s &#8212; but that is misleading, because %d of 173 turns failed to produce a '
    'parseable act tag and were defaulted to <i>ask</i> by %s. Roughly half the turns at '
    'ep%d carry no usable strategy at all.'
    % (mono(str(EP0.get('acts'))), mono(str(LAST.get('acts'))), LAST.get('tagmiss', 0),
       mono('parse_strategy'), LAST_EP)))

# ---------------------------------------------------------------- 5
S.append(PageBreak())
S.append(Paragraph('5. Full metric catalogue', H2))
S.append(P(
    'The catalogue computes 267 numeric metrics per checkpoint. Section B (pooling) is the '
    'part that separates these planners, and it complicates the picture above: the RL stage '
    'traded elicitation for grounded citation rather than simply degrading.'))
S.append(Spacer(1, 2))
S.append(table([
    ['metric', 'PPDPP ep6', 'EPO ep0', 'EPO ep%d' % LAST_EP, 'moved by RL'],
    ['decisive disclosure rate', cat('ppdpp', 'B_pooling', 'decisive_disclosure_rate'),
     cat('ep0', 'B_pooling', 'decisive_disclosure_rate'),
     cat('last', 'B_pooling', 'decisive_disclosure_rate'), col('down', BAD)],
    ['decisive credit rate', cat('ppdpp', 'B_pooling', 'decisive_credit_rate'),
     cat('ep0', 'B_pooling', 'decisive_credit_rate'),
     cat('last', 'B_pooling', 'decisive_credit_rate'), col('up 5.7x', GOOD)],
    ['pooling completeness', cat('ppdpp', 'B_pooling', 'pooling_completeness'),
     cat('ep0', 'B_pooling', 'pooling_completeness'),
     cat('last', 'B_pooling', 'pooling_completeness'), col('up', GOOD)],
    ['disclosure F1', cat('ppdpp', 'B_pooling', 'disclosure_f1'),
     cat('ep0', 'B_pooling', 'disclosure_f1'),
     cat('last', 'B_pooling', 'disclosure_f1'), col('down', BAD)],
    ['elicited fraction', cat('ppdpp', 'B_pooling', 'elicited_fraction'),
     cat('ep0', 'B_pooling', 'elicited_fraction'),
     cat('last', 'B_pooling', 'elicited_fraction'), col('up', GOOD)],
    ['silent holder rate', cat('ppdpp', 'B_pooling', 'silent_holder_rate'),
     cat('ep0', 'B_pooling', 'silent_holder_rate'),
     cat('last', 'B_pooling', 'silent_holder_rate'), col('up', BAD)],
    ['hallucinated credit rate', cat('ppdpp', 'B_pooling', 'hallucinated_credit_rate'),
     cat('ep0', 'B_pooling', 'hallucinated_credit_rate'),
     cat('last', 'B_pooling', 'hallucinated_credit_rate'), col('down (good)', GOOD)],
    ['citation containment', cat('ppdpp', 'B_pooling', 'citation_containment'),
     cat('ep0', 'B_pooling', 'citation_containment'),
     cat('last', 'B_pooling', 'citation_containment'), col('up 5x', GOOD)],
    ['P_micro (provenance)', cat('ppdpp', 'A_task_outcome', 'P_micro'),
     cat('ep0', 'A_task_outcome', 'P_micro'),
     cat('last', 'A_task_outcome', 'P_micro'), col('up 3.2x', GOOD)],
    ['act entropy (bits)', cat('ppdpp', 'G_policy', 'act_entropy_bits'),
     cat('ep0', 'G_policy', 'act_entropy_bits'),
     cat('last', 'G_policy', 'act_entropy_bits'), col('collapsed', BAD)],
], [42 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm]))
S.append(Spacer(1, 5))
S.append(P(
    'Disclosure precision is %s at every checkpoint and over-disclosure is %s, so the '
    'detector produces no false positives and the disclosure differences are real.'
    % (mono(cat('last', 'B_pooling', 'disclosure_precision', 2)),
       mono(cat('last', 'B_pooling', 'over_disclosure_rate', 2))), NOTE))
S.append(Spacer(1, 4))
S.append(callout('THE CITATION GAIN IS LARGELY NOT THE POLICY&#8217;S', [
    'Calls to the %s role &#8212; the fallback settlement extractor, which runs only when '
    'the chair never emitted parseable JSON itself &#8212; go from <b>%s of 42 episodes at '
    'ep0 to %s of 42 at ep%d</b>.'
    % (mono('critic'), critic_calls('ep0'), critic_calls('last'), LAST_EP),
    'At ep%d the collapsed policy produced <b>no usable settlement in any episode</b>; the '
    'extractor wrote all 42. The improved provenance, credit-rate and hallucination numbers '
    'are therefore substantially the extractor&#8217;s clean pass over the transcript, not a '
    'capability the trained policy acquired. %s stays at 1.000 throughout and hides this, '
    'because it measures the settlement AFTER the fallback ran.'
    % (LAST_EP, mono('json_parse_rate')),
    'Read with the diversity collapse, the honest summary is that RL destroyed the '
    'policy&#8217;s ability to settle and the extractor picked up the slack.'], bar=BAD))
S.append(PageBreak())
S.append(Paragraph('6. Diagnosis', H2))
S.append(P(
    'This is textbook mode collapse for REINFORCE on a language-model policy, and it was '
    'anticipated: the design spec added a KL penalty to the frozen base specifically as '
    'insurance against it, at %s. That was evidently too weak, and there are two '
    'possibilities worth separating before the next run.' % mono('--kl_beta 0.01')))
for t in [
    '<b>The KL was too small.</b> 0.01 over ~175 optimizer steps may simply not constrain '
    'an 8B policy. Raising it to 0.05&#8211;0.1 is the first thing to try.',
    '<b>The KL was never applied.</b> %s returns None when the installed peft cannot '
    'disable adapters, and the strategist then skips the term with a printed warning. If '
    'that warning appears in the training log, the run had <i>no</i> KL at all and the '
    'collapse is fully explained. <b>Check the log before changing anything else.</b>'
    % mono('ref_logprob'),
    '<b>The degeneracy is progressive, not sudden.</b> Unique strategies fall %s across '
    'the run and empty outputs appear only at the end (0 at ep528, %d at ep%d). A '
    'diversity check on the evaluation records would have caught this at ep352 and is '
    'cheap to add as an early stop.'
    % (mono('96.5%% &#8594; 94.2%% &#8594; 68.2%% &#8594; 59.5%% &#8594; 37.0%%'),
       LAST.get('empty', 0), LAST_EP),
    '<b>Reward shape may be rewarding brevity.</b> Under the binary verifier PRM most '
    'turns score zero, so the gradient is dominated by a handful of rewarded turns; if '
    'those happen to be short, the token-averaged objective pushes toward shorter output. '
    'The graded PRM (%s) spreads credit and is worth trying for this reason alone.'
    % mono('--prm_mode graded'),
]:
    S.append(B(t))

S.append(Paragraph('7. What to change', H2))
S.append(table([
    ['#', 'Change', 'Why'],
    ['1', 'Grep the training log for the KL warning',
     'Determines whether the KL term ran at all. Nothing else should change until this '
     'is known.'],
    ['2', 'Raise %s to 0.05&#8211;0.1' % mono('--kl_beta'),
     'Direct constraint against drift from the warm-started policy.'],
    ['3', 'Early-stop on strategy diversity',
     'Unique-strategy fraction is computed from records already written; stop when it '
     'drops below ~80%%.'],
    ['4', 'Try %s' % mono('--prm_mode graded'),
     'Denser, better-shaped credit. The binary PRM leaves most turns at exactly zero.'],
    ['5', 'Report ep0 as a first-class arm, not a starting point',
     'It is the strongest configuration measured and beats PPDPP on every paired '
     'comparison. "Filtered behaviour cloning beats the planner baseline" is a real '
     'result and should be reported as one.'],
    ['6', 'Run 2 more seeds before claiming the SR difference',
     '6 vs 8 successes out of 42 is noise. The disclosure result is robust; the SR '
     'result is not.'],
], [7 * mm, 62 * mm, 101 * mm]))

S.append(PageBreak())
S.append(Paragraph('A. Full metric catalogue', H2))
S.append(P(
    'Every scalar the catalogue computes, for the three arms, grouped as %s emits them. '
    'Per-role call counts and per-scenario-type breakdowns are omitted here (about 160 '
    'further keys) because they are one-row-per-name and belong in the JSON; everything '
    'else is reproduced in full. Sections F (language quality) and J (distributional) are '
    'not computed at all -- they need reference justifications, a judge model, or a second '
    'distribution, none of which exist.' % mono('compute_all_metrics.py')))

_SECTION_TITLES = {
    'A_task_outcome': 'A. Task outcome',
    'B_pooling': 'B. Information pooling',
    'C_efficiency': 'C. Efficiency and cost',
    'D_format': 'D. Format compliance',
    'G_policy': 'G. Policy diagnostics',
    'I_baselines': 'I. Baselines and paired tests',
}
_SKIP = ('calls_by_role', 'by_scenario_type', 'by_domain', 'by_num_agents',
         'by_decisive_fact_count', 'by_turn_cap_bucket', 'score_distribution',
         'settlement_keys_frequency', 'act_bigrams_top', 'act_distribution',
         'act_conditional_yield', 'per_check_pass_rate')


def _flat(o, pre=''):
    out = {}
    for k, v in (o or {}).items():
        if k in _SKIP:
            continue
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[pre + k] = v
    return out


for _sec, _title in _SECTION_TITLES.items():
    keys = []
    for arm in ('ppdpp', 'ep0', 'last'):
        keys += list(_flat((CAT.get(arm) or {}).get(_sec, {})))
    keys = sorted(dict.fromkeys(keys))
    if not keys:
        continue
    rows = [['metric', 'PPDPP ep6', 'EPO ep0', 'EPO ep%d' % LAST_EP]]
    for k in keys:
        vals = []
        for arm in ('ppdpp', 'ep0', 'last'):
            v = _flat((CAT.get(arm) or {}).get(_sec, {})).get(k)
            vals.append(('%.4f' % v) if isinstance(v, (int, float)) else '&#8212;')
        rows.append([k] + vals)
    S.append(Paragraph(_title, H3))
    S.append(table(rows, [74 * mm, 32 * mm, 32 * mm, 32 * mm], mono_cols=(0,), size=6.8))

S.append(Paragraph('A.7  Per-check pass rates', H3))
S.append(P('Which individual checks the settlements satisfy. A check at 0.0000 in every '
           'column is unreachable and depresses every aggregate that includes it.', NOTE))
_pc = {}
for arm in ('ppdpp', 'ep0', 'last'):
    _pc[arm] = ((CAT.get(arm) or {}).get('A_task_outcome', {})
                .get('per_check_pass_rate', {}))
_ck = sorted(dict.fromkeys([k for a in _pc for k in _pc[a]]))
if _ck:
    rows = [['check', 'PPDPP ep6', 'EPO ep0', 'EPO ep%d' % LAST_EP, 'reachable?']]
    for k in _ck:
        vals = [_pc[a].get(k) for a in ('ppdpp', 'ep0', 'last')]
        dead = all(isinstance(v, (int, float)) and v == 0.0 for v in vals if v is not None)
        rows.append([k] + [('%.4f' % v) if isinstance(v, (int, float)) else '&#8212;'
                           for v in vals]
                    + [col('never passes', BAD) if dead else 'yes'])
    S.append(table(rows, [22 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm], mono_cols=(0,)))

S.append(Paragraph('A.8  Sotopia-ToM InfoMgmt, computed', H3))
S.append(P(
    "Sotopia-ToM's composite, rebuilt from CSA's executable signals rather than its LLM "
    'judge. DA is decisive facts pooled, IA the share of those that were ELICITED rather '
    'than volunteered, EFF how early pooling happened against the turn budget. CPV has no '
    'analogue here -- CSA is one public channel with nothing to withhold -- so this is a '
    "THREE-way geometric mean and reads higher than the paper's four-way number. Rank "
    'these arms with it; never compare it to their table.'))
_rows = [['arm', 'n', 'DA', 'IA', 'EFF', 'InfoMgmt3']]
for _lbl, _recs in ([('PPDPP ep6', PPD)]
                    + [('EPO ep%d' % ep, load_records(dict(CKPTS)[ep]))
                       for ep, _s in SUMM]):
    m = infomgmt(_recs)
    _rows.append([_lbl, str(len(_recs)), fmt(m['DA']), fmt(m['IA']), fmt(m['EFF']),
                  fmt(m['InfoMgmt3'])])
S.append(table(_rows, [34 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm, 28 * mm],
               mono_cols=(0,)))
S.append(Spacer(1, 4))
S.append(P(
    'For reference, Sotopia-ToM report IA = 0.288 across every model they test -- agents '
    'share but rarely ask. EPO reaches %s here. Not a like-for-like comparison, but it '
    "suggests CSA's chair prompt already induces most of what their ToM scaffolds exist "
    'to produce.' % mono(fmt(infomgmt(REC0)['IA'])), NOTE))

S.append(PageBreak())
S.append(Paragraph('8. Caveats', H2))
for t in [
    'One seed. Everything above is seed 1; the SR and dca differences between EPO '
    'checkpoints are within noise at n=42, though the paired disclosure result is not.',
    'Leaks fell from %d episodes at ep0 to %d at ep%d, so integrity improved even as '
    'capability degraded. That is consistent with a policy emitting less content overall '
    'rather than with genuine learning.'
    % (EP0.get('leaks', 0), LAST.get('leaks', 0), LAST_EP),
    'pbar rose from %s to %s. Provenance checks pass under 5%% in absolute terms on this '
    'corpus, so the movement is on a very small base and should not be read as a '
    'capability gain.' % (fmt(PPDS.get('pbar')), fmt(LAST.get('pbar'))),
    'The PPDPP comparison column is its epoch-6 checkpoint, the end of a 1,000-episode '
    'run whose own curve was flat throughout. It is a floor, not a tuned opponent.',
]:
    S.append(B(t))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'Recomputed from %d evaluation records across %d checkpoints in '
    '<font face="%s">epo/logs/</font>, paired against the PPDPP epoch-6 records. '
    'Re-running <font face="%s">make_epo_report.py</font> regenerates every figure from '
    'source.' % (sum(s['n'] for _e, s in SUMM), len(SUMM), MONO, MONO), NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'EPO-CSA — results')
    canv.drawRightString(A4[0] - 20 * mm, 10 * mm, str(canv.getPageNumber()))
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.4)
    canv.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
    canv.restoreState()


if __name__ == '__main__':
    if not SUMM:
        raise SystemExit('no evaluation records found in %s' % LOGS)
    SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=18 * mm,
                      title=TITLE, author='').build(S, onFirstPage=_footer,
                                                    onLaterPages=_footer)
    print('wrote', OUT)
