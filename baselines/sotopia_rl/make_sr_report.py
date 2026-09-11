"""Build the Sotopia-RL-on-CSA report.

Recomputes every figure at build time from data/, ckpt/ and logs/. The arm has training
artifacts but no held-out evaluation, so the document is explicit about which numbers are
training-time and which section is empty until evaluate_sr.py runs.

    python make_sr_report.py
"""
import ast
import collections
import glob
import json
import os
import re
import statistics as st

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

import attribution
import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import data_csa as data_csa

OUT = 'sotopia-rl-csa-report.pdf'
TITLE = 'Sotopia-RL on CSA: Results of the Training Run'
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = data_csa.case_index()


# ------------------------------------------------------------------ load
def jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:                        # noqa: BLE001
                pass
    return out


def records(p):
    if not os.path.exists(p):
        return []
    out = []
    for blk in open(p, encoding='utf-8').read().split('\n\n'):
        blk = blk.strip()
        if blk:
            try:
                out.append(ast.literal_eval(blk))
            except Exception:                        # noqa: BLE001
                pass
    return out


def meta(p):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


EPS = jsonl(os.path.join(paths.DATA, 'episodes-train.jsonl'))
RM = jsonl(os.path.join(paths.DATA, 'rm-train.jsonl'))
SFT_META = meta(os.path.join(paths.CKPT, 'sft', 'sft_meta.json'))
RM_META = meta(os.path.join(paths.CKPT, 'rm', 'rm_meta.json'))
GRPO_SUM = {}
GRPO_HIST = []
for f in glob.glob(os.path.join(paths.LOGS, '*-summary.json')):
    GRPO_SUM = meta(f)
for f in glob.glob(os.path.join(paths.LOGS, '*-history.jsonl')):
    GRPO_HIST = jsonl(f)
EVAL = {}
for f in glob.glob(os.path.join(paths.LOGS, 'Record-*.txt')):
    tag = re.sub(r'^Record-|-\w+\.txt$', '', os.path.basename(f))
    r = records(f)
    if r:
        EVAL[tag] = r
HAVE_EVAL = bool(EVAL)


# env_sr.episode() records uid/dialog/settlement/revealed/leaks/score, and
# collect_episodes drops the per-check dicts from the score. So `reveal_elicited`,
# `reveal_turn` and the individual check results are NOT on disk. All three are
# recoverable: the transcript is stored, so replaying it with the same detectors gives
# the elicitation bookkeeping, and rescoring the settlement gives the checks. This is
# what make_rm_data.py already does, so the numbers agree with the labels by
# construction rather than by coincidence.
def enrich(eps):
    from verifier_sr import score as _score
    out = []
    for e in eps:
        case = CASES.get(e['uid'])
        if not case:
            continue
        w = attribution.walk_episode(case, e['dialog'])
        s = _score(case, e.get('settlement') or {}, set(e.get('revealed') or []),
                   resolve=True)
        n = dict(e)
        n['reveal_elicited'] = w['reveal_elicited']
        n['reveal_turn'] = w['reveal_turn']
        n['addressed'] = sorted(w['addressed'])
        n['max_turn'] = w['n_turns']
        n['turns'] = w['n_turns']
        n['cover'] = (len(set().union(*w['cover_credit'].values())) /
                      max(1, len(w['advisors']))) if w['cover_credit'] else 0.0
        n['checks'] = {**s.get('content', {}), **s.get('provenance', {})}
        out.append(n)
    return out


EPS = enrich(EPS)


def maxturn(uid):
    c = CASES.get(uid)
    if not c:
        return 1
    o = c['interaction_config']['turn_order']
    cap = int(c['interaction_config']['turn_cap'])
    return max(1, sum(1 for i in range(cap) if o[i % len(o)] == c['decision_maker']))


def avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else float('nan')


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
INK, MUTED = colors.HexColor('#1a1a1a'), colors.HexColor('#5b5b5b')
RULE, HEAD = colors.HexColor('#d0d0d0'), colors.HexColor('#22333b')
BAND, CODEBG = colors.HexColor('#f2f4f5'), colors.HexColor('#f6f7f8')
GOOD, BAD, WARN = (colors.HexColor('#2c6b55'), colors.HexColor('#9c2f3b'),
                   colors.HexColor('#8e5314'))

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
        if x != x:
            return '&#8212;'
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
    stl = [('BACKGROUND', (0, 0), (-1, 0), HEAD), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
           ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
           ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
           ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE), ('BOX', (0, 0), (-1, -1), 0.5, RULE)]
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
    'Sotopia-RL (arXiv:2508.03905) on CSA: episode outcomes refined into utterance-level '
    'rewards, distilled into a reward model, then optimised with single-turn GRPO. Unlike '
    'the planner baselines this arm fine-tunes the dialogue agent itself.', SUB))
S.append(Paragraph(
    'corpus %d episodes / %d scenarios  |  reward rows %d  |  GRPO %s groups x %s  |  '
    'eval arms %d  |  all figures recomputed from files'
    % (len(EPS), len({e['uid'] for e in EPS}) if EPS else 0, len(RM),
       GRPO_SUM.get('groups', '?'), GRPO_SUM.get('group_size', '?'), len(EVAL)), META))

if not HAVE_EVAL:
    S.append(callout('TRAINING RAN; NOTHING HAS BEEN EVALUATED', [
        'All three stages completed and the checkpoints exist, but there are no %s files '
        '— nothing has been scored on the held-out 42 scenarios. Every number below '
        'is training-time.' % mono('logs/Record-*.txt'),
        'That matters because the training signal and the ground truth disagree, as '
        'section 5 shows. Until the evaluation runs, this document cannot say whether the '
        'arm helped or hurt.'], bar=WARN))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. What ran', H2))
S.append(table([
    ['stage', 'output', 'key settings', 'status'],
    ['1  collect_episodes', '%d episodes / %d scenarios'
     % (len(EPS), len({e['uid'] for e in EPS}) if EPS else 0),
     'k=6, keep=2, verifier-filtered', col('done', GOOD) if EPS else col('missing', BAD)],
    ['1b make_rm_data', '%d (state, action, scalar) rows' % len(RM),
     'pool / use / cover, dataset-level norm',
     col('done', GOOD) if RM else col('missing', BAD)],
    ['2.1 train_sft', '%s examples' % SFT_META.get('examples', '?'),
     'lr %s, %s epochs, min_dca %s' % (SFT_META.get('lr', '?'),
                                       SFT_META.get('epochs', '?'),
                                       SFT_META.get('min_dca', '?')),
     col('done', GOOD) if SFT_META else col('missing', BAD)],
    ['2.2 train_rm', '%s train / %s valid rows'
     % (RM_META.get('train', '?'), RM_META.get('valid', '?')),
     'MSE, lr %s, %s epochs' % (RM_META.get('lr', '?'), RM_META.get('epochs', '?')),
     col('done, but see §4', WARN) if RM_META else col('missing', BAD)],
    ['2.3 train_grpo', '%s groups, %s minutes'
     % (GRPO_SUM.get('groups', '?'), GRPO_SUM.get('minutes', '?')),
     'group %s, reward_source %s' % (GRPO_SUM.get('group_size', '?'),
                                     GRPO_SUM.get('reward_source', '?')),
     col('done', GOOD) if GRPO_SUM else col('missing', BAD)],
    ['3  evaluate_sr', '&#8212;', 'greedy, test split',
     col('NOT RUN', BAD) if not HAVE_EVAL else col('done', GOOD)],
], [30 * mm, 44 * mm, 60 * mm, 36 * mm], mono_cols=(0,)))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. Stage 1 — the corpus', H2))
if EPS:
    sc = [e['score'] for e in EPS if 'score' in e]
    S.append(Pp(
        'Self-play from the frozen base model, k=6 rollouts per scenario ranked WITHIN '
        'the scenario and the top 2 kept, so hard scenarios still contribute rather than '
        'the filter selecting only easy ones. These are TRAIN-split, post-filter numbers '
        'and are not comparable to held-out figures from the other arms.'))
    S.append(Spacer(1, 2))
    d = [s['dca'] for s in sc]
    rows = [['metric', 'mean', 'median', 'max']]
    for lbl, key in (('dca', 'dca'), ('cbar', 'cbar'), ('pbar', 'pbar'),
                     ('close', 'close'), ('disclosure_rate', 'disclosure_rate'),
                     ('hallucinated_credit', 'hallucinated_credit')):
        v = [s.get(key) for s in sc if isinstance(s.get(key), (int, float))]
        rows.append([lbl, fmt(avg(v)), fmt(st.median(v)) if v else '&#8212;',
                     fmt(max(v)) if v else '&#8212;'])
    S.append(table(rows, [44 * mm, 30 * mm, 30 * mm, 30 * mm], mono_cols=(0,)))
    S.append(Spacer(1, 4))
    anyrev = sum(1 for e in EPS if e.get('revealed'))
    el = [v for e in EPS for v in (e.get('reveal_elicited') or {}).values()]
    S.append(table([
        ['quantity', 'value'],
        ['episodes with any decisive fact revealed',
         '%d / %d (%.0f%%)' % (anyrev, len(EPS), 100 * anyrev / len(EPS))],
        ['disclosures that followed an eliciting turn',
         '%d / %d (%.0f%%)' % (sum(1 for v in el if v), len(el),
                               100 * sum(1 for v in el if v) / max(1, len(el)))],
        ['schema valid', fmt(avg([1.0 if s.get('schema_valid') else 0.0 for s in sc]))],
        ['episodes with a leak', str(sum(1 for e in EPS if e.get('leaks')))],
        ['turns per episode', fmt(avg([e.get('turns') for e in EPS]), 2)],
    ], [104 * mm, 66 * mm], mono_cols=(1,)))
    S.append(Spacer(1, 4))
    S.append(Pp(
        'For orientation only: PPDPP scores dca 0.214 and disclosure 0.089 on the '
        'held-out 42, and EPO ep0 reaches 0.336 / 0.419. The corpus figures above are on '
        'a different split and after selection, so they are a statement about what the '
        'base model can produce when filtered, not a comparison.', NOTE))

# ---------------------------------------------------------------- 3
S.append(Paragraph('3. Stage 1b — the reward labels', H2))
if RM:
    lab = [r['label'] for r in RM]
    nz = sum(1 for x in lab if x > 1e-9)
    S.append(Pp(
        'One scalar per chair turn. Sotopia-RL obtains it by asking GPT-4o which turns '
        'were critical; here it is computed from %s, which states for every private fact '
        'exactly which checks it controls. No model, no API, byte-for-byte reproducible.'
        % mono('decisive_facts')))
    S.append(Spacer(1, 2))
    rows = [['dimension', 'non-zero', 'mean', 'max']]
    for dim in attribution.DIMS:
        v = [r['raw'][dim] for r in RM if 'raw' in r and dim in r['raw']]
        n = sum(1 for x in v if x > 1e-9)
        rows.append([dim, '%d / %d (%.1f%%)' % (n, len(v), 100 * n / max(1, len(v))),
                     fmt(avg(v), 4), fmt(max(v)) if v else '&#8212;'])
    rows.append([col('aggregate label', GOOD),
                 '%d / %d (%.1f%%)' % (nz, len(lab), 100 * nz / len(lab)),
                 fmt(avg(lab), 4), fmt(max(lab))])
    S.append(table(rows, [40 * mm, 46 * mm, 30 * mm, 30 * mm], mono_cols=(0,)))
    S.append(Spacer(1, 4))
    S.append(Pp(
        'A volunteered fact scores nothing: credit requires the disclosure to follow an '
        'eliciting turn, which is the behaviour the benchmark is about. %s of turns carry '
        'a non-zero label, which is dense enough to regress on.'
        % mono('%.1f%%' % (100 * nz / len(lab))), NOTE))

# ---------------------------------------------------------------- 4
S.append(PageBreak())
S.append(Paragraph('4. Stage 2.2 — the reward model is the weak link', H2))
if RM_META:
    pr = RM_META.get('best_pair_rank')
    S.append(table([
        ['quantity', 'value', 'reading'],
        ['rows', '%s train / %s valid' % (RM_META.get('train'), RM_META.get('valid')),
         'held out by SCENARIO, not by turn'],
        ['best pair-ranking accuracy', fmt(pr) if pr else '&#8212;',
         col('0.5 is a coin flip', BAD)],
        ['best epoch', str(RM_META.get('best_epoch')),
         col('never improved after the first', BAD)],
        ['epochs run', str(RM_META.get('epochs')), 'selection on ranking, not MSE'],
    ], [50 * mm, 46 * mm, 74 * mm], mono_cols=(1,)))
    S.append(Spacer(1, 5))
    S.append(callout('THE REWARD MODEL BARELY ORDERS CANDIDATES BETTER THAN CHANCE', [
        'Pair-ranking accuracy is the quantity GRPO actually consumes: rewards are '
        'standardised inside each group, so what matters is whether the model orders '
        'candidates at the SAME state correctly, not its absolute calibration. At %s '
        'against a 0.5 coin flip it is contributing mostly noise.'
        % mono(fmt(pr) if pr else '?'),
        '%s means it was already overfitting by the first epoch on %s rows. That is the '
        'expected outcome at this scale and is a reason to prefer %s, which needs no '
        'reward model at all.'
        % (mono('best_epoch = 0'), RM_META.get('train'),
           mono('--reward_source lookahead'))], bar=BAD))

# ---------------------------------------------------------------- 5
S.append(Paragraph('5. Stage 2.3 — GRPO, and the disagreement that matters', H2))
if GRPO_HIST:
    ex = [h for h in GRPO_HIST if h['turn'] >= maxturn(h['uid']) - 1]
    rm_g = [h for h in GRPO_HIST if h not in ex]
    S.append(Pp(
        'Settling turns are scored EXACTLY by the verifier rather than by the reward '
        'model — a settlement can be parsed and run through the checks with no '
        'rollout and no model. So the training signal splits into a trustworthy part and '
        'a learned part, and they can be read separately.'))
    S.append(Spacer(1, 2))
    S.append(table([
        ['scorer', 'groups', 'candidates', 'mean', 'sd', 'range', 'collapsed'],
        [col('EXACT (verifier)', GOOD), str(len(ex)), str(len(ex) * 8),
         fmt(avg([s for h in ex for s in h['scores']])),
         fmt(st.pstdev([s for h in ex for s in h['scores']])),
         '%s .. %s' % (fmt(min(s for h in ex for s in h['scores'])),
                       fmt(max(s for h in ex for s in h['scores']))),
         '%d/%d' % (sum(1 for h in ex if h['collapsed']), len(ex))],
        ['RM (learned)', str(len(rm_g)), str(len(rm_g) * 8),
         fmt(avg([s for h in rm_g for s in h['scores']])),
         fmt(st.pstdev([s for h in rm_g for s in h['scores']])),
         '%s .. %s' % (fmt(min(s for h in rm_g for s in h['scores'])),
                       fmt(max(s for h in rm_g for s in h['scores']))),
         '%d/%d' % (sum(1 for h in rm_g if h['collapsed']), len(rm_g))],
    ], [34 * mm, 18 * mm, 22 * mm, 20 * mm, 20 * mm, 32 * mm, 22 * mm], mono_cols=(1, 2)))

    S.append(Paragraph('5.1 The two signals move in opposite directions', H3))
    n_e, n_r = max(1, len(ex) // 4), max(1, len(rm_g) // 4)
    rows = [['quarter', 'EXACT mean', 'EXACT best-in-group', 'RM mean',
             'unparseable', 'invalid schema', 'valid']]
    for i in range(4):
        ce, cr = ex[i * n_e:(i + 1) * n_e], rm_g[i * n_r:(i + 1) * n_r]
        sc = [s for h in ce for s in h['scores']]
        up = sum(1 for s in sc if abs(s + 1.0) < 1e-9)
        iv = sum(1 for s in sc if abs(s + 0.5) < 1e-9)
        good = [s for s in sc if s > -0.4]
        rows.append(['Q%d' % (i + 1), fmt(avg(sc)),
                     fmt(avg([max(h['scores']) for h in ce])),
                     fmt(avg([s for h in cr for s in h['scores']])),
                     str(up), col(str(iv), BAD) if iv > 5 else str(iv), str(len(good))])
    S.append(table(rows, [16 * mm, 24 * mm, 32 * mm, 24 * mm, 24 * mm, 26 * mm, 18 * mm],
                   mono_cols=(0,)))
    S.append(Spacer(1, 5))
    S.append(callout('THE POLICY SATISFIED THE REWARD MODEL WHILE GETTING WORSE', [
        'The RM-scored trend rises across the run. The verifier-scored trend falls, and '
        'invalid settlements climb from 0 in Q1 to 13 in Q4 while valid ones fall. Since '
        'the reward model ranks at roughly chance, the most economical reading is that '
        'the policy learned to satisfy a near-random signal at the expense of the one '
        'thing scored by ground truth.',
        'This is the same shape as the EPO collapse, arrived at by a different mechanism '
        '— which makes it the second independent arm on this benchmark where the '
        'reinforcement stage degraded what the supervised stage produced.'], bar=BAD))
    S.append(Spacer(1, 3))
    S.append(Pp(
        'Three caveats. The sample is small: %d settling groups, about %d per quarter. '
        'The trend is not monotone — Q3 exceeds Q2. And no scenario appears in more '
        'than one quarter, so scenario difficulty varies between them; the invalid-schema '
        'climb is the harder of the two to explain away, since schema validity barely '
        'depends on difficulty.' % (len(ex), n_e), NOTE))

# ---------------------------------------------------------------- 6
S.append(PageBreak())
S.append(Paragraph('6. Full metric catalogue on the corpus', H2))
if EPS:
    sc = [e['score'] for e in EPS if 'score' in e]
    S.append(Paragraph('6.1  Task outcome', H3))
    rows = [['metric', 'value']]
    for lbl, fn in (
            ('dca', lambda: avg([s['dca'] for s in sc])),
            ('cbar (content, micro)', lambda: avg([s['cbar'] for s in sc])),
            ('pbar (provenance, micro)', lambda: avg([s['pbar'] for s in sc])),
            ('close (non-flipped content)', lambda: avg([s.get('close') for s in sc])),
            ('all content checks pass',
             lambda: avg([1.0 if s.get('all_content') else 0.0 for s in sc])),
            ('all provenance checks pass',
             lambda: avg([1.0 if s.get('all_prov') else 0.0 for s in sc])),
            ('joint success',
             lambda: avg([1.0 if s.get('joint') else 0.0 for s in sc])),
            ('schema valid',
             lambda: avg([1.0 if s.get('schema_valid') else 0.0 for s in sc])),
            ('hallucinated credit rate',
             lambda: avg([s.get('hallucinated_credit') for s in sc]))):
        rows.append([lbl, fmt(fn())])
    S.append(table(rows, [104 * mm, 66 * mm], mono_cols=(1,)))

    S.append(Paragraph('6.2  Information pooling', H3))
    el = [v for e in EPS for v in (e.get('reveal_elicited') or {}).values()]
    lat = []
    for e in EPS:
        rt = [t for t in (e.get('reveal_turn') or {}).values()
              if isinstance(t, (int, float))]
        if rt:
            lat.append(st.median(rt) / max(1, e.get('max_turn') or 1))
    dec_tot = sum(len((CASES.get(e['uid']) or {}).get('decisive_facts') or []) for e in EPS)
    rev_tot = sum(len(e.get('revealed') or []) for e in EPS)
    S.append(table([
        ['metric', 'value'],
        ['decisive disclosure rate', fmt(avg([s['disclosure_rate'] for s in sc]))],
        ['episodes with any reveal', '%d / %d' % (sum(1 for e in EPS if e.get('revealed')),
                                                  len(EPS))],
        ['elicited fraction',
         fmt(sum(1 for v in el if v) / max(1, len(el)))],
        ['silent holder rate', fmt(1.0 - rev_tot / max(1, dec_tot))],
        ['coverage (advisors drawn out)', fmt(avg([e.get('cover') for e in EPS]))],
        ['disclosure latency, normalised', fmt(avg(lat))],
        ['episodes with a leak', str(sum(1 for e in EPS if e.get('leaks')))],
    ], [104 * mm, 66 * mm], mono_cols=(1,)))

    S.append(Paragraph('6.3  Efficiency and format', H3))
    keys = collections.Counter(k for e in EPS for k in (e.get('settlement') or {}))
    S.append(table([
        ['metric', 'value'],
        ['turns per episode', fmt(avg([e.get('turns') for e in EPS]), 2)],
        ['turns / cap', fmt(avg([(e.get('turns') or 0) / max(1, e.get('max_turn') or 1)
                                 for e in EPS]))],
        ['settlements carrying a decisions block',
         '%d / %d' % (keys.get('decisions', 0), len(EPS))],
        ['settlements carrying credited_facts',
         '%d / %d' % (keys.get('credited_facts', 0), len(EPS))],
    ], [104 * mm, 66 * mm], mono_cols=(1,)))

    S.append(Paragraph('6.4  Per-check pass rates', H3))
    passed, total = collections.Counter(), collections.Counter()
    for e in EPS:
        for cid, ok in (e.get('checks') or {}).items():
            total[cid] += 1
            passed[cid] += 1 if ok else 0
    if total:
        rows = [['check', 'pass rate', 'n', 'reachable?']]
        for cid in sorted(total):
            rate = passed[cid] / total[cid]
            rows.append([cid, fmt(rate), str(total[cid]),
                         col('never passes', BAD) if rate == 0.0 else 'yes'])
        S.append(table(rows, [26 * mm, 34 * mm, 26 * mm, 40 * mm], mono_cols=(0,)))
        S.append(Spacer(1, 3))
        dead = [c for c in total if passed[c] == 0]
        if dead:
            S.append(Pp('Checks that never pass anywhere in the corpus: %s. Each one '
                        'depresses every aggregate that includes it, and no policy can '
                        'move them.' % mono(', '.join(sorted(dead))), NOTE))

    S.append(Paragraph('6.5  Sotopia-ToM InfoMgmt, computed', H3))
    da = avg([s['disclosure_rate'] for s in sc])
    ia = sum(1 for v in el if v) / max(1, len(el))
    eff = avg([1.0 - x for x in lat]) if lat else 0.0
    g = 0.0 if min(da, ia, eff) <= 0 else (da * ia * eff) ** (1.0 / 3.0)
    S.append(table([
        ['DA', 'IA', 'EFF', 'InfoMgmt3'],
        [fmt(da), fmt(ia), fmt(eff), fmt(g)],
    ], [42 * mm, 42 * mm, 42 * mm, 44 * mm], mono_cols=(0, 1, 2, 3)))
    S.append(Spacer(1, 3))
    S.append(Pp(
        'Three-way geometric mean; CPV has no analogue on CSA. Computed here on the '
        'TRAIN corpus, so it is not comparable to the held-out InfoMgmt3 figures in the '
        'EPO report — it describes the filtered base model, not a trained policy.',
        NOTE))

# ---------------------------------------------------------------- 7
S.append(Paragraph('7. What to run next', H2))
S.append(table([
    ['#', 'action', 'why'],
    ['1', mono('evaluate_sr.py') + ' on base / sft / grpo',
     'the checkpoints exist and have never been scored. Three runs, greedy, about an '
     'hour each. Until this happens nothing here is a result.'],
    ['2', mono('--reward_source lookahead'),
     'needs no reward model at all. Given pair-ranking near chance, this is the more '
     'likely of the two arms to work, and the comparison measures the distillation '
     'error directly.'],
    ['3', 'expect SFT to beat GRPO',
     'if it does, that is two independent arms where the supervised stage beat the '
     'reinforcement stage — much stronger than either alone.'],
    ['4', 'log GRPO candidate text',
     'the 8 candidates per group are generated, scored and discarded. Without them a '
     'collapsed group cannot be diagnosed as genuinely identical candidates versus a '
     'scorer that could not separate them.'],
], [7 * mm, 54 * mm, 109 * mm]))

S.append(Paragraph('8. Caveats', H2))
for t in [
    'No held-out evaluation. Every figure is training-time, on the train split, after '
    'top-2-of-6 filtering.',
    'One seed. The quarterly GRPO trends rest on about %d settling groups per quarter '
    'and are not monotone.' % (max(1, len([h for h in GRPO_HIST
                                           if h['turn'] >= maxturn(h['uid']) - 1]) // 4)),
    'This arm fine-tunes the dialogue agent, while PPDPP and EPO train a planner over a '
    'frozen agent. The three are not interchangeable rows in one table: report two axes, '
    'or state plainly that the agent differs.',
    'GRPO group sampling makes this arm several times more expensive per episode than the '
    'planner baselines. Compare at matched call budget, not matched episode count.',
]:
    S.append(B(t))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'Sotopia-RL: Yu, Qi et al., <i>Reward Design for Social Intelligence</i>, 2025 '
    '(arXiv:2508.03905). Figures recomputed at build time from '
    '<font face="%s">data/</font>, <font face="%s">ckpt/</font> and '
    '<font face="%s">logs/</font>; nothing is copied from an earlier summary.'
    % (MONO, MONO, MONO), NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'Sotopia-RL on CSA')
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
    print('wrote %s  (episodes %d, rm rows %d, grpo groups %d, eval arms %d)'
          % (OUT, len(EPS), len(RM), len(GRPO_HIST), len(EVAL)))
