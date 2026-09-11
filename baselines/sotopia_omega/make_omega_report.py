"""Build the SOTOPIA-Omega-on-CSA report.

Recomputes every figure from what is on disk at build time. Before a corpus exists it
emits the design, the corpus-A baseline that corpus B must beat, and the adversarial
ceiling analysis -- all computed, none asserted. Once corpora and records appear it fills
in the results.

    python make_omega_report.py
"""
import ast
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

import config
import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import data_csa as data_csa

OUT = 'sotopia-omega-csa-report.pdf'
TITLE = 'SOTOPIA-Ω on CSA: Stall-Triggered Corpus Construction'
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ data
def load_jsonl(p):
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


def load_records(p):
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


def corpus_stats(eps):
    if not eps:
        return {}
    d = [e['score']['dca'] for e in eps if 'score' in e]
    r = [e['score']['disclosure_rate'] for e in eps if 'score' in e]
    modes = [m for e in eps for m in (e.get('modes') or [])]
    return {'episodes': len(eps), 'scenarios': len({e['uid'] for e in eps}),
            'dca_mean': sum(d) / len(d) if d else float('nan'),
            'dca_median': st.median(d) if d else float('nan'),
            'dca_max': max(d) if d else float('nan'),
            'disc_mean': sum(r) / len(r) if r else float('nan'),
            'any_reveal': sum(1 for e in eps if e.get('revealed')),
            'slow_turns': sum(1 for m in modes if m == 'slow'),
            'all_turns': len(modes),
            'stalled': sum(1 for e in eps if e.get('stalled_at') is not None)}


# corpus A: plain self-play. sotopia_rl already generated exactly this from the same
# base model with the same verifier filter, so it is the number corpus B must beat and
# there is no reason to regenerate it.
CORPUS_A = corpus_stats(load_jsonl(os.path.join(HERE, '..', 'sotopia_rl', 'data',
                                                'episodes-train.jsonl')))
CORPUS_B = corpus_stats(load_jsonl(os.path.join(paths.DATA, 'corpus-B-train.jsonl')))
CORPUS_C = corpus_stats(load_jsonl(os.path.join(paths.DATA, 'corpus-C-train.jsonl')))
RECORDS = {}
for f in glob.glob(os.path.join(paths.LOGS, 'Record-*-test.txt')):
    tag = re.sub(r'^Record-|-test\.txt$', '', os.path.basename(f))
    r = load_records(f)
    if r:
        RECORDS[tag] = r
HAVE_CORPUS = bool(CORPUS_B or CORPUS_C)
HAVE_RESULTS = bool(RECORDS)


def ceiling_distribution():
    """If ONE advisor withholds, what dca ceiling remains? Computed from the dataset
    alone -- decisive_facts names both the owner and the checks each fact flips."""
    out = []
    for case in data_csa.load_raw():
        flips = {d['fact_id']: set(d.get('flips') or [])
                 for d in (case.get('decisive_facts') or [])}
        phi = set().union(*flips.values()) if flips else set()
        if not phi:
            continue
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            lost = set()
            for fid, f in case['private_facts'].items():
                if f['owner'] == a['agent_id'] and fid in flips:
                    lost |= flips[fid]
            out.append(len(phi - lost) / len(phi))
    return out


CEIL = ceiling_distribution()


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
    'SOTOPIA-Ω (Zhang et al., ACL 2025) adapted to CSA. Data synthesis plus '
    'supervised fine-tuning, no reinforcement learning anywhere — the fourth '
    'distinct category in this project, and the one that speaks to the result the other '
    'arms keep producing.', SUB))
S.append(Paragraph(
    'corpora %d/2 built  |  eval arms %d  |  expert %s  |  student %s  |  all figures '
    'recomputed from files'
    % (int(bool(CORPUS_B)) + int(bool(CORPUS_C)), len(RECORDS),
       config.Defaults.expert_model, config.Defaults.student_model), META))

if not HAVE_CORPUS and not HAVE_RESULTS:
    S.append(callout('NO CORPUS AND NO RESULTS YET', [
        'Nothing has been generated or evaluated for this arm. Everything below is the '
        'design, the corpus-A baseline that corpus B must beat, and the adversarial '
        'ceiling analysis — all computed from files that already exist, none of it '
        'asserted.',
        'Re-running %s once a corpus is generated fills in the results automatically.'
        % mono('make_omega_report.py')], bar=WARN))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. Keep the mechanism, swap the protocol', H2))
S.append(Pp(
    'Omega’s thesis is not "negotiate". It is: let the expert talk, detect that it '
    'has stalled, switch it into a structured mode, and fine-tune on what results. The '
    'four-stage negotiation protocol is Omega’s instantiation of that intervention '
    'because SOTOPIA is a negotiation benchmark.'))
S.append(Pp(
    'CSA is not a negotiation. The chair and the advisors want the same outcome; the '
    'advisors are not opponents holding leverage, they simply have not been asked. '
    'Estimating an opponent’s utility is meaningless when utilities are aligned. So '
    'the protocol is replaced by one with the same shape, pointed at elicitation. '
    'Forcing CSA into a negotiation frame would be adapting the dataset to the method.'))
S.append(Spacer(1, 2))
S.append(table([
    ['Omega (negotiation)', 'Here (elicitation)'],
    ['own utility over the matters', 'which decision fields are still unsupported'],
    ['guess the opponent’s utility', 'whose ROLE makes them likely to hold it'],
    ['draft a proposal', 'draft the targeted question'],
    ['confirm the proposal', 'put the settlement on the record'],
], [80 * mm, 90 * mm]))
S.append(Spacer(1, 5))
S.append(Pp(
    'The three reasoning stages are scaffolding, never labels. Only the utterance the '
    'chair actually spoke enters the transcript and the corpus, so the student learns to '
    'produce it without the scaffold. That is where the distillation happens.', NOTE))

S.append(Paragraph('1.1 The stall trigger is computed, not judged', H3))
S.append(Pp(
    'Omega detects deadlock with an LLM scoring goal articulation, proposal quality and '
    'agreement against a 7.5 threshold. On CSA the thing that should be happening is '
    'decisive facts reaching the chair, and whether that happened is a set membership '
    'test.'))
S.append(codebox(
    'stalled  <=>  step >= stall_after  AND  |revealed & decisive| has not grown\n'
    '              for stall_patience consecutive chair turns\n\n'
    'current settings:  stall_after = %d   stall_patience = %d'
    % (config.Defaults.stall_after, config.Defaults.stall_patience), bar=GOOD))
S.append(Spacer(1, 4))
S.append(Pp(
    'Deterministic, zero model calls, nothing to tune against a threshold. Once stalled '
    'the chair stays in slow mode for the rest of the episode, matching Omega, whose '
    'agent does not revert once it goes hard.', NOTE))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. Three corpora, and why more than one', H2))
S.append(Pp(
    'Omega’s headline — a 7B trained on the corpus beats the expert that '
    'produced it — confounds <b>strategy injection</b> with <b>the teacher being '
    'larger than the student</b>. Separating them takes one extra run.'))
S.append(Spacer(1, 2))
S.append(table([
    ['corpus', 'expert', 'isolates', 'status'],
    ['A', 'plain self-play, Qwen2.5-7B', 'baseline',
     col('exists', GOOD) + ' (from sotopia_rl)'],
    ['B', 'Qwen2.5-7B, strategy-injected', 'strategy injection alone',
     col('built', GOOD) if CORPUS_B else col('not built', BAD)],
    ['C', 'frontier model via API, injected', 'injection + teacher strength',
     col('built', GOOD) if CORPUS_C else col('not built', BAD)],
], [16 * mm, 46 * mm, 56 * mm, 52 * mm], mono_cols=(0,)))
S.append(Spacer(1, 5))
S.append(Pp(
    'B vs A is Omega’s mechanism. C vs B is the distillation gradient. C vs A is the '
    'headline and confounds both — Omega’s own confound, made visible here '
    'rather than hidden. The student is Qwen2.5-7B throughout, so the comparison to the '
    'other three arms holds.'))
S.append(Spacer(1, 3))
S.append(callout('CORPUS C CARRIES A DATA ADVANTAGE NO OTHER ARM HAS', [
    'Every other baseline generated its data from Qwen2.5-7B self-play. If Omega’s '
    'corpus comes from a frontier model, a win partly means "the frontier model is better '
    'than Qwen2.5-7B", which is not a finding. Run C alongside B, never instead of it.'],
    bar=WARN))

# ---------------------------------------------------------------- 3
S.append(PageBreak())
S.append(Paragraph('3. Corpus A: the baseline B must beat', H2))
if CORPUS_A:
    S.append(Pp(
        'sotopia_rl already generated plain self-play from the same base model with the '
        'same verifier filter, so corpus A exists and does not need regenerating. These '
        'are its numbers, on the 99 TRAIN scenarios — filtered top-2-of-6, so they '
        'reflect selection as well as behaviour, and are not comparable to held-out '
        'figures from the other arms.'))
    S.append(Spacer(1, 2))
    rows = [['', 'A (plain)', 'B (injected)', 'C (frontier)']]

    def _c(k, d=3, pct=False):
        vals = []
        for cs in (CORPUS_A, CORPUS_B, CORPUS_C):
            v = cs.get(k) if cs else None
            vals.append(('%d' % v if isinstance(v, int) and not pct else fmt(v, d))
                        if v is not None else '&#8212;')
        return vals

    for label, key, dd in (('episodes', 'episodes', 0), ('scenarios', 'scenarios', 0),
                           ('dca mean', 'dca_mean', 3), ('dca median', 'dca_median', 3),
                           ('dca max', 'dca_max', 3),
                           ('disclosure mean', 'disc_mean', 3),
                           ('episodes with a reveal', 'any_reveal', 0),
                           ('rollouts that stalled', 'stalled', 0),
                           ('chair turns in slow mode', 'slow_turns', 0)):
        rows.append([label] + _c(key, dd))
    S.append(table(rows, [52 * mm, 36 * mm, 36 * mm, 36 * mm], mono_cols=(1, 2, 3)))
    S.append(Spacer(1, 5))
    S.append(Pp(
        'Corpus A reaches dca %s and disclosure %s with %d of %d episodes surfacing '
        'something. Corpus B has to beat that using the same model, the same filter and '
        'the same budget — the only difference being that stalled episodes get the '
        'structured intervention.'
        % (mono(fmt(CORPUS_A['dca_mean'])), mono(fmt(CORPUS_A['disc_mean'])),
           CORPUS_A['any_reveal'], CORPUS_A['episodes'])))
    S.append(Spacer(1, 3))
    S.append(callout('THE PROBE DECIDES WHETHER ANY OF THIS IS WORTH RUNNING', [
        'The method rests on one assumption: a 7B WITH the scaffold produces better turns '
        'than the same 7B without. If that is false, corpus B is no better than the '
        'numbers above and the fine-tuning is pointless.',
        '%s forces ten scenarios through both modes and prints the disclosure delta. It '
        'costs minutes; the corpus pass costs hours. The script stops and says so if slow '
        'mode does not improve disclosure.'
        % mono('python generate_omega.py --probe 10')], bar=WARN))
else:
    S.append(Pp('Corpus A not found. It is %s from the sotopia_rl arm.'
                % mono('sotopia_rl/data/episodes-train.jsonl')))

# ---------------------------------------------------------------- 4
S.append(Paragraph('4. Results', H2))
if HAVE_RESULTS:
    rows = [['arm', 'n', 'SR', 'dca', 'dca/ceiling', 'disclosure', 'calls']]
    for tag, recs in sorted(RECORDS.items()):
        def avg(fn, rs=recs):
            v = [fn(r) for r in rs if fn(r) is not None]
            return sum(v) / len(v) if v else float('nan')
        rows.append([tag, str(len(recs)),
                     fmt(avg(lambda r: 1.0 if r.get('done') == 1 else 0.0)),
                     fmt(avg(lambda r: (r.get('score') or {}).get('dca'))),
                     fmt(avg(lambda r: r.get('dca_norm'))),
                     fmt(avg(lambda r: (r.get('score') or {}).get('disclosure_rate'))),
                     fmt(avg(lambda r: r.get('n_calls')), 1)])
    S.append(table(rows, [34 * mm, 14 * mm, 18 * mm, 18 * mm, 26 * mm, 26 * mm, 18 * mm],
                   mono_cols=(0,)))
else:
    S.append(Pp('No evaluation records. The commands that fill this section in:'))
    S.append(codebox(
        'python generate_omega.py --probe 10                    # decides the rest\n'
        'python generate_omega.py --split train --k 6 --keep 2  # corpus B\n'
        'python train_sft_om.py --episodes data/corpus-B-train.jsonl --epochs 3\n'
        'python evaluate_om.py --adapter "" --tag base\n'
        'python evaluate_om.py --adapter ckpt/sft --tag omega-B', bar=WARN))

# ---------------------------------------------------------------- 5
S.append(Paragraph('5. The adversarial variant', H2))
S.append(Pp(
    'One randomly chosen advisor becomes evasive: vague, deferring, never volunteering '
    'exact figures, and never lying — a licence to lie would break the benchmark '
    'rather than test it. Omega’s deadlock mechanism fits this setting far better '
    'than the cooperative one, because the stall has a cause.'))
if CEIL:
    S.append(Spacer(1, 2))
    S.append(table([
        ['quantity', 'value'],
        ['advisor choices across the corpus', str(len(CEIL))],
        ['dca ceiling, mean', fmt(sum(CEIL) / len(CEIL))],
        ['dca ceiling, median', fmt(st.median(CEIL))],
        ['dca ceiling, min / max', '%s / %s' % (fmt(min(CEIL)), fmt(max(CEIL)))],
        ['choices leaving the ceiling at 1.0',
         col('%d of %d' % (sum(1 for c in CEIL if c > 0.999), len(CEIL)), BAD)],
    ], [96 * mm, 74 * mm], mono_cols=(1,)))
    S.append(Spacer(1, 5))
    S.append(Pp(
        'Every advisor in this corpus holds a decisive fact, so there is no harmless '
        'choice of opponent: silencing any of them makes some checks unreachable. The '
        'ceiling is computable per episode because %s names both the owner and the checks '
        'each fact controls, so a low score never has to be ambiguous between "the policy '
        'failed" and "the task was impossible". Report %s, and only against runs with the '
        'same %s.' % (mono('decisive_facts'), mono('dca_norm'), mono('--opponent'))))
S.append(Spacer(1, 3))
S.append(callout('THREE THINGS THIS VARIANT NEEDS, ALL HANDLED', [
    '<b>The opponent is exempt from the leak gate.</b> It is playing a role; its leaks go '
    'to %s and never invalidate the episode. Without this the arm would poison its own '
    'reward, because scripted evasion that happens to echo an unseen fact would trip the '
    'integrity gate.' % mono('opponent_leaks'),
    '<b>The evasion margin is thin.</b> Careful evasion and real disclosure sit close '
    'together around the 0.35 threshold. %s prints the gap on every run; hand-label about '
    'fifty flagged disclosures before trusting any adversarial number.' % mono('selftest.py'),
    '<b>Run the no-opponent version first.</b> It is the control that makes these numbers '
    'interpretable — without it you cannot separate "the intervention helped" from '
    '"the adversary made everything worse" from "the ceiling dropped".'], bar=BAD))

# ---------------------------------------------------------------- 6
S.append(PageBreak())
S.append(Paragraph('6. Deviations from the paper', H2))
S.append(table([
    ['#', 'Paper', 'Here', 'Why'],
    ['1', 'stall via an LLM score, threshold 7.5', 'set membership on decisive facts',
     'computable, zero calls, nothing to tune'],
    ['2', 'four-stage negotiation protocol', 'four-stage elicitation protocol',
     'CSA has aligned goals and no opponent to model'],
    ['3', 'Qwen2.5-72B / GPT-4 expert', 'Qwen2.5-7B, or an API expert',
     '72B does not fit the hardware; same-size isolates injection from teacher size'],
    ['4', 'LLM-scored quality filter', 'the verifier',
     'deterministic and already built'],
    ['5', 'two-party, agent1 / agent2', 'N-party with a turn order and a cap',
     'CSA has 3-5 agents with asymmetric roles'],
    ['6', 'corpus keeps every turn', 'same, with a per-turn mode flag',
     'pre-stall turns CAUSED the deadlock; %s makes dropping them a filter rather than a '
     'regeneration' % mono('--mode_filter slow')],
], [7 * mm, 40 * mm, 44 * mm, 69 * mm]))

S.append(Paragraph('7. Caveats', H2))
for t in [
    'Nothing has been generated or evaluated. Every number in this document comes from '
    'the dataset itself or from the sotopia_rl corpus; not one is a Omega result.',
    'With expert == student this is self-improvement through filtered, strategy-injected '
    'self-play — the STaR / rejection-sampling family — not distillation. '
    '"Student beats teacher" is definitionally unavailable and must not be claimed.',
    'Corpus A’s figures are on the TRAIN split and top-2-of-6 filtered. They are the '
    'right comparison for corpus B, and the wrong comparison for held-out results from '
    'the other arms.',
    'The generator warns if nothing ever stalls (the intervention never fired, so the '
    'corpus is plain self-play) and if everything stalls (slow mode is always on, so the '
    'adaptive part is untested). Read those lines before trusting a corpus.',
]:
    S.append(B(t))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'SOTOPIA-Ω: Zhang et al., <i>Dynamic Strategy Injection Learning and Social '
    'Instruction Following Evaluation for Social Agents</i>, ACL 2025 '
    '(arXiv:2502.15538). Reimplemented from the paper and the public repository. Figures '
    'recomputed at build time from <font face="%s">data/</font>, '
    '<font face="%s">logs/</font> and the CSA scenarios; nothing is copied from an '
    'earlier summary.' % (MONO, MONO), NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'SOTOPIA-Omega on CSA')
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
    print('wrote %s  (corpora built: %d/2, eval arms: %d)'
          % (OUT, int(bool(CORPUS_B)) + int(bool(CORPUS_C)), len(RECORDS)))
