"""Build the CSA baselines rerun runbook PDF.

The status column is not hard-coded. It is derived at build time from what is actually on
disk -- which record files exist, which arms they cover, what the reward-model metadata
says -- so re-running this after you fix something produces a document that reflects the
new state instead of the old one.

    python make_rerun_runbook.py
"""
import ast
import glob
import json
import os
import re

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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
ART = os.environ.get('CSA_ARTIFACTS_DIR') or os.path.join(os.path.dirname(HERE),
                                                          'csa-artifacts')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csa-rerun-runbook.pdf')
TITLE = 'CSA Baselines: Rerun Runbook'


# ------------------------------------------------------------------ live state
def records(path):
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


def g(pat):
    """Matches in the repo and in the artifacts archive. Records are results and live
    outside the repo, so a status read from HERE alone would report every arm empty."""
    out = []
    for root in dict.fromkeys((HERE, ART)):
        out += sorted(glob.glob(os.path.join(root, pat)))
    return out


def jload(p):
    for root in dict.fromkeys((HERE, ART)):
        try:
            return json.load(open(os.path.join(root, p), encoding='utf-8'))
        except Exception:                            # noqa: BLE001
            continue
    return None


def state():
    """What exists on disk right now, per baseline."""
    s = {}

    ppd = g('ppdpp/tmp/csa/eval_result/Record-epoch-*.txt')
    rewards = {m.group(1) for f in ppd
               for m in [re.search(r'-(critic|verifier)-seed', f)] if m}
    s['ppdpp'] = {'records': len(ppd), 'rewards': sorted(rewards),
                  'epochs': sorted({int(re.search(r'epoch-(\d+)', f).group(1))
                                    for f in ppd})}

    epo = g('epo/logs/Record-*seed1-ep*.txt')
    s['epo'] = {'records': len(epo),
                'checkpoints': sorted({int(re.search(r'-ep(\d+)\.txt$', f).group(1))
                                       for f in epo
                                       if re.search(r'-ep(\d+)\.txt$', f)}),
                'seeds': len(g('epo/logs/*seed*-history.jsonl'))}

    rm = jload('sotopia_rl/ckpt/rm/rm_meta.json') or {}
    s['sotopia_rl'] = {'records': len(g('sotopia_rl/logs/Record-*.txt')),
                       'rm_rank': rm.get('best_pair_rank'),
                       'rm_best_epoch': rm.get('best_epoch'),
                       'grpo_history': len(g('sotopia_rl/logs/grpo-*-history.jsonl')),
                       'has_sft': bool(g('sotopia_rl/ckpt/sft/sft_meta.json'))}

    s['sotopia_tom'] = {'records': len(g('sotopia_tom/logs/Record-tom-*.txt')),
                        'summary': bool(g('sotopia_tom/logs/summary*.json'))}
    s['sotopia_omega'] = {'records': len(g('sotopia_omega/logs/Record-*.txt')),
                          'corpus': len(g('sotopia_omega/data/corpus-*.jsonl')),
                          'has_sft': bool(g('sotopia_omega/ckpt/sft/*'))}
    return s


S = state()


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
        pdfmetrics.registerFont(TTFont('DejaVuMono',
                                       os.path.join(d, 'DejaVuSansMono.ttf')))
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
BODY = ParagraphStyle('BODY', parent=ss['Normal'], fontName=SANS, fontSize=9,
                      leading=12.5, textColor=INK, spaceAfter=6)
BUL = ParagraphStyle('BUL', parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=3.5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8, leading=11, textColor=MUTED)
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontName=MONO, fontSize=7.4, leading=10,
                      textColor=INK, leftIndent=0, spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName=SANS, fontSize=7.4,
                      leading=9.4, textColor=INK)
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
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return t


def rule():
    t = Table([['']], colWidths=[170 * mm], rowHeights=[0.5], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), RULE)]))
    return t


# ------------------------------------------------------------------ verdicts
RERUN = col('RERUN', BAD)
RUN = col('RUN &#8212; never ran', BAD)
PARTIAL = col('PARTIAL', WARN)
KEEP = col('KEEP', GOOD)


def verdicts():
    v = {}
    p = S['ppdpp']
    v['ppdpp'] = (RERUN if 'critic' not in p['rewards'] else KEEP)
    v['epo'] = KEEP if S['epo']['records'] >= 5 else PARTIAL
    r = S['sotopia_rl']
    v['sotopia_rl'] = (RUN if r['records'] == 0 else
                       PARTIAL if (r['rm_rank'] or 0) < 0.6 else KEEP)
    v['sotopia_tom'] = RUN if S['sotopia_tom']['records'] == 0 else KEEP
    v['sotopia_omega'] = RUN if S['sotopia_omega']['records'] == 0 else KEEP
    return v


V = verdicts()


# ------------------------------------------------------------------ document
def story():
    st = []
    st.append(P(TITLE, H1))
    st.append(P('What to re-run, what to leave alone, and the exact command for each. '
                'The status column below is read off the filesystem when this PDF is '
                'built, so it is current as of this build.', SUB))
    st.append(P('build state &#183; ppdpp %d records (%s) &#183; epo %d records &#183; '
                'sotopia_rl %d records &#183; sotopia_tom %d &#183; sotopia_omega %d'
                % (S['ppdpp']['records'], '/'.join(S['ppdpp']['rewards']) or 'none',
                   S['epo']['records'], S['sotopia_rl']['records'],
                   S['sotopia_tom']['records'], S['sotopia_omega']['records']), META))

    # ---------------------------------------------------------- triage
    st.append(P('0. Triage &#8212; read this first', H2))
    st.append(P('Re-running a baseline is only worth the GPU time when the existing '
                'numbers answer a different question from the one you are asking. That '
                'happens in exactly three ways here: the arm was never run at all, the '
                'arm was run under a configuration that is not the method (PPDPP), or '
                'the arm was run but a component inside it is at chance (the Sotopia-RL '
                'reward model). Everything else keeps its numbers.'))
    st.append(table([
        ['baseline', 'status', 'verdict', 'why', 'cost'],
        ['PPDPP',
         '%d records, reward = %s' % (S['ppdpp']['records'],
                                      '/'.join(S['ppdpp']['rewards']) or 'none'),
         V['ppdpp'],
         'Every stored record used the shared verifier reward. PPDPP’s own dense '
         'critic reward &#8212; the thing the method actually proposes &#8212; has no '
         'records. The flat learning curve currently measures the reward config, not '
         'the algorithm.',
         '~6 GPU-h'],
        ['EPO',
         '%d records, checkpoints %s' % (S['epo']['records'],
                                         ', '.join(map(str, S['epo']['checkpoints']))),
         V['epo'],
         'Training completed, all checkpoints reproduce with 0 mismatches, and the '
         'collapse after ep0 is a genuine finding rather than a bug. Nothing to redo '
         '&#8212; only the metrics layer is new.',
         'metrics only'],
        ['Sotopia-RL',
         'training artifacts present, %d eval records' % S['sotopia_rl']['records'],
         V['sotopia_rl'],
         'The reward model ranks pairs at %.3f (0.500 is chance) with best_epoch=%s, so '
         'GRPO optimised noise; and no test-split evaluation was ever written, so the '
         'arm has no results to compare.'
         % (S['sotopia_rl']['rm_rank'] or float('nan'),
            S['sotopia_rl']['rm_best_epoch']),
         '~5 GPU-h'],
        ['Sotopia-ToM',
         '%d records' % S['sotopia_tom']['records'], V['sotopia_tom'],
         'Prompting-only arm. Code and self-tests pass but it has never been executed '
         'against the test split, so there is nothing to score.',
         '~3 GPU-h'],
        ['SOTOPIA-Omega',
         '%d records, %d corpus files' % (S['sotopia_omega']['records'],
                                          S['sotopia_omega']['corpus']),
         V['sotopia_omega'],
         'Corpus generation, SFT and evaluation have all never been run. This is the '
         'only arm that needs its full pipeline from the top.',
         '~9 GPU-h'],
    ], [22 * mm, 30 * mm, 22 * mm, 74 * mm, 18 * mm], mono_cols=(1,)))

    st.append(Spacer(1, 6))
    st.append(callout('The one rule for deciding', [
        'Re-run an arm when the numbers on disk answer a different question from the one '
        'you are asking. Do not re-run because a script changed, because the metrics '
        'layer grew, or because time has passed &#8212; the record files are the '
        'evaluation, and section 6 recomputes every metric from them without touching a '
        'GPU. EPO is the worked example: its metric catalogue expanded substantially and '
        'it still needs no re-run.'], bar=HEAD))

    st.append(Spacer(1, 4))
    st.append(P('Order matters only in two places: each baseline’s own steps run '
                'top to bottom, and section 6 runs last because it reads every '
                'baseline’s records. The five baselines are independent of each '
                'other and can be run in any order, or on separate GPUs at the same '
                'time.', NOTE))

    st.append(PageBreak())

    # ---------------------------------------------------------- 1 PPDPP
    st.append(P('1. PPDPP &#8212; %s' % V['ppdpp'], H2))
    st.append(P('Records on disk cover epochs %s, all with the verifier reward. '
                'PPDPP’s native configuration is <b>--csa_reward critic</b>, which '
                'is the default in <b>run.py</b> and was overridden for every run. The '
                'result is that the paper’s dense per-turn critic signal has never '
                'been tested on CSA, while the flat curve currently attributed to PPDPP '
                'came from the sparse shared verifier.'
                % ', '.join(map(str, S['ppdpp']['epochs']))))
    st.append(P('1a. Run the missing arm A (critic reward)', H3))
    st.append(codebox(
        'cd ppdpp\n'
        '\n'
        '# --csa_reward critic is the DEFAULT, but pass it explicitly so the\n'
        '# filename records which arm produced the file.\n'
        'python run.py \\\n'
        '    --csa_reward critic \\\n'
        '    --seed 1 \\\n'
        '    --max_epochs 6\n'
        '\n'
        '# writes tmp/csa/eval_result/Record-epoch-{0,2,4,6}-...-critic-seed1.txt',
        bar=BAD))
    st.append(P('1b. Confirm the gradient is no longer degenerate', H3))
    st.append(P('The run prints <b>zero-gradient updates: X%</b> beside every learning '
                'curve. Under the verifier reward that fraction was high, because a '
                'sparse binary reward is constant across most episodes and a constant '
                'raw reward carries no cross-episode signal. Under the critic reward it '
                'should drop sharply. If it does not, the problem is the algorithm on '
                'this task and not the reward wiring &#8212; report that, it is a '
                'result.', NOTE))
    st.append(P('1c. Re-score the old verifier records', H3))
    st.append(P('The stored verifier scores predate the current verifier: they carry no '
                '<b>close</b> key and provenance resolution never fired. Re-scoring '
                'moved pbar from 0.048 to 0.071. Do this so arm A and arm B are judged '
                'by the same verifier.'))
    st.append(codebox(
        'cd ppdpp\n'
        'python compute_all_metrics.py --rescore\n'
        '\n'
        '# compute_all_metrics.py falls back to a torch-free case loader, so this\n'
        '# step needs no GPU and no CUDA install.', bar=WARN))

    # ---------------------------------------------------------- 2 EPO
    st.append(P('2. EPO &#8212; %s' % V['epo'], H2))
    st.append(P('Do not re-train. The 700-episode run is complete, every checkpoint in '
                '{%s} re-evaluates to the recorded numbers with zero mismatches, and the '
                'policy collapse after ep0 (unique strategies 96.5%% &#8594; 37.0%%, act '
                'entropy 1.110 &#8594; 0.000, 89 tag misses, 20 empty strategy strings) '
                'is a real property of the run rather than a defect in the harness. It '
                'is the arm’s headline finding: <b>the SFT-only checkpoint ep0 is '
                'the best one</b>.'
                % ', '.join(map(str, S['epo']['checkpoints']))))
    st.append(P('Only re-run these, and only because the metrics layer is new:', H3))
    st.append(codebox(
        'cd epo\n'
        'python make_epo_report.py          # rebuilds the PDF from the records\n'
        '\n'
        '# optional: a second seed strengthens the collapse claim from\n'
        '# "it happened" to "it reproduces". ~5 GPU-h.\n'
        'python run_epo.py --seed 99 --episodes 700', bar=GOOD))
    st.append(callout('Why no re-run, in one line', [
        'The records already contain every field sections E through H need &#8212; '
        'domain, num_agents, per-component verifier scores, call counts, strategies. '
        'Adding metrics does not invalidate an evaluation.'], bar=GOOD))

    st.append(PageBreak())

    # ---------------------------------------------------------- 3 Sotopia-RL
    st.append(P('3. Sotopia-RL &#8212; %s' % V['sotopia_rl'], H2))
    st.append(P('Two separate problems, and the second one is the blocking one.'))
    st.append(B('<b>The reward model is at chance.</b> <b>best_pair_rank = %.3f</b> '
                'against a 0.500 floor, with <b>best_epoch = %s</b> &#8212; the first '
                'epoch was the best, meaning it never learned. Exact-scored GRPO signal '
                'declined across quartiles (0.340 &#8594; 0.149) while invalid schemas '
                'climbed from 0 to 13, even though the RM-scored signal rose. The policy '
                'was optimising the reward model’s error.'
                % (S['sotopia_rl']['rm_rank'] or float('nan'),
                   S['sotopia_rl']['rm_best_epoch'])))
    st.append(B('<b>There is no evaluation.</b> %d record files exist, so the arm '
                'currently contributes nothing to any comparison table.'
                % S['sotopia_rl']['records']))
    st.append(P('3a. Rebuild the attributed-reward data, then retrain the RM', H3))
    st.append(codebox(
        'cd sotopia_rl\n'
        '\n'
        '# 1. attribution data. Only redo this if you change attribution.py;\n'
        '#    data/rm-train.jsonl already holds 832 rows.\n'
        'python make_rm_data.py\n'
        '\n'
        '# 2. retrain the reward model. The old run peaked at epoch 0, which is\n'
        '#    what an under-fit-then-diverge curve looks like: try a lower LR and\n'
        '#    more epochs before concluding the signal is not learnable.\n'
        'python train_rm.py \\\n'
        '    --epochs 8 \\\n'
        '    --lr 5e-6 \\\n'
        '    --holdout 0.15 \\\n'
        '    --grad_checkpointing\n'
        '\n'
        '# 3. STOP and read ckpt/rm/rm_meta.json before spending GPU time on GRPO.\n'
        'python -c "import json;m=json.load(open(\'ckpt/rm/rm_meta.json\'));'
        'print(m[\'best_pair_rank\'], m[\'best_epoch\'])"', bar=BAD))
    st.append(callout('Gate: do not run GRPO on a chance-level reward model', [
        'If <b>best_pair_rank</b> is still below roughly 0.60, or <b>best_epoch</b> is '
        'still 0, the reward model has not learned and GRPO against it will reproduce '
        'the same result. In that case run step 3b with <b>--reward_source lookahead</b> '
        'instead: it commits the candidate, lets one advisor answer and reads the '
        'disclosure detector, so no reward model is involved at all. A reward model that '
        'does not rank is a finding worth reporting, not a step to push past.'], bar=BAD))
    st.append(P('3b. GRPO &#8212; only past the gate', H3))
    st.append(codebox(
        'python train_grpo.py \\\n'
        '    --adapter ckpt/sft \\\n'
        '    --rm ckpt/rm \\\n'
        '    --reward_source rm \\\n'
        '    --groups 700 \\\n'
        '    --seed 1 \\\n'
        '    --grad_checkpointing\n'
        '\n'
        '# the honest fallback if the RM never ranks:\n'
        'python train_grpo.py --adapter ckpt/sft --reward_source lookahead \\\n'
        '    --groups 700 --seed 1 --grad_checkpointing', bar=WARN))
    st.append(P('3c. Evaluate &#8212; required either way', H3))
    st.append(P('This is the step that has never been run. Without it the arm has no '
                'records and cannot appear in any table in section 6.'))
    st.append(codebox(
        '# the untrained chair, as the floor\n'
        'python evaluate_sr.py --adapter "" --split test --tag base\n'
        '\n'
        '# SFT only, the fair mid-point\n'
        'python evaluate_sr.py --adapter ckpt/sft --split test --tag sft\n'
        '\n'
        '# the trained policy\n'
        'python evaluate_sr.py --adapter ckpt/grpo/final --split test --tag grpo\n'
        '\n'
        '# writes logs/Record-*.txt, which section 6 discovers automatically\n'
        'python make_sr_report.py', bar=BAD))

    st.append(PageBreak())

    # ---------------------------------------------------------- 4 ToM
    st.append(P('4. Sotopia-ToM &#8212; %s' % V['sotopia_tom'], H2))
    st.append(P('Nothing is broken here; it simply has never been executed. This is the '
                'cheapest arm in the study &#8212; prompting only, no training, no '
                'gradient step &#8212; so it is the best value per GPU-hour of anything '
                'in this runbook.'))
    st.append(P('Run all five arms in one pass', H3))
    st.append(codebox(
        'cd sotopia_tom\n'
        '\n'
        'python selftest.py                 # ~1 min, catches prompt regressions\n'
        '\n'
        'python run_tom.py \\\n'
        '    --strategies stripped basic cot tom_coach tom_belief \\\n'
        '    --split test \\\n'
        '    --compare\n'
        '\n'
        'python make_tom_report.py', bar=BAD))
    st.append(callout('Check this before trusting the comparison', [
        'The <b>tom_coach</b> and <b>tom_belief</b> arms were identical prompts at one '
        'point and the difference between them was therefore zero by construction. They '
        'now carry distinct headers, and <b>selftest.py</b> asserts it. If the five arms '
        'come back with suspiciously close scores, diff the rendered prompts before '
        'writing that theory-of-mind prompting does not help.'], bar=WARN))
    st.append(P('InfoMgmt = [DA &#183; IA &#183; (1 &#8722; CPV) &#183; EFF]^(1/4) is '
                'computed by <b>metrics_tom.py</b> and is the arm’s own headline. '
                'It is a geometric mean, so any single component at zero zeroes the '
                'whole score &#8212; if InfoMgmt comes back at 0.000 for an arm, read '
                'the four components before concluding the arm failed.', NOTE))

    # ---------------------------------------------------------- 5 Omega
    st.append(P('5. SOTOPIA-Omega &#8212; %s' % V['sotopia_omega'], H2))
    st.append(P('The full pipeline has never been run: corpus generation, SFT, then '
                'evaluation, in that order. Each step consumes the previous step’s '
                'output, so unlike the other baselines this one cannot be partially '
                'skipped.'))
    st.append(P('5a. Probe first &#8212; do not launch the full generation blind', H3))
    st.append(codebox(
        'cd sotopia_omega\n'
        'python selftest.py\n'
        '\n'
        '# --probe runs a handful of scenarios and prints the stall-detection\n'
        '# decisions. If stalls never fire, the fast/slow switch never engages and\n'
        '# the corpus is plain rollouts with extra steps.\n'
        'python generate_omega.py --split train --probe 5 --expert local', bar=WARN))
    st.append(P('5b. Generate the corpus', H3))
    st.append(codebox(
        'python generate_omega.py \\\n'
        '    --split train \\\n'
        '    --expert local \\\n'
        '    --stall_after 3 \\\n'
        '    --stall_patience 2 \\\n'
        '    --seed 1\n'
        '\n'
        'python generate_omega.py --split valid --expert local --seed 1\n'
        '\n'
        '# --restart resumes an interrupted generation instead of starting over',
        bar=BAD))
    st.append(P('5c. SFT the student, then evaluate', H3))
    st.append(codebox(
        'python train_sft_om.py \\\n'
        '    --mode_filter all \\\n'
        '    --grad_checkpointing\n'
        '\n'
        '# the untrained student is the floor this arm is measured against\n'
        'python evaluate_om.py --adapter "" --split test --tag base\n'
        'python evaluate_om.py --adapter ckpt/sft --split test --tag omega\n'
        '\n'
        '# adaptive lets the student choose fast vs slow at inference time;\n'
        '# run it as a second arm rather than instead of the fast one\n'
        'python evaluate_om.py --adapter ckpt/sft --split test \\\n'
        '    --eval_mode adaptive --tag omega-adaptive\n'
        '\n'
        'python make_omega_report.py', bar=BAD))
    st.append(callout('Information leakage, the thing to watch here', [
        'The single-opponent adaptation means one agent is selected and the rest '
        'negotiate against it. That agent sees the other agents’ disclosures, so a '
        'corpus built from its trajectories can encode private facts the student should '
        'not have. Run the <b>--opponent withhold</b> evaluation as a control: if the '
        'trained student loses most of its advantage when the opponent view is '
        'withheld, the gain was leakage.'], bar=BAD))

    st.append(PageBreak())

    # ---------------------------------------------------------- 6 metrics
    st.append(P('6. Cross-arm metrics &#8212; run last, no GPU', H2))
    st.append(P('One command scores every arm that has records. It discovers them by '
                'globbing the five log directories, so an arm you re-ran appears '
                'automatically and an arm you did not touch keeps its existing numbers. '
                'Pure stdlib &#8212; no torch, no CUDA, no judge model. Seconds, not '
                'hours.'))
    st.append(codebox(
        'cd baselines\n'
        'python compute_extended_metrics.py\n'
        '\n'
        '# writes extended_metrics.json alongside the printed tables\n'
        'python compute_extended_metrics.py --out extended_metrics.json', bar=GOOD))
    st.append(P('What it now produces', H3))
    st.append(table([
        ['section', 'metrics', 'note'],
        ['Headline',
         'dca and disclosure with bootstrap 95% CI (10k resamples); mean calls; '
         'dca per 100 calls; dca per 10k tokens; ceiling; floor-normalised gain',
         'Cost-normalised columns stop a method winning purely by spending more '
         'inference.'],
        ['Paired',
         'win / tie / loss per scenario, two-sided sign test, Cliff’s delta with '
         'Romano bands, bootstrap CI on the paired difference, Holm-corrected p, '
         'full N&#215;N head-to-head win matrix',
         'The delta CI is the interval to quote: both arms see the same 42 scenarios, '
         'so the paired difference is much tighter than either marginal interval. Holm '
         'controls the family-wise error across the 16 tests.'],
        ['E',
         'gold recovery from content checks; exact-match accuracy; per-field error rate; '
         'PE@10 / PE@20 / PE@30; MAE; RMSE; MAPE; MPE; tolerance accuracy; R&#178;; '
         'Pearson; Spearman; Kendall',
         '355 of 789 decision fields yield a gold but only 20 parse as numeric, so every '
         'numeric figure prints beside its n and the exact-match column is the one that '
         'carries weight.'],
        ['F',
         'distinct-1/2/3; utterance length mean, median, sd; role-adherence violation '
         'rate; private-view leakage episodes and total',
         'Language quality and the safety property the benchmark actually cares about.'],
        ['G',
         'act distribution and entropy in bits against the max; act bigrams; number of '
         'acts used; return mean and variance; success rate',
         'Entropy at 0.000 bits is how the EPO collapse shows up numerically.'],
        ['H',
         'dca by domain with macro average; dca by table size; full per-scenario spread '
         '(sd, min, quartiles, max, IQR, fraction at floor and ceiling); every verifier '
         'subscore with its own CI; disclosure-to-accuracy correlation; difficulty '
         'tertiles set by the baseline',
         'Answers the three questions a reviewer asks first: is the gain one domain, '
         'what shape is the distribution, and where in the difficulty range does the '
         'gain live.'],
    ], [20 * mm, 72 * mm, 78 * mm]))

    st.append(Spacer(1, 6))
    st.append(P('What it still cannot compute, and why', H3))
    st.append(table([
        ['metric', 'blocker'],
        ['Expected calibration error, bin confusion matrix',
         'Settlements carry no confidence value and no bins are defined, so there is '
         'nothing to calibrate.'],
        ['G-Eval, human rubric, Srel', 'Needs a judge model or human annotators.'],
        ['BLEU / ROUGE', 'No reference justifications exist in CSA.'],
        ['Sdiv, steered vs unsteered perplexity',
         'Needs a GPU forward pass over both policies. Add it as an optional pass once '
         'the arms above are re-run and the adapters are resident anyway.'],
        ['KL to the SFT policy',
         'Needs both policies on a GPU at once. Same optional pass.'],
        ['Sample efficiency (episodes to 90% of final joint success)',
         'Joint success is 0.000 for every arm, so the 90% target is undefined. This is '
         'a property of the benchmark difficulty, not a gap in the tooling.'],
    ], [58 * mm, 112 * mm]))

    st.append(Spacer(1, 8))
    st.append(rule())
    st.append(Spacer(1, 4))
    st.append(P('Sequencing, if you have two GPUs: put PPDPP arm A on one and the '
                'Omega pipeline on the other, since those are the two longest jobs. '
                'Sotopia-ToM fits in the gaps on either card. Sotopia-RL’s gate at '
                'step 3a means you find out early whether 3b is worth starting. '
                'Section 6 runs on the CPU at the end and takes seconds.', NOTE))
    return st


def main():
    SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=16 * mm,
                      title=TITLE).build(story())
    print('wrote %s' % OUT)


if __name__ == '__main__':
    main()
