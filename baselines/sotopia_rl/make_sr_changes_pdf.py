"""Build sotopia-rl-vs-vanilla.pdf -- deviations from the published Sotopia-RL method.

DejaVu is registered from matplotlib's bundled TTFs: the base-14 faces are WinAnsi and
drop every maths glyph in the reward section.

    python make_sr_changes_pdf.py
"""
import os

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

OUT = 'sotopia-rl-vs-vanilla.pdf'
TITLE = 'Sotopia-RL on CSA: Deviations from the Published Method'


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
        pdfmetrics.registerFont(TTFont('DejaVuMono-Bold',
                                       os.path.join(d, 'DejaVuSansMono-Bold.ttf')))
        addMapping('DejaVuMono', 0, 0, 'DejaVuMono')
        addMapping('DejaVuMono', 1, 0, 'DejaVuMono-Bold')
        return 'DejaVu', 'DejaVu-Bold', 'DejaVuMono', 'DejaVuMono-Bold'
    except Exception as e:                           # noqa: BLE001
        print('DejaVu unavailable (%s); maths glyphs will drop' % e)
        return 'Helvetica', 'Helvetica-Bold', 'Courier', 'Courier-Bold'


SANS, SANSB, MONO, MONOB = _fonts()

INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5b5b5b')
RULE = colors.HexColor('#d0d0d0')
HEAD = colors.HexColor('#22333b')
BAND = colors.HexColor('#f2f4f5')
CODEBG = colors.HexColor('#f6f7f8')
ACCENT = colors.HexColor('#1f5e8c')
KEEP = colors.HexColor('#2c6b55')
WARN = colors.HexColor('#8e5314')
RISK = colors.HexColor('#9c2f3b')
REASON = {'TASK': ACCENT, 'BUDGET': WARN, 'INTEGRITY': RISK, 'GAIN': KEEP}

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
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontName=MONO, fontSize=7.6, leading=10.4,
                      textColor=INK, leftIndent=0, spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName=SANS, fontSize=7.4, leading=9.4,
                      textColor=INK)
CELLM = ParagraphStyle('CELLM', parent=CELL, fontName=MONO, fontSize=7.0)
CELLH = ParagraphStyle('CELLH', parent=CELL, fontName=SANSB, textColor=colors.white)
CALL = ParagraphStyle('CALL', parent=BODY, fontSize=8.4, leading=11.6, spaceAfter=5)


def P(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BUL, bulletText='•')


def mono(t):
    return '<font face="%s">%s</font>' % (MONO, t)


def why(tag):
    return '<font face="%s" color="#%s"><b>%s</b></font>' % (
        SANSB, REASON[tag].hexval()[2:], tag)


def table(rows, widths, mono_cols=(), size=7.2):
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


def codebox(text, bar=ACCENT, width=170 * mm):
    t = Table([[XPreformatted(text, CODE)]], colWidths=[width], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), CODEBG),
                           ('BOX', (0, 0), (-1, -1), 0.5, RULE),
                           ('LINEBEFORE', (0, 0), (0, -1), 2.2, bar),
                           ('LEFTPADDING', (0, 0), (-1, -1), 8),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                           ('TOPPADDING', (0, 0), (-1, -1), 6),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    return t


def callout(title, paras, bar=RISK, width=170 * mm):
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


S = []
S.append(Paragraph(TITLE, H1))
S.append(Paragraph(
    'Every place this implementation departs from Sotopia-RL (Yu, Qi et al., 2025), what '
    'forced it, and the flag that reverts it. Written so a reviewer can tell which results '
    'are the published method and which are ours.', SUB))
S.append(Paragraph(
    'reference arXiv:2508.03905  |  target CSA, 99/9/42 scenarios  |  agents '
    'Qwen2.5-7B-Instruct  |  no API key required', META))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. What is unchanged', H2))
S.append(P('The method is Sotopia-RL. These are load-bearing and were not touched:'))
for t in [
    'Three stages: offline utterance-level reward design, then behaviour cloning, then a '
    'distilled reward model, then single-turn online GRPO.',
    'The reward form %s (Eq. 3): an episode-level score times a per-utterance attribution, '
    'per dimension, then normalised and averaged (Eq. 4).' % mono('r_t = G . A(a_t, tau)'),
    'The attributor sees the FINISHED episode; the reward model sees only the prefix. That '
    'asymmetry is the whole point of distillation and is preserved exactly.',
    'The reward model is a scalar regressor trained with MSE (Eq. 5), not a preference '
    'model. The labels are attributed scalars, so no preference pairs exist or are needed.',
    'GRPO is single-turn: prompt is the dialogue history, completion is the next '
    'utterance, advantage is standardised within the group of G samples.',
    'The dialogue agent being trained is the one that speaks; there is no planner and no '
    'explicit reasoning trace, matching the paper\'s "utterance generation without '
    'explicit reasoning".',
]:
    S.append(B(t))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. The changes, at a glance', H2))
S.append(P('Reason codes: %s the CSA task demands it &nbsp; %s the budget demands it '
           '&nbsp; %s an integrity hole opens without it &nbsp; %s CSA offers something '
           'SOTOPIA cannot.' % (why('TASK'), why('BUDGET'), why('INTEGRITY'), why('GAIN'))))
S.append(Spacer(1, 2))
S.append(table([
    ['#', 'Vanilla Sotopia-RL', 'Here', 'Why', 'Revert with'],
    ['1', 'GPT-4o attributor assigns A(a_t, tau)',
     'A computed from decisive_facts and reveal_turn', why('GAIN'), 'n/a (structural)'],
    ['2', 'Episode score G from an LLM judge', 'G from the executable verifier',
     why('GAIN'), 'n/a'],
    ['3', 'Rubric REL / KNO / GOAL', 'POOL / USE / COVER', why('TASK'), '--weights'],
    ['4', 'COVER-analogue = social rubric dimension',
     'advisor must be addressed AND then disclose', why('TASK'), 'n/a'],
    ['5', 'Eq.4 min-max, k unbound',
     'normalised across the DATASET, so G survives', why('GAIN'),
     '--within_episode_norm'],
    ['6', 'no integrity notion',
     'leak or invalid schema zeroes the whole episode', why('INTEGRITY'), 'gate flag'],
    ['7', 'BC on GPT-4o self-play',
     'BC on Qwen self-play, filtered by executable checks', why('BUDGET'), '--min_dca'],
    ['8', 'reward model is required',
     'optional: lookahead scores candidates exactly', why('GAIN'),
     '--reward_source rm'],
    ['9', 'act/eliciting label unavailable',
     'is_eliciting read from the utterance itself', why('TASK'), 'n/a'],
    ['10', 'full fine-tune, Qwen2.5-7B', 'LoRA r=16', why('BUDGET'), 'n/a'],
    ['10b', 'agent, policy and RM are separate models',
     'ONE backbone; the three roles are adapters over it', why('BUDGET'),
     'n/a (structural)'],
    ['11', 'SFT 500 epochs, RM 30 epochs', 'SFT 3, RM 4, early-stopped', why('BUDGET'),
     '--epochs'],
    ['12', 'GRPO 16 generations', '8 generations', why('BUDGET'), '--group 16'],
    ['13', 'RM selected on MSE', 'selected on within-episode pair ranking', why('TASK'),
     'n/a'],
    ['14', 'two symmetric agents self-play',
     'chair only; advisors frozen', why('INTEGRITY'), 'n/a (see &#167;5)'],
    ['15', 'settling turn scored like any other',
     'settlement scored EXACTLY by the verifier', why('GAIN'), 'n/a'],
], [7 * mm, 43 * mm, 54 * mm, 20 * mm, 26 * mm], mono_cols=(4,)))

# ---------------------------------------------------------------- 3
S.append(PageBreak())
S.append(Paragraph('3. The reward is computed, not judged', H2))
S.append(P(
    'Sotopia-RL\'s Eq. 3 multiplies an episode-level score by an LLM attributor\'s guess at '
    'how much each utterance contributed. Both factors are GPT-4o calls, both drift, and on '
    'SOTOPIA there is nothing to check either against. CSA states the answer: '
    '%s says which checks each private fact controls, and the transcript says which turn '
    'drew it out.' % mono('decisive_facts')))
S.append(codebox(
    '<font color="#5b5b5b"># Phi = every check some decisive fact controls;  '
    'passed = those that passed</font>\n\n'
    'A_pool(t)  = |U flips(f) : f ELICITED at turn t| / |Phi|\n'
    'A_use(t)   = |{c in Phi : c passed, c in flips(f), f elicited at t}| / |Phi|\n'
    'A_use(settle) += |{c in Phi : c passed, no elicited fact behind it}| / |Phi|\n'
    'A_cover(t) = |advisors addressed at t that later disclosed| / |advisors|\n\n'
    'r_{t,d}    = G_d . A_d(t) . gate(tau)        gate = 0 on a leak or invalid schema',
    bar=KEEP))
S.append(Spacer(1, 4))
for t in [
    '<b>Union, not sum.</b> The flips lists overlap, so summing lets one turn score above '
    '1. Measured: summing produced a maximum of 1.222; the union gives exactly 1.0.',
    '<b>A_use only pays out if the settlement used the information.</b> If the settlement '
    'is wrong, no checks pass and every eliciting turn earns zero. The structure enforces '
    'that without a separate rule.',
    '<b>Volunteered facts earn nothing.</b> Credit requires the disclosure to follow an '
    'eliciting turn, which is the behaviour the benchmark exists to measure.',
    '<b>Provenance resolution is applied before scoring</b>, as in the other arms. Without '
    'it this arm runs a harsher provenance rule and the comparison breaks.',
]:
    S.append(B(t))
S.append(Spacer(1, 2))
S.append(P('Measured over 297 real episodes (1,206 chair turns): 30.8%% of turns carry a '
           'non-zero label, mean 0.056, std 0.135, max 0.99 &#8212; dense enough to '
           'regress on, and every value reproducible without a single API call.', NOTE))

S.append(Paragraph('3.1 Eq. 4 normalisation: an ambiguity worth resolving', H3))
S.append(P(
    'The paper writes %s without binding k. If k ranges over the turns of ONE EPISODE then '
    'G_d, constant across turns, cancels exactly:' % mono('min_k r_{k,d}')))
S.append(codebox(
    '(G.A_t - G.min A) / (G.max A - G.min A)  =  (A_t - min A) / (max A - min A)'))
S.append(Spacer(1, 4))
S.append(P(
    'and the episode score stops affecting the labels at all &#8212; a strong episode and a '
    'failed one produce identical targets. Ranging k over the dataset keeps G_d, which here '
    'is the verifier\'s own judgement and the most trustworthy signal available. '
    'Dataset-level is the default; %s reproduces the other reading for the ablation.'
    % mono('--within_episode_norm')))

# ---------------------------------------------------------------- 4
S.append(PageBreak())
S.append(Paragraph('4. The rubric', H2))
S.append(P('SOTOPIA\'s seven social dimensions do not apply to a hidden-profile decision '
           'task, but CSA has its own, and two of the three map almost directly:'))
S.append(Spacer(1, 2))
S.append(table([
    ['Paper', 'Here', 'What it measures', 'Non-zero'],
    ['KNO (knowledge seeking)', 'POOL', 'did private information surface at all', '13.4%'],
    ['GOAL (goal completion)', 'USE', 'did it land in checks that passed', '28.3%'],
    ['REL (relationship)', 'COVER', 'were the holders actually drawn out', '12.9%'],
    ['&#8212;', 'aggregate r_t', 'equal-weight average, as the paper uses', '30.8%'],
], [40 * mm, 22 * mm, 82 * mm, 20 * mm]))
S.append(Spacer(1, 6))
S.append(callout('WHY COVER IS NOT "ADVISORS MENTIONED BY NAME"', [
    'The obvious definition &#8212; count advisors whose surname appears in a chair turn '
    '&#8212; saturates on turn 0, because the chair opens by greeting everyone. Measured '
    'that way it fired on 34.1% of turns and dominated the aggregate, rewarding politeness '
    'rather than elicitation. Worse, it was the densest of the three, so it was doing most '
    'of the work.',
    'Here an advisor counts only once it has been addressed AND has subsequently disclosed '
    'something, crediting the turn that ASKED. Density drops to 12.9% and the aggregate '
    'from 51.7% to 30.8%, which is the correct trade: less signal, but signal that tracks '
    'the behaviour under test.'], bar=WARN))

S.append(Paragraph('4.1 Eliciting-ness has to be read from the text', H3))
S.append(P(
    'The planner baselines know whether a chair turn was an eliciting act because a planner '
    'chose the act. Here there is no planner, so %s applies a lexical rule: the turn must '
    'address an advisor by name AND carry an interrogative. Requiring the name is what '
    'separates drawing information out of someone from thinking aloud.'
    % mono('detectors_sr.is_eliciting')))
S.append(P(
    'This is a new source of measurement error, so it is measured rather than assumed. '
    'Against 650 turns carrying published act annotations: <b>accuracy 0.817, precision '
    '0.845, recall 0.803</b>. selftest.py recomputes this on every run.'))

# ---------------------------------------------------------------- 5
S.append(Paragraph('5. Self-play, and why it is chair-only', H2))
S.append(P(
    'Sotopia-RL self-plays two symmetric agents. CSA is not symmetric: one chair holds no '
    'private facts and makes the decision, N advisors each hold one and do not.'))
S.append(callout('THE NAIVE EXTENSION DESTROYS THE BENCHMARK', [
    'Training advisors on the same reward makes them state their private fact in the first '
    'utterance. Disclosure goes to 1, the hidden profile stops being hidden, the elicited '
    'fraction goes to 0 because nothing was elicited, and the chair needs no skill at all. '
    'The measured improvement would be real and entirely uninformative.',
    'Advisors are therefore frozen and only the chair is trained. That is a genuine '
    'restriction of the published method and is reported as one.'], bar=RISK))

S.append(Paragraph('6. The comparison caveat', H2))
S.append(callout('THIS ARM IS NOT INTERCHANGEABLE WITH THE PLANNER BASELINES', [
    'Sotopia-RL fine-tunes the dialogue agent. The planner baselines hold the agent frozen '
    'and train a planner on top &#8212; precisely so the comparison isolates the planner. '
    'Putting all three in one table as though they answered the same question would be '
    'wrong.',
    'Report two axes ("planner over a frozen agent" versus "fine-tuned agent, no planner"), '
    'or state plainly that the agent differs. Call counts and the frozen-advisor setup stay '
    'comparable; goal-completion numbers do not. Note also that GRPO\'s group sampling '
    'makes this arm several times more expensive per episode, so compare at matched budget '
    'rather than matched episode count.'], bar=RISK))

S.append(Paragraph('7. Reproducibility', H2))
S.append(P(
    'This package imports no code from the other baselines. It rebuilds the split from the '
    'raw scenarios and reimplements the verifier, then checks both against the published '
    'artefacts as data:'))
S.append(Spacer(1, 2))
S.append(table([
    ['Check', 'Result'],
    ['split reproduces the published 99/9/42, order included', 'exact'],
    ['verifier reproduces cbar / pbar / disclosure on 297 episodes', 'exact, all 297'],
    ['chair prompt free of private facts and oracle fields, 150 scenarios', 'clean'],
    ['advisors see only their own private fact', 'clean'],
    ['attribution stays within [0, 1]', 'max 0.9865'],
    ['is_eliciting vs 650 annotated act labels', 'acc 0.817'],
], [104 * mm, 36 * mm]))
S.append(Spacer(1, 8))
S.append(Paragraph(
    'Sotopia-RL: Yu, Qi, et al., <i>Sotopia-RL: Reward Design for Social Intelligence</i>, '
    '2025 (arXiv:2508.03905). GRPO: Shao et al., 2024. Hidden profiles: Stasser &amp; '
    'Titus, 1985. All figures in this document were recomputed from the CSA corpus by '
    '<font face="%s">selftest.py</font>; none are copied from an earlier summary.' % MONO,
    NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'Sotopia-RL on CSA — deviations from the paper')
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
    print('wrote', OUT)
