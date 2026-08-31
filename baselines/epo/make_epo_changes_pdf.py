"""Build epo-vs-vanilla.pdf -- what this port changes relative to the EPO paper.

Same house style as ../make_report.py. DejaVu is registered from matplotlib's bundled
TTFs because the base-14 faces are WinAnsi-encoded and drop every maths glyph.

    python make_epo_changes_pdf.py
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

OUT = 'epo-vs-vanilla.pdf'
TITLE = 'EPO on CSA: Deviations from the Published Method'


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


def table(rows, widths, mono_cols=(), band=True, size=None):
    st_cell = CELL if size is None else ParagraphStyle('c', parent=CELL, fontSize=size,
                                                       leading=size + 2)
    st_mono = CELLM if size is None else ParagraphStyle('m', parent=CELLM, fontSize=size,
                                                        leading=size + 2)
    data = [[P(c, CELLH) for c in rows[0]]]
    for r in rows[1:]:
        data.append([P(c, st_mono if j in mono_cols else st_cell) for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    style = [('BACKGROUND', (0, 0), (-1, 0), HEAD),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('TOPPADDING', (0, 0), (-1, -1), 4),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
             ('LEFTPADDING', (0, 0), (-1, -1), 5),
             ('RIGHTPADDING', (0, 0), (-1, -1), 5),
             ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
             ('BOX', (0, 0), (-1, -1), 0.5, RULE)]
    if band:
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(('BACKGROUND', (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(style))
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
    'Every place this implementation departs from EPO (Liu et al., ACL 2025), what forced '
    'the departure, and the flag that reverts it. Written so a reviewer can tell which '
    'results are EPO and which are ours.', SUB))
S.append(Paragraph(
    'reference arXiv:2502.12486  |  target CSA, 99/9/42 scenarios  |  budget 700 episodes '
    ' |  agent Qwen2.5-7B-Instruct', META))

# ---------------------------------------------------------------- 1
S.append(Paragraph('1. What is unchanged', H2))
S.append(P(
    'The method is EPO. These are load-bearing and were not touched:'))
for t in [
    'One trainable model. %s is optimised; the dialogue agent and the process reward '
    'model stay frozen.' % mono('LLM_s'),
    'Open-ended natural-language strategy as the action, injected into the agent prompt '
    'as context the agent may override.',
    'Turn-level REINFORCE, %s -- token-averaged within a turn, then turn-averaged.'
    % mono('L = -(1/T) Σ_t A_t · (1/|k_t|) Σ_i log π_θ(a_{t,i})'),
    'Credit assignment is POST-HOC over the finished trajectory, not online per step. '
    'This is the structural difference from PPDPP and it is preserved exactly.',
    'Discounted returns with %s, and a strategy budget of a single short instruction.'
    % mono('γ = 0.99'),
]:
    S.append(B(t))

# ---------------------------------------------------------------- 2
S.append(Paragraph('2. The changes, at a glance', H2))
S.append(P('Reason codes: %s the CSA task demands it &nbsp; %s the 700-episode budget '
           'demands it &nbsp; %s an integrity hole opens without it &nbsp; %s CSA offers '
           'something SOTOPIA cannot.'
           % (why('TASK'), why('BUDGET'), why('INTEGRITY'), why('GAIN'))))
S.append(Spacer(1, 2))
S.append(table([
    ['#', 'Vanilla EPO', 'Here', 'Why', 'Revert with'],
    ['1', 'Action is free text', 'Hybrid <b>tau: sigma</b> — act tag plus open strategy',
     why('TASK'), 'n/a (structural)'],
    ['2', 'PRM is a frozen GPT-4o judge',
     '<b>VerifierPRM</b>: r_t computed from decisive_facts', why('GAIN'), '--prm judge'],
    ['3', 'A_t = R_t / max|R|', 'Group-relative, k=4 rollouts per scenario',
     why('BUDGET'), '--advantage maxabs'],
    ['4', 'Process rewards only', 'Terminal outcome added at t = T', why('BUDGET'),
     '--w_outcome 0'],
    ['5', 'Full fine-tune of Llama3-8B', 'LoRA r=16 on Qwen2.5-7B-Instruct',
     why('BUDGET'), 'n/a'],
    ['6', 'lr 1e-6', 'lr 3e-5', why('BUDGET'), '--lr 1e-6'],
    ['7', 'No KL term reported', 'KL to the frozen base, beta = 0.01', why('BUDGET'),
     '--kl_beta 0'],
    ['8', 'SFT optional; pure RL preferred', 'SFT warm-start mandatory', why('BUDGET'),
     '--adapter ""'],
    ['9', 'Iterative self-play, ~5 rounds', 'None. Chair-only, advisors frozen',
     why('TASK'), 'n/a (see &#167;5)'],
    ['10', 'Two-party alternating dialogue',
     'N-party turn_order with a per-case turn_cap', why('TASK'), 'n/a'],
    ['11', 'No view filtering (no hidden profiles)',
     'Strategist prompt filtered to the chair, and asserted', why('INTEGRITY'), 'n/a'],
    ['12', 'No leak notion',
     'Leak detection extended to CHAIR turns', why('INTEGRITY'), 'n/a'],
    ['13', 'Agent is Llama3-8B / GPT-4o', 'Agent is Qwen2.5-7B-Instruct', why('TASK'),
     '--agent_model'],
    ['14', 'All strategy tokens weighted alike', 'Act-tag tokens upweighted 2x',
     why('TASK'), '--tag_weight 1.0'],
], [7 * mm, 42 * mm, 55 * mm, 20 * mm, 26 * mm], mono_cols=(4,), size=7.0))

# ---------------------------------------------------------------- 3
S.append(PageBreak())
S.append(Paragraph('3. Forced by the task', H2))

S.append(Paragraph('3.1 The action is hybrid, not fully open (change 1)', H3))
S.append(P(
    'EPO\'s action is unconstrained text. On CSA three sites in the environment branch on '
    'the act <i>symbolically</i>, and all three break if the act cannot be recovered:'))
for t in ['the settling act raises the generation budget to 512 tokens and suppresses '
          'sentence trimming, without which the settlement JSON is truncated and every '
          'check fails with it;',
          'an eliciting act drives the addressed-advisor detector, which populates the '
          'coverage measure;',
          'the same test sets %s, the elicited-versus-volunteered distinction the entire '
          'hidden-profile framing rests on.' % mono('reveal_elicited')]:
    S.append(B(t))
S.append(P(
    'Recovering the act from free text would need a classifier, reintroducing exactly the '
    'error source this benchmark exists to remove. The tag costs the policy one word, and '
    '%s records how often it fails to emit one.' % mono('tag_misses')))
S.append(codebox(
    'followup: press Patel for the tilt-bed turning clearance he has not mentioned\n'
    '^^^^^^^^  ^\n'
    '  tau     sigma — free text, <= 20 words, EPO\'s open action space'))

S.append(Paragraph('3.2 Self-play is dropped (change 9)', H3))
S.append(P(
    'EPO alternates two symmetric instances as dialogue partners. CSA is not symmetric: '
    'one chair holds no private facts and makes the decision, N advisors each hold one and '
    'do not. There is no seat-swap that leaves the game the same.'))
S.append(callout('WHY THE NAIVE EXTENSION IS WRONG', [
    'Giving advisors their own trainable strategist on the same reward <b>collapses the '
    'benchmark</b>. An advisor optimised for global %s learns to state its private fact in '
    'its first utterance. Disclosure goes to 1, the profile stops being hidden, %s goes to '
    '0 because nothing was elicited, and the chair needs no skill at all. The measured '
    'improvement would be real and entirely uninformative.'
    % (mono('dca'), mono('reveal_elicited')),
    'This run is therefore <b>EPO without self-play</b>, which is one of the paper\'s own '
    'ablation rows, and is reported as such rather than as full EPO. Seat-rotation (one '
    'strategist trained across role-permuted variants) is the faithful analogue and is '
    'left as the next arm.'], bar=RISK))

# ---------------------------------------------------------------- 4
S.append(PageBreak())
S.append(Paragraph('4. The one change that is a gain, not a compromise', H2))
S.append(Paragraph('4.1 The process reward is computed, not judged (change 2)', H3))
S.append(P(
    'EPO\'s %s reads the finished trajectory and names the turns whose strategies were '
    'critical. That costs a call per episode, drifts with the judge, and on SOTOPIA there '
    'is no way to check it. CSA ships %s, which states for every private fact exactly '
    'which checks it flips, so criticality is a lookup rather than a judgement. The '
    'environment already records the turn each fact surfaced on and whether a question '
    'preceded it.' % (mono('LLM_p'), mono('decisive_facts'))))
S.append(codebox(
    '<font color="#5b5b5b"># D = decisive fact ids;  F(f) = checks f flips;  Φ = ⋃ F(f)</font>\n\n'
    'binary  r_t = 1  if some f ∈ D was ELICITED at turn t\n'
    '            1  if t is the settling turn and dca ≥ τ, schema valid, no leaks\n'
    '            0  otherwise\n\n'
    'graded  r_t = Σ_{f ∈ D} 1[reveal_turn[f]=t]·1[elicited[f]]·|F(f)|/|Φ|  +  λ·dca|_settle',
    bar=KEEP))
S.append(Spacer(1, 4))
S.append(P(
    'Zero model calls, byte-for-byte reproducible, and weighted by how much each fact '
    'actually moves the outcome. A fact volunteered without being drawn out scores '
    'nothing, which is the behaviour the benchmark is about.'))
S.append(P(
    '<b>Both PRMs ship.</b> Running them over the same trajectories yields Cohen\'s kappa '
    'between a judged and a computed process reward — a validation of LLM-as-process-judge '
    'that no environment in the EPO paper can support, because none has executable ground '
    'truth to check the judge against. That is the measurement this port exists to make.'))

# ---------------------------------------------------------------- 5
S.append(Paragraph('5. Forced by the budget', H2))
S.append(P(
    'EPO trains on roughly 2,050 SOTOPIA episodes. This runs 700 episodes over 99 training '
    'scenarios — about 3.4x fewer episodes and, more importantly, far less scenario '
    'diversity for a policy that must generalise to 42 unseen scenarios.'))

S.append(Paragraph('5.1 Group-relative advantage (change 3)', H3))
S.append(P(
    'EPO normalises %s. That is a scaling, not a baseline: it never subtracts anything, so '
    'scenario difficulty dominates the signal and an all-zero episode contributes no '
    'gradient at all. On this benchmark that is not hypothetical — the PPDPP run recorded '
    'disclosure 0.0 on <b>2,718 of 2,982</b> logged steps, and its learning curve was flat '
    'for 1,000 episodes.' % mono('A_t = R_t / max|R_{1:T}|')))
S.append(P(
    'So k = 4 rollouts of the same scenario are centred on the group mean. A rollout that '
    'elicited one more fact than its siblings gets positive advantage even when absolute '
    'reward is low, which is precisely the regime here. %s restores the paper\'s rule for '
    'the ablation.' % mono('--advantage maxabs')))

S.append(Paragraph('5.2 Terminal outcome kept alongside the process reward (change 4)', H3))
S.append(P(
    'Vanilla EPO uses process rewards alone. Under a group-relative baseline that makes an '
    'invalid or leaking episode — all zeros — indistinguishable from one that simply did '
    'nothing. The terminal reward supplies the negative signal, so a fabricated or leaking '
    'settlement is pushed away from rather than ignored. %s reverts to process-only.'
    % mono('--w_outcome 0')))

S.append(Paragraph('5.3 LoRA, learning rate, KL, warm-start (changes 5-8)', H3))
for t in [
    '<b>LoRA r=16 rather than full fine-tuning.</b> A full 8B update on 99 scenarios '
    'memorises. Adapters are also ~150-300 MB against PPDPP\'s 1.05 GB checkpoints, which '
    'matters on a disk that has already killed one run mid-write.',
    '<b>lr 3e-5, not EPO\'s 1e-6.</b> 1e-6 is a full fine-tuning rate; applied to LoRA '
    'adapters it barely moves them. At 4 episodes per update, 700 episodes is only ~175 '
    'optimizer steps, so the rate has to do more work per step.',
    '<b>KL to the frozen base, beta = 0.01.</b> EPO reports no KL term. An unconstrained '
    'LM policy over few steps can collapse onto one strategy string that still scores. '
    'PEFT supplies the reference model by disabling adapters, so this costs no extra '
    'memory. Set %s to reproduce the paper.' % mono('--kl_beta 0'),
    '<b>SFT warm-start is mandatory.</b> EPO reports pure RL beating SFT+RL, at ~3x the '
    'episodes. Here the budget cannot be spent teaching the model to emit a well-formed '
    'tagged line. Pure RL is retained as an ablation (%s), and the comparison is itself a '
    'result.' % mono('--adapter ""'),
]:
    S.append(B(t))

# ---------------------------------------------------------------- 6
S.append(PageBreak())
S.append(Paragraph('6. Forced by integrity', H2))
S.append(Paragraph('6.1 The strategist is a new prompt surface (change 11)', H3))
S.append(P(
    'SOTOPIA has no hidden profiles, so EPO never has to ask what its strategist may see. '
    'On CSA the question is the whole benchmark. The strategist prompt is rendered through '
    'the chair\'s view and asserts, at construction time, that no private fact text and no '
    'oracle field appears in it. A strategist that could see the answer key would score '
    'perfectly while measuring nothing.'))

S.append(Paragraph('6.2 The chair became a leak surface (change 12)', H3))
S.append(callout('DEFECT FOUND IN THE BASELINE, FIXED HERE', [
    'In %s the leak detector runs on advisor turns only — the chair branch breaks out of '
    'the loop before reaching it. With a four-way classifier that was harmless: there is no '
    'channel through which four symbols can smuggle a fact.'
    % mono('ppdpp_csa/env.py:279'),
    'Under EPO the strategist writes free text straight into the chair\'s prompt and is '
    'optimised against a reward that rises with disclosure. <i>"Tell them Lane 2 has the '
    'floor-load certification"</i> is a one-step reward hack that would score as a '
    'successful elicitation. %s now runs on chair turns too. The chair\'s view provably '
    'excludes every private fact, so the existing detector is correct there unmodified.'
    % mono('_note_leaks')], bar=RISK))

S.append(Paragraph('6.3 Shaping retired', H3))
S.append(P(
    'The PPDPP arm carried potential-based shaping over disclosure, elicitation and '
    'coverage, justified by Ng et al. (1999). That argument assumes the potential is a '
    'function of state alone, layered on a fixed reward stream. The verifier PRM rewards '
    'elicited disclosure of decisive facts — the same event the disclosure potential '
    'measures — so keeping both counts it twice and the invariance guarantee no longer '
    'applies. Shaping is absent from this environment entirely.'))

# ---------------------------------------------------------------- 7
S.append(Paragraph('7. Reading the result honestly', H2))
S.append(P(
    'The PPDPP baseline did not learn. Over 1,000 episodes on 42 test scenarios:'))
S.append(Spacer(1, 2))
S.append(table([
    ['Metric', 'Floor', 'epoch 0', 'epoch 6', 'Verdict'],
    ['SR', '&#8212;', '0.000', '0.000', 'never fires'],
    ['disclosure_rate', '0.000', '0.077', '0.089', 'flat'],
    ['episodes with any reveal', '0/42', '8/42', '9/42', 'flat'],
    ['dca', '0.000', '&#8212;', '0.209', 'above floor, but see below'],
    ['cbar', '0.000', '0.245', '0.245', 'flat'],
    ['pbar', '0.000', '0.048', '0.048', 'flat'],
    ['reward', '&#8212;', '-0.094', '-0.095', 'constant'],
], [46 * mm, 20 * mm, 20 * mm, 20 * mm, 54 * mm], mono_cols=(0, 1, 2, 3)))
S.append(Spacer(1, 6))
S.append(P(
    'Two consequences for how this port is reported. First, the primary claim is <b>a '
    'non-flat curve</b>, not a headline number: any monotone movement in disclosure across '
    'checkpoints is the result, because the comparison is against a flat line.'))
S.append(P(
    'Second, %s is a poor headline metric here. It reads 0.209 while disclosure reads '
    '0.089, meaning most passing decisive checks pass from shared context without the '
    'private fact ever surfacing. The metrics that isolate the capability under test are '
    'disclosure rate, the elicited fraction, and %s conditioned on disclosure being '
    'non-zero. Report those first.' % (mono('dca'), mono('dca'))))
S.append(P(
    'Guardrails that must not regress: schema validity at 0.976, leaks at zero, and calls '
    'per episode at about 14. The last is not cosmetic — a planner that wins by making more '
    'calls has not won, and EPO adds one strategist call per chair turn on top of the '
    'agent\'s.'))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'EPO: Liu, Wang, Li, Wu, Ma, Kong, Huang, Jiao &amp; Zhang, <i>EPO: Explicit Policy '
    'Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</i>, ACL 2025 '
    '(arXiv:2502.12486). Shaping invariance: Ng, Harada &amp; Russell, ICML 1999. '
    'Hidden profiles: Stasser &amp; Titus, 1985. Baseline figures recomputed from '
    '<font face="%s">ppdpp_csa/tmp/csa/eval_result/Record-epoch-{0,6}-*.txt</font>.'
    % MONO, NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'EPO on CSA — deviations from the paper')
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
