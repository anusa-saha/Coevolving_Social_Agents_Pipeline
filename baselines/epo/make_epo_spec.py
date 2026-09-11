"""Build the EPO-on-CSA design-spec PDF.

Same house style as make_report.py, with one difference: the base-14 Helvetica/Courier
faces are WinAnsi-encoded and drop every Greek and maths glyph in the formulation
section, so DejaVu is registered from matplotlib's bundled TTFs instead.
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
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle,
                                XPreformatted)

OUT = 'epo-csa-design-spec.pdf'
TITLE = 'EPO on CSA: Planner Substitution Design Spec'

# ------------------------------------------------------------------ fonts
def _register_dejavu():
    """Unicode-capable family, or fall back to the base-14 faces."""
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__),
                         'mpl-data', 'fonts', 'ttf')
        faces = [('DejaVu', 'DejaVuSans.ttf', 0, 0),
                 ('DejaVu-Bold', 'DejaVuSans-Bold.ttf', 1, 0),
                 ('DejaVu-Oblique', 'DejaVuSans-Oblique.ttf', 0, 1),
                 ('DejaVu-BoldOblique', 'DejaVuSans-BoldOblique.ttf', 1, 1)]
        for name, fn, _b, _i in faces:
            pdfmetrics.registerFont(TTFont(name, os.path.join(d, fn)))
        for name, _fn, b, i in faces:
            addMapping('DejaVu', b, i, name)
        pdfmetrics.registerFont(TTFont('DejaVuMono',
                                       os.path.join(d, 'DejaVuSansMono.ttf')))
        pdfmetrics.registerFont(TTFont('DejaVuMono-Bold',
                                       os.path.join(d, 'DejaVuSansMono-Bold.ttf')))
        addMapping('DejaVuMono', 0, 0, 'DejaVuMono')
        addMapping('DejaVuMono', 1, 0, 'DejaVuMono-Bold')
        return 'DejaVu', 'DejaVu-Bold', 'DejaVuMono', 'DejaVuMono-Bold'
    except Exception as e:
        print('DejaVu unavailable (%s); maths glyphs will be dropped' % e)
        return 'Helvetica', 'Helvetica-Bold', 'Courier', 'Courier-Bold'


SANS, SANSB, MONO, MONOB = _register_dejavu()

# ------------------------------------------------------------------ palette
INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5b5b5b')
RULE = colors.HexColor('#d0d0d0')
HEAD = colors.HexColor('#22333b')
BAND = colors.HexColor('#f2f4f5')
CODEBG = colors.HexColor('#f6f7f8')

ACCENT = colors.HexColor('#1f5e8c')
KEEP = colors.HexColor('#2c6b55')
MODIFY = colors.HexColor('#8e5314')
REPLACE = colors.HexColor('#9c2f3b')
RETIRE = colors.HexColor('#6e7883')

STATUS = {'KEEP': KEEP, 'MODIFY': MODIFY, 'REPLACE': REPLACE,
          'NEW': ACCENT, 'RETIRE': RETIRE}

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontName=SANSB,
                    fontSize=17, leading=21, textColor=INK, alignment=TA_LEFT,
                    spaceAfter=2)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontName=SANS,
                     fontSize=9, leading=13, textColor=MUTED, spaceAfter=4)
META = ParagraphStyle('META', parent=SUB, fontName=MONO, fontSize=7.6,
                      leading=11, spaceAfter=12)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontName=SANSB,
                    fontSize=12, leading=15, textColor=HEAD,
                    spaceBefore=14, spaceAfter=5)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontName=SANSB,
                    fontSize=9.5, leading=12, textColor=INK,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle('BODY', parent=ss['Normal'], fontName=SANS,
                      fontSize=9, leading=12.5, textColor=INK, spaceAfter=6)
BULLET = ParagraphStyle('BULLET', parent=BODY, leftIndent=10, bulletIndent=2,
                        spaceAfter=3.5)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8, leading=11, textColor=MUTED)
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontName=MONO,
                      fontSize=7.6, leading=10.4, textColor=INK,
                      leftIndent=0, spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName=SANS,
                      fontSize=7.6, leading=9.6, textColor=INK)
CELLM = ParagraphStyle('CELLM', parent=CELL, fontName=MONO, fontSize=7.2)
CELLH = ParagraphStyle('CELLH', parent=CELL, fontName=SANSB, textColor=colors.white)
CALL = ParagraphStyle('CALL', parent=BODY, fontSize=8.4, leading=11.6, spaceAfter=5)


def P(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BULLET, bulletText='•')


def mono(t):
    """Inline mono span."""
    return '<font face="%s">%s</font>' % (MONO, t)


def chip(status):
    return '<font face="%s" color="#%s"><b>%s</b></font>' % (
        SANSB, STATUS[status].hexval()[2:], status)


def table(rows, widths, first_col_mono=False, band_body=True):
    data = [[P(c, CELLH) for c in rows[0]]]
    for r in rows[1:]:
        cells = []
        for j, c in enumerate(r):
            cells.append(P(c, CELLM if (first_col_mono and j == 0) else CELL))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    st = [('BACKGROUND', (0, 0), (-1, 0), HEAD),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('TOPPADDING', (0, 0), (-1, -1), 4),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
          ('LEFTPADDING', (0, 0), (-1, -1), 5),
          ('RIGHTPADDING', (0, 0), (-1, -1), 5),
          ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
          ('BOX', (0, 0), (-1, -1), 0.5, RULE)]
    if band_body:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(('BACKGROUND', (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(st))
    return t


def codebox(text, bar=ACCENT, width=170 * mm):
    """Shaded, left-barred code block. Markup-aware, so escape &, <, > in `text`."""
    t = Table([[XPreformatted(text, CODE)]], colWidths=[width], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CODEBG),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('LINEBEFORE', (0, 0), (0, -1), 2.2, bar),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def callout(title, paras, bar=REPLACE, width=170 * mm):
    inner = [Paragraph(title, ParagraphStyle(
        'CT', parent=CALL, fontName=SANSB, fontSize=8,
        textColor=bar, spaceAfter=4))]
    inner += [Paragraph(p, CALL) for p in paras]
    t = Table([[inner]], colWidths=[width], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BAND),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('LINEBEFORE', (0, 0), (0, -1), 2.2, bar),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


S = []

# ================================================================= masthead
S.append(Paragraph('EPO on CSA: Planner Substitution Design Spec', H1))
S.append(Paragraph(
    "Replacing PPDPP's four-way discrete planner with EPO's open-ended strategic "
    'reasoner, holding the CSA environment, verifier, splits and metric catalogue '
    'byte-identical so the two planners are directly comparable.', SUB))
S.append(Paragraph(
    'base ppdpp_csa/ &nbsp;&nbsp;|&nbsp;&nbsp; target EPO (Liu et al., ACL 2025; '
    'arXiv:2502.12486) &nbsp;&nbsp;|&nbsp;&nbsp; splits 99 / 9 / 42 scenarios', META))

# ================================================================= 1
S.append(Paragraph('1. Thesis: one substitution, not a rewrite', H2))
S.append(P(
    'The CSA port was built so that everything below the planner is planner-agnostic. '
    'The environment walks %s, filters every prompt through %s, records disclosures and '
    'leaks lexically, and scores settlements with executable checks &#8212; none of which '
    'knows or cares what produced the per-turn conditioning signal. PPDPP supplies that '
    'signal as one of four symbols from a RoBERTa classifier. EPO supplies it as a short '
    'natural-language strategy from a trainable LLM.'
    % (mono('turn_order'), mono("case['views']"))))
S.append(P(
    'So the port is genuinely narrow: keep %s dynamics, %s in full, %s splits and the '
    'record schema; replace %s policy and optimiser; add a process-reward module and a '
    'strategist prompt. The interesting work is not the swap &#8212; it is that CSA can '
    '<b>compute</b> EPO\'s process reward instead of judging it, which is the one thing '
    'SOTOPIA cannot do.'
    % (mono('env.py'), mono('verifier.py'), mono('export_csa.py'), mono('agent.py'))))

# ================================================================= 2
S.append(Paragraph('2. What EPO specifies', H2))
S.append(P('From arXiv:2502.12486, for the parts the port has to honour.', NOTE))
for t in [
    '<b>Three models, one trained.</b> %s (strategic reasoner) is optimised; %s (dialogue '
    'agent) stays frozen to preserve its generality; %s (process reward model, GPT-4o) is '
    'frozen.' % (mono('LLM_s'), mono('LLM_d'), mono('LLM_p')),

    '<b>Open action space.</b> %s is free text, prompted to a single phrase or sentence '
    'under ten words. The agent then generates %s and may override a strategy it disagrees '
    'with.' % (mono('a_t = LLM_s(s_sys, G, h_{1:t-1}, a_{1:t-1}, x_t)'),
               mono('y_t = LLM_d(d_sys, G, h_{1:t-1}, a_{1:t}, x_t)')),

    '<b>Turn-level REINFORCE</b> with token-averaged log-probability and max-absolute '
    'normalisation of the discounted return, %s, &#947; = 0.99. Stated as '
    'algorithm-agnostic; no KL term is reported.' % mono('A_t = R_t / max|R_{1:T}|'),

    '<b>Post-hoc process rewards.</b> %s reads the <i>completed</i> trajectory plus its '
    'outcome score and returns the indices of turns whose strategies were critical to the '
    'goal; %s for those turns, 0 elsewhere.' % (mono('LLM_p'), mono('r_t = 1')),

    '<b>Iterative self-play</b> &#8212; two EPO instances alternate as partners, retrain, '
    'repeat; reported to plateau around five iterations.',

    '<b>Pure RL preferred over SFT warm-start</b>, to avoid over-optimising on cloned '
    'behaviour.',
]:
    S.append(B(t))
S.append(Spacer(1, 3))
S.append(P(
    'Two of these &#8212; the post-hoc reward and the self-play loop &#8212; are structural '
    'changes to the training loop rather than to the environment, and they are where the '
    'port needs real decisions.'))

# ================================================================= 3
S.append(PageBreak())
S.append(Paragraph('3. Ledger: component-by-component disposition', H2))
S.append(P(
    'The spine of the port. Anything marked %s must not be touched, or the PPDPP numbers '
    'stop being a baseline.' % chip('KEEP')))
S.append(Spacer(1, 2))
S.append(table([
    ['Component', 'Status', 'Note'],
    ['export_csa.py', chip('KEEP'),
     'Same seed, same 70/10/20 stratified scenario-disjoint split. Same %s scheme, same '
     'invariant checks, same dead-check audit.' % mono('uid')],
    ['views / _csa_render', chip('KEEP'),
     'View filtering becomes <i>more</i> load-bearing: %s is a new prompt surface that must '
     'also be filtered. See &#167;5b.' % mono('LLM_s')],
    ['_csa_reset / turn order', chip('KEEP'),
     '%s, %s, %s derivation all unchanged. One strategist action per chair slot, exactly as '
     'one planner action per chair slot today.'
     % (mono('utterance_cap'), mono('dm_slots'), mono('max_turn'))],
    ['_csa_note_disclosures', chip('KEEP'),
     'Threshold 0.35 stays frozen. Changing it would invalidate every disclosure figure '
     'already reported.'],
    ['_csa_note_leaks', chip('MODIFY'),
     'Currently called on advisor turns only. Must also run on chair turns under EPO. '
     'See &#167;5b.'],
    ['verifier.py', chip('KEEP'),
     '%s, canonicalisation, %s/%s, %s &#8212; verbatim. This is what makes the new process '
     'reward possible.' % (mono('score()'), mono('SafeDict'), mono('NormStr'),
                           mono('floor_score'))],
    ['_csa_terminal_reward', chip('KEEP'),
     "Becomes EPO's outcome reward at %s: pool/use/close mix, hallucination penalty, leak "
     'gate.' % mono('t = T')],
    ['_csa_shaping', chip('RETIRE'),
     'Off by default in the EPO arm &#8212; it double-counts disclosure against the new %s. '
     'Flags retained for ablation. See &#167;5c.' % mono('r_t')],
    ['CSAAct (4 keys)', chip('MODIFY'),
     'Demoted from action space to <i>act tag</i>. Still emitted, still parsed, no longer '
     'the whole action.'],
    ["CSAMessages('system')", chip('MODIFY'),
     '%s becomes %s. Settlement-schema injection on the settling act is unchanged.'
     % (mono('instr + CSAAct[action]'), mono('instr + act_instr(&#964;) + &#963;'))],
    ["CSAMessages('strategist')", chip('NEW'),
     'Prompt for %s, rendered through the chair&#8217;s view. Never the raw case.'
     % mono('LLM_s')],
    ['PPDPP (RoBERTa)', chip('REPLACE'),
     'Swap for %s: causal LM, LoRA, returns text plus per-token log-probs.'
     % mono('EPOStrategist')],
    ['optimize_model()', chip('REPLACE'),
     'Per-episode whitening replaced by EPO&#8217;s max-abs normalisation and '
     'token-averaged log-prob.'],
    ['online step reward', chip('REPLACE'),
     'Rewards now assigned <i>after</i> the episode from the recorded trace. The env still '
     'returns %s; its per-step reward is ignored in this arm.' % mono('done')],
    ['prm.py', chip('NEW'),
     'Two implementations behind one interface &#8212; %s (EPO-faithful) and %s '
     '(deterministic). See &#167;4.' % (mono('JudgePRM'), mono('VerifierPRM'))],
    ['self-play loop', chip('NEW'),
     'Scoped carefully &#8212; CSA is asymmetric and naive self-play collapses the '
     'benchmark. See &#167;6.'],
    ['make_sft_data.py', chip('MODIFY'),
     'Optional. Filtered rollouts already carry an act per turn; add a verbalisation pass '
     'to get strategy-SFT pairs.'],
    ['record dict / metrics', chip('KEEP'),
     '%s runs unchanged on EPO records. Add one PRM-agreement section.'
     % mono('compute_all_metrics.py')],
], [36 * mm, 16 * mm, 118 * mm], first_col_mono=True))

# ================================================================= 4
S.append(PageBreak())
S.append(Paragraph('4. Formulation', H2))

S.append(Paragraph('4.1 Action', H3))
S.append(P(
    'Keep the act tag, open up its content. The strategist emits %s where %s and %s is free '
    'text under roughly twenty words:'
    % (mono('&#964;: &#963;'),
       mono('&#964; &#8712; {ask, followup, share, decide}'),
       mono('&#963;'))))
S.append(codebox(
    'ask:      get Morgan to state the floor-load certification for the heavy pallet\n'
    'followup: Patel gave a clearance figure with no units — pin the exact metres\n'
    'decide:   record it now, citing the two lane constraints you were given'))
S.append(Spacer(1, 6))
S.append(P(
    "This is a deliberate departure from EPO's fully-open action, and it earns its keep "
    'three times over. Three pieces of CSA machinery read the act <i>symbolically</i>, and '
    'all three break under free text alone:'))
for t in [
    '%s raises the token budget to 512 and skips %s, which would otherwise truncate the '
    'settlement JSON at a sentence boundary (%s, %s).'
    % (mono('action == CSA_SETTLING_ACT'), mono('postprocess_response'),
       mono('env.py:254'), mono('env.py:265')),
    '%s drives %s, which populates the coverage potential (%s).'
    % (mono('action in CSA_ELICITING_ACTS'), mono('_csa_note_addressed'),
       mono('env.py:269')),
    'The same test sets %s &#8212; the elicited-versus-volunteered distinction that the '
    'whole hidden-profile framing rests on (%s).'
    % (mono('reveal_elicited[fid]'), mono('env.py:492')),
]:
    S.append(B(t))
S.append(Spacer(1, 3))
S.append(P(
    'Recovering those from free text would need a classifier, which reintroduces exactly '
    'the error source the port removed. The tag costs the strategist one word and keeps '
    'every existing metric exact. Parse %s with a prefix match; fall back to %s and log '
    'the miss.' % (mono('&#964;'), mono('ask'))))

S.append(Paragraph('4.2 Objective', H3))
_cmp = Table([[
    [Paragraph('PPDPP-CSA TODAY', ParagraphStyle(
        'CH', parent=CELL, fontName=SANSB, fontSize=7, textColor=MUTED, spaceAfter=5)),
     XPreformatted(
        'a_t ~ π_θ(· | h_{1:t-1}),  a_t ∈ A, |A| = 4\n'
        'R_t = Σ_{t′≥t} γ^{t′-t} r_{t′}\n'
        'Â   = (R − mean R) / (std R + ε)\n'
        'L   = − Σ_t  log π_θ(a_t) · Â_t', CODE),
     Paragraph('One log-prob per turn. Rewards arrive online from '
               + mono('env.step') + '.', NOTE)],
    [Paragraph('EPO-CSA', ParagraphStyle(
        'CH2', parent=CELL, fontName=SANSB, fontSize=7, textColor=MUTED, spaceAfter=5)),
     XPreformatted(
        'a_t = (τ_t, σ_t),  a_t = LLM_s(· | view_dm, h, a_{1:t-1})\n'
        'R_t = Σ_{t′≥t} γ^{t′-t} r_{t′}\n'
        'A_t = R_t / max_t |R_t|\n'
        'L   = − (1/T) Σ_t A_t · (1/|k_t|) Σ_i log π_θ(a_{t,i})',
        CODE),
     Paragraph('Token-averaged over the strategy&#8217;s ' + mono('k_t')
               + ' tokens. Rewards assigned after the episode.', NOTE)],
]], colWidths=[85 * mm, 85 * mm], hAlign='LEFT')
_cmp.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), CODEBG),
    ('BOX', (0, 0), (-1, -1), 0.5, RULE),
    ('INNERGRID', (0, 0), (-1, -1), 0.5, RULE),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 7),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
]))
S.append(_cmp)
S.append(Spacer(1, 7))
S.append(P(
    '&#947; stays at 0.999 or drops to EPO&#8217;s 0.99 &#8212; report whichever, but hold '
    'it fixed across arms. The normalisation change matters more than it looks: max-abs '
    'sets the largest-magnitude turn&#8217;s advantage to &#177;1 and leaves the rest '
    'proportional, where whitening centres them. With sparse binary process rewards, '
    'whitening would push negative advantage onto every non-critical turn; max-abs leaves '
    'them near zero. Keep EPO&#8217;s.'))

S.append(Paragraph('4.3 Process reward, computed rather than judged', H3))
S.append(P(
    'This is the part worth writing a paper about. EPO&#8217;s %s comes from GPT-4o reading '
    'the finished trajectory and naming the critical turns. It costs a call per episode, it '
    'drifts with the judge, and on SOTOPIA there is no way to check it. CSA ships %s, which '
    'states for every private fact exactly which checks it flips &#8212; so criticality is '
    'not a judgement call, it is a lookup.' % (mono('r_t'), mono('decisive_facts'))))
S.append(P(
    'The environment already records everything needed. %s maps each disclosed fact to the '
    'chair turn it surfaced on; %s says whether an eliciting act preceded it. Define:'
    % (mono('reveal_turn'), mono('reveal_elicited'))))
S.append(codebox(
    '<font color="#6e7883"># D = decisive fact ids;  F(f) = checks that fact f flips;  '
    'Φ = ⋃ F(f)</font>\n\n'
    'r_t^elicit = Σ_{f ∈ D}  1[reveal_turn[f] == t] · '
    '1[reveal_elicited[f]] · |F(f)| / |Φ|\n\n'
    'r_t^settle = 1[t == t_settle] · 1[schema_valid ∧ ¬ leaks] '
    '· dca\n\n'
    'r_t        = r_t^elicit + λ · r_t^settle', bar=KEEP))
S.append(Spacer(1, 3))
S.append(P(
    'Graded variant. The EPO-faithful binary form is %s iff %s or the settling turn clears '
    '%s.' % (mono('r_t = 1'), mono('r_t^elicit &gt; 0'), mono('dca &#8805; &#964;')), NOTE))
S.append(Spacer(1, 4))
S.append(P(
    'Every term is already in the trace. Zero extra model calls, byte-for-byte reproducible, '
    'and weighted by how much each fact actually moves the outcome &#8212; a fact flipping '
    'four checks credits its turn more than one flipping a single check. The graded form is '
    'strictly more informative than EPO&#8217;s binary label and costs nothing.'))
S.append(P(
    'Ship both PRMs. %s reproduces EPO faithfully; %s is the contribution. That mirrors the '
    'existing %s split exactly, so the result table is a clean 2&#215;2 of {PPDPP, EPO} '
    '&#215; {judged, verified} &#8212; and you can report Cohen&#8217;s &#954; between the '
    'two PRMs on identical trajectories, which is a validation of LLM-as-process-judge that '
    'no existing EPO environment supports.'
    % (mono('JudgePRM'), mono('VerifierPRM'), mono('--csa_reward critic|verifier'))))

# ================================================================= 5
S.append(PageBreak())
S.append(Paragraph('5. Three things that break, and the fixes', H2))

S.append(Paragraph('5a. %s &nbsp; The trainer no longer sees rewards online'
                   % chip('MODIFY'), H3))
S.append(P(
    'PPDPP&#8217;s loop appends a reward inside the %s body and calls %s at episode end. '
    'EPO&#8217;s process rewards are only knowable once the trajectory is complete, so the '
    'loop becomes: run the episode ignoring %s reward, collect the trace, call the PRM, '
    'compute returns, take one gradient step. The env needs no change beyond an %s accessor '
    '&#8212; %s, %s, %s, %s and %s are all already populated. %s still comes from %s so '
    'termination and SR@t are unchanged.'
    % (mono('for t in count()'), mono('optimize_model()'), mono('env.step'),
       mono('episode_trace()'), mono('reveal_turn'), mono('reveal_elicited'),
       mono('act_history'), mono('last_score'), mono('leaks'), mono('done'),
       mono('_csa_step_verifier'))))

S.append(Paragraph('5b. %s &nbsp; Chair leaks are not detected' % chip('REPLACE'), H3))
S.append(callout('REQUIRED CHANGE', [
    '%s is called only on advisor turns (%s); the chair branch breaks out of the loop '
    'before reaching it. Under PPDPP that was harmless &#8212; a four-way classifier has no '
    'channel through which to smuggle a fact. Under EPO, %s writes free text directly into '
    'the chair&#8217;s prompt and is optimised against a reward that rises with disclosure. '
    'If it ever sees the full case, <i>&#8220;tell them Lane 2 has the floor-load '
    'certification&#8221;</i> is a one-step reward hack that scores as a successful '
    'elicitation.'
    % (mono('_csa_note_leaks'), mono('env.py:279'), mono('LLM_s')),

    'Two changes, both small. First, %s must render through %s &#8212; the chair&#8217;s '
    'view &#8212; and must never touch %s. Second, call %s on chair turns too. %s '
    '%s already proves no private fact is in the chair&#8217;s view, so the existing '
    'detector is correct for the chair with no modification.'
    % (mono("CSAMessages(case, 'strategist', &#8230;)"), mono('_csa_render(case, dm)'),
       mono('CSA_ORACLE_ONLY'), mono('_csa_note_leaks(self.dm, resp)'),
       mono('export_csa.py'), mono('check_invariants')),
]))
S.append(Spacer(1, 6))
S.append(P(
    'With both in place the leak gate in %s does real work: a strategist that finds the '
    'shortcut gets &#8722;1.0 and the episode is invalidated, rather than being quietly '
    'rewarded.' % mono('_csa_terminal_reward')))

S.append(Paragraph('5c. %s &nbsp; Shaping double-counts the process reward'
                   % chip('RETIRE'), H3))
S.append(P(
    '%s is potential-based over disclosure, elicitation and coverage, and its correctness '
    'argument is Ng et al. (1999): &#947;&#934;(s&#8242;) &#8722; &#934;(s) cannot change '
    'the optimal policy. That argument assumes &#934; is a function of state alone, layered '
    'on a fixed reward stream. The verifier PRM&#8217;s %s <i>is</i> elicited disclosure of '
    'decisive facts &#8212; the same event the disclosure potential measures. Stacking them '
    'counts it twice and the invariance guarantee no longer applies.'
    % (mono('_csa_shaping'), mono('r_t^elicit'))))
S.append(P(
    'Default the EPO arm to %s. Keep the flags, run shaping-on as an ablation, and say in '
    'the writeup why it is off rather than leaving it at the PPDPP default.'
    % mono('--csa_w_shape_disc 0 --csa_w_shape_elic 0 --csa_w_shape_cover 0')))

# ================================================================= 6
S.append(PageBreak())
S.append(Paragraph('6. Self-play: an asymmetric game with a collapse mode', H2))
S.append(P(
    'EPO&#8217;s self-play has two symmetric instances alternating as partners. CSA is not '
    'symmetric: one chair holds no private facts and makes the decision, N advisors each '
    'hold one and do not. There is no seat-swap that leaves the game the same.'))
S.append(P(
    'The naive extension &#8212; give advisors their own trainable strategist on the same '
    'reward &#8212; <b>collapses the benchmark</b>. An advisor optimised for global %s '
    'learns to state PF1 in its first utterance. %s goes to 1, the profile stops being '
    'hidden, %s goes to 0 because nothing was elicited, and the chair needs no skill at all. '
    'The measured improvement would be real and completely uninformative.'
    % (mono('dca'), mono('disclosure_rate'), mono('reveal_elicited'))))
S.append(Spacer(1, 2))
S.append(table([
    ['Option', 'What it is', 'Verdict'],
    ['chair-only', 'Advisors stay frozen environment, as today',
     'Use for the headline number &#8212; directly comparable to PPDPP-CSA. Report it '
     'honestly as <i>EPO without self-play</i>, which is one of EPO&#8217;s own ablation '
     'rows.'],
    ['seat-rotation',
     'Role-permuted scenario variants so one %s trains in every chair seat; advisor policy '
     'still frozen' % mono('LLM_s'),
     'The closest faithful analogue of EPO self-play. Needs a permutation generator that '
     're-derives %s and %s while preserving %s.'
     % (mono('views'), mono('decision_maker'), mono('decisive_facts'))],
    ['two-sided',
     'Advisors get strategists with an <i>opposing</i> objective &#8212; rewarded on their '
     'own section&#8217;s constraints being honoured, not on global %s' % mono('dca'),
     'Genuinely interesting, and the only version where &#8220;when to volunteer&#8221; is '
     'learned rather than assumed. Scope it as separate work; it changes what the benchmark '
     'measures.'],
], [26 * mm, 63 * mm, 81 * mm], first_col_mono=True))
S.append(Spacer(1, 6))
S.append(P(
    'Recommendation: chair-only for the main table, seat-rotation as the self-play arm, '
    'two-sided as future work &#8212; and be explicit in the writeup about which EPO '
    'components each arm actually includes.'))

# ================================================================= 7
S.append(Paragraph('7. Risk: scale, and what to do about it', H2))
S.append(P(
    'EPO trains on SOTOPIA with roughly 2,050 training episodes and ~19k steps, batch 32, '
    'full fine-tuning of Llama3-8B. CSA has <b>99 training scenarios</b>. Rollouts can '
    'inflate the episode count, but scenario diversity is the binding constraint on a policy '
    'that must generalise to 42 unseen scenarios &#8212; and the SFT split already shows how '
    'thin the labelled layer is at 650 turns.'))
for t in [
    '<b>LoRA on %s, not full fine-tuning.</b> At 99 scenarios a full 8B update will '
    'memorise. Rank 16&#8211;32 on attention projections is the right starting point.'
    % mono('LLM_s'),
    '<b>Small strategist, unchanged agent.</b> A 3B&#8211;7B %s, with %s pinned to %s '
    '&#8212; the same backend the PPDPP-CSA runs used, so the dialogue agent is held fixed '
    'across the whole comparison.'
    % (mono('LLM_s'), mono('LLM_d'), mono('Qwen2.5-7B-Instruct')),
    '<b>Lean on reward density.</b> The verifier PRM gives signal on most turns for free, '
    'where SOTOPIA&#8217;s terminal goal score gives one number per episode. Sample '
    'efficiency should be substantially better; that is a claim worth measuring, not '
    'assuming.',
    '<b>Memory.</b> Two resident models &#8212; frozen 7B env plus trainable strategist '
    '&#8212; where the current runs already fill a card with one. Either two GPUs, or serve '
    '%s behind vLLM and keep only %s in the training process.'
    % (mono('LLM_d'), mono('LLM_s')),
]:
    S.append(B(t))
S.append(Spacer(1, 3))
S.append(P(
    'If the pure-RL configuration will not move at this scale, the SFT warm-start is already '
    'built: %s surviving rollouts carry an act per chair turn, so a verbalisation pass turns '
    'them into strategy-SFT pairs at no annotation cost. EPO reports pure RL beating SFT+RL '
    '&#8212; at 99 scenarios that finding is not obviously portable, and testing it is a '
    'legitimate result either way.' % mono('make_sft_data.py')))

# ================================================================= 8
S.append(PageBreak())
S.append(Paragraph('8. Protocol: what must be held fixed', H2))
S.append(P(
    'The value of this port is that a PPDPP number already exists on the same data. Five '
    'things have to be identical or the comparison is worthless:'))
for i, t in enumerate([
    '<b>Splits.</b> %s, 99/9/42, unregenerated.' % mono('export_csa.py --seed 0'),
    '<b>Dialogue agent and sampling.</b> Same %s / %s backend; %s already forces greedy at '
    'test, keep that.' % (mono('--system'), mono('--user'), mono('generate_response')),
    '<b>Record schema.</b> The %s record dict unchanged, so %s runs on both without a '
    'branch.' % (mono('evaluate()'), mono('compute_all_metrics.py')),
    '<b>Call budget.</b> %s, %s and %s are logged for exactly this reason &#8212; the '
    'comment in %s says a planner that wins by making more calls has not won. EPO adds one '
    '%s call per chair turn, plus one PRM call per episode in the judged arm. Report SR at '
    'matched budget alongside raw SR.'
    % (mono('n_calls'), mono('calls_by_role'), mono('prompt_chars'),
       mono('generate_response'), mono('LLM_s')),
    '<b>Seeds.</b> Three at minimum. The existing %s shows why.' % mono('logs_variance.txt'),
]):
    S.append(Paragraph(t, BULLET, bulletText='%d.' % (i + 1)))

S.append(Paragraph('New measurements worth taking', H3))
for t in [
    '<b>PRM agreement.</b> Cohen&#8217;s &#954; between %s and %s over the same '
    'trajectories. Direct evidence on whether LLM process judging is trustworthy &#8212; '
    'unavailable in any environment EPO currently uses.'
    % (mono('JudgePRM'), mono('VerifierPRM')),
    '<b>Strategy adherence.</b> EPO says %s &#8220;selectively adopts&#8221; the strategy. '
    'On CSA you can check: did an %s-tagged turn actually address a named advisor? %s '
    'already answers it.'
    % (mono('LLM_d'), mono('ask'), mono('_csa_note_addressed')),
    '<b>Open-vocabulary gain.</b> Ablate %s to empty, leaving only the act tag. That reduces '
    'EPO exactly to PPDPP&#8217;s action space with an LLM policy, and isolates what the '
    'open action space is actually buying &#8212; the central claim of the paper, tested '
    'deterministically.' % mono('&#963;'),
    '<b>Elicited versus volunteered.</b> Already logged. The cleanest available measure of '
    'whether the strategist learned to draw information out rather than wait for it.',
]:
    S.append(B(t))

# ================================================================= 9
S.append(Paragraph('9. Work plan: files to touch', H2))
S.append(table([
    ['File', 'Change', 'Detail'],
    ['prompt.py', chip('MODIFY'),
     'Add %s; accept a strategy string in %s; add the %s role branch rendered through the '
     'chair&#8217;s view; add %s.'
     % (mono('_csa_act_instr(&#964;)'), mono("CSAMessages('system')"),
        mono("'strategist'"), mono('CSA_STRATEGY_MAX_WORDS'))],
    ['env.py', chip('MODIFY'),
     '%s / %s parse &#964; from the action; leak check on chair turns; %s returning outcome '
     'only at terminal; %s.'
     % (mono('_csa_is_settling'), mono('_csa_is_eliciting'), mono("csa_reward='epo'"),
        mono('episode_trace()'))],
    ['epo_agent.py', chip('NEW'),
     '%s; %s with max-abs advantage and token-averaged log-prob; LoRA save/load mirroring '
     '%s.' % (mono('EPOStrategist.select_action(state) &#8594; (text, logprobs)'),
              mono('optimize_model()'), mono('PPDPP.save_model'))],
    ['prm.py', chip('NEW'),
     '%s and %s, one interface: %s. Graded and binary modes on the former.'
     % (mono('VerifierPRM'), mono('JudgePRM'),
        mono('trace &#8594; [r_1 &#8230; r_T]'))],
    ['run_epo.py', chip('NEW'),
     'Episode &#8594; trace &#8594; PRM &#8594; returns &#8594; gradient step; self-play '
     'iteration wrapper; %s reused as-is.' % mono('evaluate()')],
    ['make_sft_data.py', chip('MODIFY'),
     'Optional verbalisation pass over surviving rollouts to emit strategy-SFT pairs.'],
    ['compute_all_metrics.py', chip('KEEP'),
     'Add a PRM-agreement section; everything else runs unchanged.'],
    ['verifier.py', chip('KEEP'), 'Untouched.'],
    ['export_csa.py', chip('KEEP'), 'Untouched.'],
], [34 * mm, 16 * mm, 120 * mm], first_col_mono=True))
S.append(Spacer(1, 6))
S.append(P(
    'Order of work: prompt and env changes first (they are small and testable against '
    'existing records via %s), then %s with %s validated offline on the recorded test '
    'trajectories you already have, then the strategist and the loop. The PRM can be built '
    'and checked before a single training step runs, because the traces exist.'
    % (mono('smoke_csa.py'), mono('prm.py'), mono('VerifierPRM'))))

S.append(Spacer(1, 10))
S.append(Paragraph(
    'References &#8212; EPO: Liu, Wang, Li, Wu, Ma, Kong, Huang, Jiao &amp; Zhang, '
    '<i>EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement '
    'Learning</i>, ACL 2025 (arXiv:2502.12486). Shaping invariance: Ng, Harada &amp; '
    'Russell, ICML 1999. Logit adjustment: Menon et al., ICLR 2021. '
    'Line references are to <font face="%s">ppdpp_csa/</font> as of this working tree.'
    % MONO, NOTE))


def _footer(canv, doc):
    canv.saveState()
    canv.setFont(MONO, 7)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 10 * mm, 'EPO on CSA — design spec')
    canv.drawRightString(A4[0] - 20 * mm, 10 * mm, str(canv.getPageNumber()))
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.4)
    canv.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
    canv.restoreState()


SimpleDocTemplate(OUT, pagesize=A4,
                  leftMargin=20 * mm, rightMargin=20 * mm,
                  topMargin=18 * mm, bottomMargin=18 * mm,
                  title=TITLE, author='').build(S, onFirstPage=_footer,
                                                onLaterPages=_footer)
print('wrote', OUT)
