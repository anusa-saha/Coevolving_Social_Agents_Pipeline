"""Build the PPDPP-CSA architecture + results PDF."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = 'ppdpp-csa-report.pdf'
INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5b5b5b')
RULE = colors.HexColor('#d0d0d0')
HEAD = colors.HexColor('#22333b')
BAND = colors.HexColor('#f2f4f5')

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontName='Helvetica-Bold',
                    fontSize=17, leading=21, textColor=INK, alignment=TA_LEFT,
                    spaceAfter=2)
SUB = ParagraphStyle('SUB', parent=ss['Normal'], fontName='Helvetica',
                     fontSize=9, leading=13, textColor=MUTED, spaceAfter=12)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontName='Helvetica-Bold',
                    fontSize=12, leading=15, textColor=HEAD,
                    spaceBefore=14, spaceAfter=5)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontName='Helvetica-Bold',
                    fontSize=9.5, leading=12, textColor=INK,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle('BODY', parent=ss['Normal'], fontName='Helvetica',
                      fontSize=9, leading=12.5, textColor=INK, spaceAfter=6)
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8, leading=11, textColor=MUTED)
CELL = ParagraphStyle('CELL', parent=ss['Normal'], fontName='Helvetica',
                      fontSize=7.6, leading=9.6, textColor=INK)
CELLB = ParagraphStyle('CELLB', parent=CELL, fontName='Helvetica-Bold')
CELLH = ParagraphStyle('CELLH', parent=CELL, fontName='Helvetica-Bold',
                       textColor=colors.white)


def P(t, s=CELL):
    return Paragraph(t, s)


def table(rows, widths, band_body=True):
    data = [[P(c, CELLH) for c in rows[0]]]
    data += [[P(c) for c in r] for r in rows[1:]]
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


S = []
S.append(Paragraph('PPDPP-CSA: Architecture Changes and Results', H1))
S.append(Paragraph(
    'What was changed from the published PPDPP baseline, what the run produced, '
    'and what could stop someone reproducing it.', SUB))

# ----------------------------------------------------------------- SECTION 1
S.append(Paragraph('1. Architecture: core PPDPP vs. what we changed', H2))
S.append(Paragraph(
    'Upstream is <font face="Courier">github.com/dengyang17/PPDPP</font>. It is kept '
    'unmodified in <font face="Courier">upstream/</font>. Everything below lives in '
    '<font face="Courier">ppdpp_csa/</font>.', BODY))

S.append(table([
    ['Core PPDPP architecture', 'Changes Made'],
    ['Task is a two-party dialogue: ESConv (emotional support), CIMA (tutoring), '
     'CraigslistBargain (haggling).',
     'New <b>csa</b> dataset. A meeting of 3-5 agents: one chair plus advisors who must '
     'agree a decision.'],
    ['All information is shared. Both speakers see the same thing.',
     'Each advisor holds <b>private facts</b>. A <b>views</b> field controls who sees what. '
     'The chair must ask to find out.'],
    ['Reward comes from an LLM judge reading the transcript.',
     'New <b>verifier</b> arm: reward comes from Python expressions shipped with each case. '
     'Same input always gives the same score; no judge needed. The old judge arm still works.'],
    ['Reward value is an ordinal ladder sampled 10 times and averaged '
     '(worse -1.0, same -0.5, better 0.5, solved 1.0), giving a smooth number.',
     'Weighted sum of check pass-fractions: <font face="Courier">0.5*cbar + 0.5*pbar</font>. '
     'No sampling, no partial-credit rungs.'],
    ['Success is a threshold on that one number '
     '(<font face="Courier">reward &gt; 0.5</font>, <font face="Courier">== 1</font>, or '
     '<font face="Courier">&gt;= 0</font>, depending on dataset).',
     '<font face="Courier">all_content AND all_prov</font> - every check in the case must '
     'pass at the same time (5 to 11 of them).'],
    ['No reward shaping.',
     'Potential-based shaping added over three signals: disclosure, elicitation, coverage.'],
    ['Model replies are free text.',
     'The chair must emit a <b>JSON settlement</b>: decisions, commitments, credited_facts, '
     'justification_fact_ids. Gets a larger token budget so the JSON is not cut off.'],
    ['No notion of who revealed what.',
     '<b>Disclosure detector</b>: word-overlap above a threshold marks a private fact as '
     'revealed, and records whether a question prompted it.'],
    ['No integrity checking.',
     '<b>Leak detector</b>: flags an agent stating a fact it was never shown.'],
    ['Backends: vicuna, llama2, chatgpt.',
     '<b>qwen</b> added (Qwen2.5-7B-Instruct) and used for every agent in our runs.'],
    ['Two fixed roles take turns.',
     'Per-case <font face="Courier">turn_order</font> cursor and <font face="Courier">turn_cap</font>. '
     'Episode length is the number of chair slots inside the cap.'],
    ['Output paths built from backbone and sft_dir only.',
     'Seed and reward mode added to the path, so two seeds or the two arms cannot '
     'overwrite each other.'],
    ['Every checkpoint is kept.',
     '<font face="Courier">--keep_ckpt</font> prunes old ones. Each is 1.42 GB.'],
    ['No cost accounting.',
     'Counts model calls and prompt characters per episode, so a planner cannot "win" by '
     'simply making more calls.'],
    ['Reply trimmed at the other role\'s bare name.',
     'Trimmed at "Name:" instead, so "Dr. Chen recommends X" is not cut at the name.'],
    ['Data loaded from <font face="Courier">../data</font>.',
     'Falls back to <font face="Courier">./data</font>, where the splits actually live.'],
], [78 * mm, 92 * mm]))

S.append(Paragraph(
    'Size of the change: env.py 316&#8594;748 lines, prompt.py 128&#8594;291, run.py 213&#8594;334, '
    'sft.py 315&#8594;358, agent.py 112&#8594;131, data_reader.py 95&#8594;118, utils.py 74&#8594;80. '
    'Plus verifier.py and 14 other scripts that do not exist upstream.', NOTE))

# ----------------------------------------------------------------- SECTION 2
S.append(PageBreak())
S.append(Paragraph('2. Results', H2))
S.append(Paragraph(
    'The run completed <b>7 training steps = 700 episodes</b>, then the process was killed '
    '69 episodes into step 8. Evaluation runs every 2 steps, so the snapshots are at '
    '<b>0, 200, 400 and 600 episodes</b>. There is no evaluation at 700. Every number below '
    'is under the reward configuration described in section 1. '
    'Test set is 42 meetings, one seed.', BODY))

S.append(Paragraph('2a. Training side (all 700 episodes)', H3))
S.append(table([
    ['Step', 'Episodes', 'Loss', 'Success', 'Avg turns', 'Reward'],
    ['1', '100', '0.651', '0.0', '4.17', '-0.0974'],
    ['2', '200', '1.019', '0.0', '4.21', '-0.1174'],
    ['3', '300', '0.786', '0.0', '4.17', '-0.1433'],
    ['4', '400', '0.762', '0.0', '4.21', '-0.1301'],
    ['5', '500', '0.807', '0.0', '4.17', '-0.1465'],
    ['6', '600', '1.296', '0.0', '4.21', '-0.1287'],
    ['7', '700', '0.776', '0.0', '4.17', '-0.1406'],
], [16 * mm, 22 * mm, 22 * mm, 22 * mm, 24 * mm, 24 * mm]))
S.append(Paragraph(
    'Reward drifts downward and settles near -0.14. Loss moves without direction. '
    'This is what a missing gradient looks like.', NOTE))

S.append(Paragraph('2b. Evaluation fields', H3))
S.append(table([
    ['Field', 'Value (0 &#8594; 600 ep)', 'What it means'],
    ['Joint Success Rate', '0.0000 &#8594; 0.0000',
     'Share of meetings where <i>every</i> check passed. Cannot be reached - see risk 1.'],
    ['DCA', '0.1973 &#8594; 0.2007',
     '<b>Main measure.</b> Of the checks that a hidden fact changes, the share passed. '
     'Tests whether pooling information actually helped.'],
    ['NDG', '0.1973 &#8594; 0.2007',
     'DCA compared against a no-conversation baseline. Equal to DCA here because that '
     'baseline scores 0.'],
    ['Decisive Credit Rate', '0.0238 &#8594; 0.0198',
     'How often the chair correctly credits a hidden fact it relied on.'],
    ['C_micro / C_macro', '0.2366 / 0.2451 &#8594; 0.2366 / 0.2445',
     'Share of content checks passed, averaged per check / per meeting. '
     '"Content" = was the decision right.'],
    ['P_micro / P_macro', '0.0400 / 0.0476 (flat)',
     'Same for provenance. "Provenance" = did the chair cite the facts it used.'],
    ['Provenance Success Rate', '0.0476 (flat)',
     'Share of meetings passing all provenance checks.'],
    ['Non-decisive accuracy', '0.3333 &#8594; 0.2778',
     'Pass rate on checks that do not depend on any hidden fact.'],
    ['Commitment coverage', '0.8571 &#8594; 0.9286',
     'Share of meetings that produced any follow-up commitments at all.'],
    ['Commitment wellformedness', '0.0000 (flat)',
     'Share with the exact commitment wording the checks demand. <b>Never once</b> - see risk 2.'],
    ['Decisive Disclosure Rate', '0.0774 &#8594; 0.0893',
     'Share of hidden facts that were actually revealed in conversation.'],
    ['Silent holder rate', '0.9226 &#8594; 0.9107',
     'Share of advisors holding a private fact who never shared it. <b>The real bottleneck.</b>'],
    ['Disclosure precision / recall', '1.0000 / 0.0774 &#8594; 1.0000 / 0.0893',
     'Precision is 1.0 by construction (the detector defines disclosure), so only recall is '
     'informative.'],
    ['Disclosure F1', '0.1436 &#8594; 0.1639',
     'Combined disclosure score.'],
    ['Elicited fraction', '1.0000 &#8594; 0.7778',
     'Of facts revealed, the share that followed a question rather than being volunteered.'],
    ['Pooling completeness', '0.0000 (flat)',
     'Meetings where every hidden fact came out. Never happened.'],
    ['Hallucinated credit rate', '0.5476 &#8594; 0.6667',
     'Credits given to facts never disclosed. <b>Got worse with training</b> - nothing '
     'penalised it.'],
    ['Citation containment', '0.4524 &#8594; 0.3333',
     'Share of cited IDs that actually exist. <b>Also got worse.</b>'],
    ['Avg turns / cap, Timeout rate', '1.0000 / 1.0000 (flat)',
     'Every meeting ran to its turn limit. So average turns is fixed by the cap and tells '
     'you nothing about the policy.'],
    ['JSON parse rate', '1.0000 (flat)',
     'The settlement was always valid JSON. Formatting is not the problem.'],
    ['Act entropy (bits)', '0.6488 &#8594; 1.3953 (dips to 0.2954)',
     'Variety in the planner\'s choices. It collapsed onto "ask" by 400 episodes, then '
     'recovered. <b>Proof the policy did change.</b>'],
    ['Return mean / variance', '-0.0943 / 0.0685 &#8594; -0.0946 / 0.0620',
     'Average reward per episode. Flat - the old reward carried no signal.'],
    ['Calls per episode', '14.19 &#8594; 13.93',
     'Model calls used per meeting. Cost measure.'],
    ['Prompt tokens per episode', '~9,750 &#8594; ~9,665',
     'Prompt size per meeting. Cost measure.'],
    ['vs floor (win / tie / p)', '0 / 42 / 1.0',
     'Compared against no conversation at all. Uninformative here, because it is paired on '
     'joint success, which is 0 on both sides.'],
], [38 * mm, 40 * mm, 92 * mm]))

S.append(Paragraph('2c. What the numbers say', H3))
S.append(Paragraph(
    '<b>The policy did change, but not usefully.</b> Action entropy moves a lot '
    '(0.65 &#8594; 0.30 &#8594; 1.40), so the planner was learning something. But average '
    'reward stayed flat and joint success was pinned at zero throughout.', BODY))
S.append(Paragraph(
    '<b>Two things got steadily worse.</b> Hallucinated credit rose from 0.55 to 0.67 and '
    'citation containment fell from 0.45 to 0.33. The policy was learning to claim facts it '
    'never obtained, because nothing in the reward pushed back against it. A subtractive '
    'penalty on undisclosed credit is needed to stop this.', BODY))
S.append(Paragraph(
    '<b>Pooling barely happened.</b> About 91% of advisors holding a private fact never '
    'shared it. That, not the decision logic, is where the task is being lost.', BODY))
S.append(Paragraph(
    '<b>One mild positive.</b> DCA at 600 episodes (0.2007) is the highest of the four '
    'snapshots, alongside better disclosure. With one seed and 42 cases this is well inside '
    'noise, but it is the only sign that the metric the new reward trains on can move.', BODY))

# ----------------------------------------------------------------- SECTION 3
S.append(PageBreak())
S.append(Paragraph('3. Risks to reproducing this', H2))
S.append(table([
    ['#', 'Risk', 'Effect and what to do'],
    ['1', 'Five checks can never pass: C7, C8, C9, P2, P3 score 0% across all 42 cases.',
     'Because success needs <i>all</i> checks, joint success is 0 by arithmetic, whatever '
     'the model does. Do not report joint success as a capability measure. Use DCA.'],
    ['2', 'Commitment wording is effectively random: 143 distinct types across 150 cases, '
     '100 of them used once. Only <b>15%</b> of test wordings ever appear in training.',
     'No model can guess them; no prompt or normalisation fixes it (canonicalising merges '
     'only 6 of 143). Any metric needing exact commitment matching will read near zero for '
     'reasons unrelated to the model.'],
    ['3', 'No gold settlement is stored in any case.',
     'There is no way to confirm a check is satisfiable before training. This is why the '
     'five dead checks shipped unnoticed. Adding one worked answer per case would make '
     'every future check self-verifying.'],
    ['4', 'Under the old reward, disclosure was 0 in 91.3% of steps and shaping cancels out '
     'over an episode, so reward was the constant -0.1.',
     'No gradient, hence the flat curves above. Any rerun on these settings will reproduce '
     'the flat result exactly, however long it is left training.'],
    ['5', 'Checkpoints are 1.42 GB and run.py <b>saves before pruning</b>.',
     'The first 1000-episode run died with a full disk mid-write, leaving a corrupt file. '
     'Keep at least 3 GB free, or reorder to prune first.'],
], [8 * mm, 68 * mm, 94 * mm]))

S.append(Spacer(1, 8))
S.append(Paragraph(
    'Prepared from the run log (rl-armB-1000ep-seed1.log), the four evaluation records at '
    'epochs 0/2/4/6, and the dataset splits. All figures recomputed from source rather than '
    'copied from earlier summaries.', NOTE))

SimpleDocTemplate(OUT, pagesize=A4,
                  leftMargin=20 * mm, rightMargin=20 * mm,
                  topMargin=18 * mm, bottomMargin=16 * mm,
                  title='PPDPP-CSA: Architecture Changes and Results',
                  author='').build(S)
print('wrote', OUT)
