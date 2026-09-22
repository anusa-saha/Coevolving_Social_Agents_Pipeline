"""Conversation and rollout logs, written the same way by every arm.

    EvalLog      one evaluation pass      -> <arm>/logs/eval/<run>/
    RolloutLog   training-time rollouts   -> <arm>/logs/rollouts/<run>/

An eval directory holds
    conversations.log   every episode as a readable transcript, headline metrics on top
    episodes.jsonl      the full record per episode, plus its `headline` metrics
    scenarios.csv       one row of headline metrics per scenario
    turns.csv           one row per utterance
    summary.log/.json   eval.py's SUMMARY block over the run

A rollout directory holds conversations.log, rollouts.jsonl, rollouts.csv and the summary.
A rollout is any episode generated in order to train on it -- a GRPO candidate, a REINFORCE
episode, a self-play corpus rollout. Each is tagged with the algorithm that produced it and
whatever that algorithm knows about it (step, group, candidate, reward, advantage), so a
new trainer needs nothing more than

    log = RolloutLog(LOGS, run, algo='gdpo')
    log.add(case, record, step=s, group=g, candidate=i, reward=r, advantage=a)
    ...
    log.close()

Rollout logs APPEND, so a resumed run keeps what came before; the summary covers the
episodes written by the current process only. Eval logs overwrite, like the Record files
beside them. Every write is flushed, so a run killed mid-way still leaves readable logs.
"""
import csv
import json
import os
import re

from csa_core import headline as H

CSV_MAX_CHARS = 4000                   # eval.py's cap on one CSV cell
ROLLOUT_COLS = ('algo', 'step', 'group', 'candidate', 'reward', 'advantage')
TURN_COLS = ('uid', 'turn', 'speaker', 'role', 'content')


def _safe(name):
    return re.sub(r'[^A-Za-z0-9._+=-]+', '_', str(name)).strip('_') or 'run'


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, (dict, list, tuple)):
        v = json.dumps(v, ensure_ascii=False, default=_default)
    s = str(v)
    return s if len(s) <= CSV_MAX_CHARS else s[:CSV_MAX_CHARS] + '...[clipped]'


def _fmt(v, spec):
    return '-' if v is None else spec % v


def conversation_text(case, rec, m, title, meta=None):
    """One episode as a readable block: the headline metrics, then the transcript."""
    settled = ''
    if m['settled']:
        settled = ' by %s' % m['settled_by']
        if m['turns_to_settle']:
            settled += ' at chair turn %s' % m['turns_to_settle']
    lines = [
        '=' * 96,
        title,
        'uid=%s  domain=%s  agents=%s' % (m['uid'], m['domain'], m['num_agents']),
        'checks %s/%s (%s)  content %s/%s  prov %s/%s  success=%s' % (
            _fmt(m['checks_passed'], '%d'), m['checks_total'],
            _fmt(m['checks_frac'], '%.2f'),
            _fmt(m['content_passed'], '%d'), m['content_total'],
            _fmt(m['prov_passed'], '%d'), m['prov_total'], _fmt(m['success'], '%d')),
        'reveals=%d  decisive %d/%d  settled=%d%s  turns=%s/%s  leaks=%d  dca=%s' % (
            m['reveals'], m['decisive_revealed'], m['decisive_total'], m['settled'],
            settled, m['turns_used'], m['max_turn'], m['leaks'], _fmt(m['dca'], '%.3f')),
    ]
    if meta:
        lines.append('rollout: ' + '  '.join('%s=%s' % (k, _cell(v))
                                             for k, v in meta.items()))
    lines.append('-' * 96)
    for t in rec.get('dialog') or ():
        if not isinstance(t, dict):
            continue
        mark = '[CHAIR] ' if H.speaker_of(case, t) == 'sys' else ''
        lines.append('%s%s: %s' % (mark, t.get('role'), t.get('content')))
    lines.append('-' * 96)
    lines.append('settlement: %s' % json.dumps(rec.get('settlement'), ensure_ascii=False,
                                               default=_default))
    elicited = [f for f, v in sorted((rec.get('reveal_elicited') or {}).items()) if v]
    lines.append('revealed: %s   elicited: %s'
                 % (', '.join(str(f) for f in rec.get('revealed') or ()) or '-',
                    ', '.join(elicited) or '-'))
    if rec.get('leaks'):
        lines.append('LEAKS: %s' % json.dumps(rec['leaks'], ensure_ascii=False,
                                              default=_default))
    return '\n'.join(lines) + '\n\n'


class _Log(object):
    KIND = RECORDS = ROWS = None
    COLS = H.METRIC_COLS

    def __init__(self, root, run, append):
        self.run = str(run)
        self.dir = os.path.join(root, self.KIND, _safe(run))
        os.makedirs(self.dir, exist_ok=True)
        self._mode = 'a' if append else 'w'
        self._files = []
        self._rec = self._open(self.RECORDS)
        self._conv = self._open('conversations.log')
        self._rows = self._csv(self.ROWS, self.COLS)
        self.rows = []
        self.summary = None

    def _open(self, name):
        f = open(os.path.join(self.dir, name), self._mode, encoding='utf-8')
        self._files.append(f)
        return f

    def _csv(self, name, cols):
        path = os.path.join(self.dir, name)
        fresh = (self._mode == 'w' or not os.path.isfile(path)
                 or not os.path.getsize(path))
        f = open(path, self._mode, newline='', encoding='utf-8')
        self._files.append(f)
        w = csv.DictWriter(f, fieldnames=list(cols), quoting=csv.QUOTE_ALL,
                           extrasaction='ignore')
        if fresh:
            w.writeheader()
        return w

    def _add(self, case, rec, key, stored, row_extra, title, meta=None):
        m = H.episode_metrics(case, rec)
        out = dict(rec)
        out[key] = stored
        out['headline'] = m
        self._rec.write(json.dumps(out, ensure_ascii=False, default=_default) + '\n')
        row = dict(m)
        row.update(row_extra)
        self._rows.writerow({k: _cell(row.get(k)) for k in self._rows.fieldnames})
        self._conv.write(conversation_text(case, rec, m, title, meta))
        self.rows.append(m)
        return m

    def flush(self):
        for f in self._files:
            f.flush()

    def _summary_extra(self):
        return {}

    def close(self, log=print):
        """Write eval.py's SUMMARY block for this run, close the files, return the summary.
        Pass log=None to write it without printing."""
        if self.summary is not None:
            return self.summary
        lines = H.summary_lines(self.run, self.rows)
        self.summary = H.summary(self.rows)
        with open(os.path.join(self.dir, 'summary.log'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        with open(os.path.join(self.dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(self.summary, run=self.run, **self._summary_extra()), f,
                      indent=1, default=_default)
        for f in self._files:
            f.close()
        if log:
            for line in lines:
                log(line)
            log('  logs -> %s' % self.dir)
        return self.summary


class EvalLog(_Log):
    """Every episode of one evaluation pass."""
    KIND, RECORDS, ROWS = 'eval', 'episodes.jsonl', 'scenarios.csv'
    COLS = ('tag',) + H.METRIC_COLS

    def __init__(self, root, run, tag=None):
        _Log.__init__(self, root, run, append=False)
        self.tag = tag or self.run
        self._turns = self._csv('turns.csv', TURN_COLS)

    def add(self, case, rec):
        m = self._add(case, rec, 'eval', {'tag': self.tag}, {'tag': self.tag},
                      '[eval %s | episode %d]' % (self.tag, len(self.rows) + 1))
        for i, t in enumerate(rec.get('dialog') or ()):
            if isinstance(t, dict):
                self._turns.writerow({'uid': _cell(m['uid']), 'turn': i,
                                      'speaker': H.speaker_of(case, t),
                                      'role': _cell(t.get('role')),
                                      'content': _cell(t.get('content'))})
        self.flush()
        return m

    def _summary_extra(self):
        return {'tag': self.tag, 'scope': 'every episode of this evaluation'}


class RolloutLog(_Log):
    """Episodes generated to train on, tagged with the algorithm that produced them."""
    KIND, RECORDS, ROWS = 'rollouts', 'rollouts.jsonl', 'rollouts.csv'
    COLS = ROLLOUT_COLS + H.METRIC_COLS + ('meta',)

    def __init__(self, root, run, algo, append=True):
        _Log.__init__(self, root, run, append=append)
        self.algo = algo

    def add(self, case, rec, **meta):
        stored = dict(meta, algo=self.algo)
        row_extra = {k: stored.get(k) for k in ROLLOUT_COLS}
        row_extra['meta'] = {k: v for k, v in meta.items() if k not in ROLLOUT_COLS} or None
        where = '  '.join('%s=%s' % (k, meta[k]) for k in ('step', 'group', 'candidate')
                          if meta.get(k) is not None)
        m = self._add(case, rec, 'rollout', stored, row_extra,
                      '[rollout %s | %s]' % (self.algo, where or '-'), meta)
        self.flush()
        return m

    def _summary_extra(self):
        return {'algo': self.algo,
                'scope': 'episodes written by this process; the files may also hold '
                         'earlier runs, since rollout logs append'}
