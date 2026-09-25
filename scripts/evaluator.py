"""Frozen, video-only teacher roles. No MOS or synthesis metadata interface."""
import collections
import json
import re

LABELS = ('A', 'B', 'near_equal', 'abstain')
RUBRIC = ('Judge overall human-perceived quality of rendered video, considering visible spatial defects '
          'and temporal stability. Ignore artistic appeal and object identity. Infer causes only as hypotheses. '
          'Do not assume equal timestamps in A/B show the same geometry. ')
SCHEMA = ('Return compact JSON with preference (A, B, near_equal or abstain), confidence (0 to 1), '
          'and evidence: a list of at most 2 objects with video (A or B), time (original seconds), '
          'observation (brief visible fact). near_equal means evidence supports similar quality; '
          'abstain means insufficient evidence. ')


def parse(text):
    decoder = json.JSONDecoder()
    for index in (m.start() for m in re.finditer(r'\{', text)):
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    return {}


def majority(labels):
    valid = [x for x in labels if x in LABELS and x != 'abstain']
    if not valid:
        return 'abstain', 0.
    label, count = collections.Counter(valid).most_common(1)[0]
    return (label if count > len(valid)/2 else 'abstain'), count/len(valid)


class Evaluator:
    def __init__(self, runner, config):
        self.runner, self.config = runner, config

    def evaluate(self, paths, method):
        records = []
        def call(role, prompt, windows=None, swapped=False):
            if len(records) >= self.config['max_calls']:
                return {'preference':'abstain', 'budget_stop':True}
            actual = paths[::-1] if swapped else paths
            inputs, trace = self.runner.inputs(actual, RUBRIC+prompt, local=True, windows=windows)
            text, _ = self.runner.generate(inputs, max_tokens=self.config['report_tokens'],
                                            temperature=self.config['temperature'])
            parsed = parse(text)
            if swapped and parsed.get('preference') in ('A','B'):
                parsed['preference'] = 'B' if parsed['preference']=='A' else 'A'
            records.append({'role':role, 'swapped':swapped, 'output':text, 'parsed':parsed,
                            'observations':trace, 'cost':self.runner.timings[-1]})
            return parsed

        if method == 'baseline':
            votes = [call('independent_vote', SCHEMA, swapped=bool(i%2))
                     for i in range(self.config['baseline_votes'])]
            label, agreement = majority([v.get('preference') for v in votes])
            return {'label':label, 'reliability':agreement, 'records':records}

        neutral = call('neutral', SCHEMA)
        claims = []
        for _ in range(self.config['advocate_pairs']):
            for side in ('A','B'):
                report = call('advocate_'+side,
                    f'Search for visual evidence that video {side} has better quality. '
                    'Keep contrary observations and acknowledge weak evidence; never invent defects. '+SCHEMA)
                # Strip role identity and confidence from auditor input.
                for claim in report.get('evidence', [])[:2]:
                    if isinstance(claim, dict):
                        claims.append({k:claim[k] for k in ('video','time','observation') if k in claim})
        # Deduplicate shared claims; repeated descriptions are not independent votes.
        claims = list({json.dumps(c,sort_keys=True):c for c in claims}.values())
        audit_prompt = ('Check these claims against the pixels: '+json.dumps(claims)+
            '. Return JSON with supported (list of claim indices), contradicted (list), '
            'unresolved (list), and question (the decisive remaining visual issue, or empty string).')
        audit = call('auditor', audit_prompt)
        audits = [audit]

        def recurse(parent, depth):
            unresolved = parent.get('unresolved', [])
            if not unresolved or depth > self.config['recursion_depth']:
                return
            windows = {0:[], 1:[]}
            for index in unresolved[:self.config['branching_factor']]:
                if not isinstance(index,int) or not 0 <= index < len(claims):
                    continue
                claim = claims[index]
                side = 0 if claim.get('video')=='A' else 1
                try:
                    start = max(0.,float(claim.get('time',2.))-.5)
                except (ValueError,TypeError):
                    start = 2.
                windows[side].append((start,1.))
            child = call('local_audit_depth_'+str(depth),
                'Resolve this visual subquestion using the added local observations: '+
                str(parent.get('question',''))+' '+audit_prompt, windows=windows)
            audits.append(child)
            recurse(child,depth+1)
        recurse(audit,1)
        ledger = {'claims':claims,'audits':audits,'independent_initial':neutral}
        conclusions = [call('hyperjudge', 'Independently judge overall quality using the videos and '
                           'this evidence ledger; prioritize verified perceptual impact over report count. '+
                           json.dumps(ledger)+' '+SCHEMA)
                       for _ in range(self.config['hyperjudges'])]
        label, agreement = majority([h.get('preference') for h in conclusions])
        denominator = max(1,len(claims))
        supported, unresolved = set(), set(range(len(claims)))
        for a in audits:
            valid = lambda key:{i for i in a.get(key,[]) if isinstance(i,int) and 0<=i<len(claims)}
            s,c,u = valid('supported'),valid('contradicted'),valid('unresolved')
            supported = (supported | s) - c
            unresolved = (unresolved - s - c) | u
        support = len(supported)/denominator
        resolved = 1-len(unresolved)/denominator if claims else 0.
        reliability = agreement*(.5+.5*support)*(.5+.5*resolved)
        return {'label':label,'reliability':reliability,'hyperjudge_agreement':agreement,
                'support_fraction':support,'resolved_fraction':resolved,'ledger':ledger,'records':records}
