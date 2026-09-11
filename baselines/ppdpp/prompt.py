import json

ESConvAct = {"Question": "Please ask the Patient to elaborate on the situation they just described.",
            "Self-disclosure": "Please provide a statement relating to the Patient about the situation they just described.",
            "Affirmation and Reassurance": "Please provide affirmation and reassurance to the Patient on the situation they just described.",
            "Providing Suggestions": "Please provide suggestion to the Patient on the situation they just described.",
            "Others": "Please chat with the Patient.",
            "Reflection of feelings": "Please acknowledge the Patient's feelings about the situation they described.",
            "Information": "Please provide factual information to help the Patient with their situation.",
            "Restatement or Paraphrasing": "Please acknowledge the Patient's feelings by paraphrasing their situation."}

CIMAAct = {"Hint": "Please provide knowledge to the Student via a hint.",
           "Question": "Please ask a question to the Student to determine the Student's understanding or continue the conversation.",
           "Correction": "Please correct the mistake or addresses the misconception the Student has.",
           "Confirmation": "Please confirm the tudent's answer or understanding is correct.",
           "Others": "Please chat with the Student without any pedagogical strategy."}

CBAct = {'greet': 'Please say hello or chat randomly.',
         'inquire': 'Please ask any question about product, year, price, usage, etc.',
         'inform': 'Please provide information about the product, year, usage, etc.',
         'propose': 'Please initiate a price or a price range for the product.',
         'counter': 'Please propose a new price or a new price range.',
         'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
         'confirm': 'Please ask a question about the information to be confirmed.',
         'affirm': 'Please give an affirmative response to a confirm.',
         'deny': 'Please give a negative response to a confirm.',
         'agree': 'Please agree with the proposed price.',
         'disagree': 'Please disagree with the proposed price.'}


# Four intents, named so the annotation boundary is legible from the name itself:
# ASK opens a new topic, FOLLOWUP refers back to an answer already given.
CSAAct = {
    'ask': 'Please ask one named participant for information relevant to the decision.',
    'followup': 'Please push on an answer already given: get the exact detail, number or '
                'date it left vague, or contest it.',
    'share': 'Please tell the others what one participant has disclosed, and ask how it '
             'affects the decision.',
    'decide': 'Please put forward or record the settlement now, as a single JSON object '
              'matching the settlement schema and nothing else.',
}

# The act that produces a settlement, and therefore triggers scoring.
CSA_SETTLING_ACT = 'decide'

# The answer key. None of these may ever reach a participant prompt.
CSA_ORACLE_ONLY = ('content_checks', 'provenance_checks', 'acceptance_conditions',
                   'decisive_facts')

# Acts that seek information from a participant. Used to distinguish a fact the chair
# drew out from one an advisor volunteered -- see the elicitation metric.
CSA_ELICITING_ACTS = ('ask', 'followup')


def _csa_agent(case, agent_id):
    return next(a for a in case['agents'] if a['agent_id'] == agent_id)


def _csa_visible(case, agent_id):
    keys = case['views'][agent_id]
    shared = {k: v for k, v in case['shared_context'].items() if k in keys}
    private = {k: v['text'] for k, v in case['private_facts'].items() if k in keys}
    return shared, private


def _csa_render(case, agent_id):
    """The persona block for one agent, filtered to that agent's view."""
    shared, private = _csa_visible(case, agent_id)
    agent = _csa_agent(case, agent_id)
    lines = ['You are %s, %s.' % (agent['name'], agent['role']),
             'Scenario: %s' % case['description'],
             'What everyone in the meeting knows:']
    lines += ['  %s: %s' % (k, v) for k, v in sorted(shared.items())]
    if private:
        lines.append('What only you know (the others cannot see this; share it when it '
                     'bears on the decision):')
        lines += ['  %s: %s' % (k, v) for k, v in sorted(private.items())]
    return '\n'.join(lines)


def _csa_roster(case):
    return ', '.join('%s (%s)' % (a['name'], a['role']) for a in case['agents'])


def _csa_dial(conversation):
    return ' '.join('%s: %s' % (t['role'], t['content']) for t in conversation)


def CSAMessages(case, role, conversation, action=None, agent_id=None):
    """Message builder for Coevolving Social Agents.

    Differs from the two-party builders in that `user` is one of several advisors,
    selected by agent_id, and every participant prompt is filtered through case['views']
    so hidden profiles stay hidden. action=None is the no-planner control.
    """
    dm = case['decision_maker']

    if role == 'system':
        head = ('Now enter the role-playing mode. You are chairing a meeting that must '
                'reach one decision. The other participants are %s.\n\n%s'
                % (_csa_roster(case), _csa_render(case, dm)))
        instr = ('You lead this discussion and you alone will record the final settlement. '
                 'The others hold information you do not have; it is your job to draw it '
                 'out. Please reply with only one short and succinct turn.')
        if action is not None:
            instr = '%s %s' % (instr, CSAAct[action])
        if action == CSA_SETTLING_ACT:
            instr += ('\nThe settlement schema is: %s\nReply with the JSON object only.'
                      % json.dumps(case['settlement_schema']))
        messages = [{"role": "system", "content": head},
                    {"role": "USER", "content": instr}]
        messages.extend(conversation)

    elif role == 'user':
        agent = _csa_agent(case, agent_id)
        head = ('Now enter the role-playing mode. You are taking part in a meeting chaired '
                'by %s. The participants are %s.\n\n%s'
                % (_csa_agent(case, dm)['name'], _csa_roster(case),
                   _csa_render(case, agent_id)))
        instr = ('You are advising the chair, who cannot see what only you know. Please '
                 'reply with only one short and succinct turn. Answer direct questions '
                 'truthfully. Are you ready to play the game?')
        messages = [{"role": "system", "content": head},
                    {"role": "USER", "content": instr},
                    {"role": agent['name'], "content": "Yes, I'm ready to play the game!"}]
        messages.extend(conversation)

    elif role == 'critic':
        # Oracle judge for arm A. Sees the hidden facts so it can tell whether they were
        # pooled, but never the executable checks or acceptance conditions -- those are
        # reserved for the independent measurement.
        facts = '\n'.join('  %s (held by %s): %s' % (k, v['owner'], v['text'])
                          for k, v in sorted(case['private_facts'].items()))
        messages = [
            {"role": "system", "content":
                "Given a meeting transcript, please assess whether the chair's settlement "
                "reflects the information that the other participants privately held."},
            {"role": "USER", "content":
                "You can only reply with one of the following sentences: "
                "No, the group has not reached a settlement. "
                "No, the settlement ignores the information the advisors held. "
                "No, but the settlement reflects some of the information the advisors held. "
                "Yes, the settlement reflects all of the decisive information the advisors "
                "held.\n\nDecision to be made: %s\n\nInformation privately held by the "
                "advisors:\n%s\n\nThe following is the meeting: %s\nQuestion: Does the "
                "settlement reflect the information the advisors held? Answer: "
                % (case['description'], facts, _csa_dial(conversation))}]

    elif role == 'settlement':
        # Fallback extractor, used only when the chair never emitted parseable JSON.
        # Structured extraction, not judgement: it scores nothing.
        messages = [
            {"role": "system", "content":
                "Extract the final settlement from the meeting transcript."},
            {"role": "USER", "content":
                "Return only a JSON object matching this schema, with no commentary: %s\n\n"
                "Use exactly the wording the chair used. If the chair never settled a "
                "field, use an empty string. 'credited_facts' and 'justification_fact_ids' "
                "must list fact identifiers such as PF1 that the chair actually relied on."
                "\n\nMeeting: %s\nJSON: "
                % (json.dumps(case['settlement_schema']), _csa_dial(conversation))}]

    elif role == 'acceptance':
        # One condition at a time, greedily -- a narrow yes/no, not a holistic verdict.
        messages = [
            {"role": "system", "content":
                "You check whether a recorded decision satisfies one stated requirement. "
                "Judge only the requirement you are given. Answer with exactly one word: "
                "Yes or No."},
            {"role": "USER", "content":
                "Recorded settlement:\n%s\n\nRequirement: %s\n\nDoes the recorded "
                "settlement satisfy this requirement? Answer Yes or No."
                % (json.dumps(action, indent=1)[:4000], agent_id)}]

    return messages


def ESConvMessages(case, role, conversation, action=None):
    if role == 'system':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."}, {"role":"PATIENT", "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. Please reply with only one short and succinct sentence. %s Are you ready to play the game?" % ESConvAct[action]}, {"role":"Therapist", "content":"Yes, I'm ready to play the game!"}]
        messages.extend(conversation)
    elif role == 'user':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist."}, {"role":"Therapist", "content":"You are the patient who is looking for the help from the therapist, because you have the emotional issue about %s regarding %s. Please reply with only one short and succinct sentence. Now tell me your issue." % (case['emotion_type'], case['problem_type'])}]
        messages.extend(conversation)
    elif role == 'critic':
        dial = ''
        for turn in conversation:
            dial += '%s: %s ' % (turn['role'], turn['content'])

        messages = [{"role":"system", "content":"Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation."}, {"role":"USER", "content":"You can only reply with one of the following sentences: No, the Patient feels worse. No, the Patient feels the same. No, but the Patient feels better. Yes, the Patient's issue has been solved.\n\nThe following is a conversation about %s regarding %s: %s\nQuestion: Has the Patient's issue been solved? Answer: " % (case['emotion_type'], case['problem_type'], dial)}] 
        

    
    return messages


def CIMAMessages(case, role, conversation, action=None):
    if role == 'system':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a teacher in a tutoring conversation with a student."}, {"role":"Student", "content": "You are the teacher who is trying to teach the student to translate \"%s\" into Italian. Please reply with only one short and succinct sentence. Please do not tell the student the answer or ask the student about other exercises. %s Now ask me an exercise." % (case['sentence'], CIMAAct[action])}]
        messages.extend(conversation)
    elif role == 'user':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a student who does not know Italian in a tutoring conversation with a teacher."}, {"role":"Teacher", "content":"You are the student who is trying to translate a English sentence into Italian. You don't know the translation of \"%s\" in Italian. Please reply with only one short and succinct sentence. Are you ready to play the game?" % case['sentence']}, {"role":"Student", "content":"Yes, I'm ready to play the game!"}]
        messages.extend(conversation)
    elif role == 'critic':
        
        dial = ''
        for turn in conversation:
            dial += '%s: %s ' % (turn['role'], turn['content'])

        messages = [{"role":"system", "content":"Given a conversation between a Teacher and a Student, please assess whether the Student correctly translate the English sentence into Italian in the conversation."}, {"role":"USER", "content":"Please assess whether the Student correctly translated the whole sentence of \"%s\" into Italian in the conversation. You can only reply with one of the following sentences: No, the Student made an incorrect translation. No, the Student did not try to translate. No, the Student only correctly translated a part of \"%s\". Yes, the Student correctly translated the whole sentence of \"%s\".\n\nThe following is the conversation: %s\nQuestion: Did the Student correctly translated the whole sentence of \"%s\" into Italian? Answer: " % (case['sentence'], case['sentence'], case['sentence'], dial, case['sentence'])}] 

    return messages



def CBMessages(case, role, conversation, action=None):
    if role == 'system':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game."}, {"role":"Seller", "content": "You are the buyer who is trying to buy the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. %s Now start the game." % (case['item_name'], case['buyer_price'], case['buyer_item_description'], CBAct[action])}]
        messages.extend(conversation)
    elif role == 'user':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game."}, {"role":"Buyer", "content": "You are the seller who is trying to sell the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. Are you ready to play the game?" % (case['item_name'], case['seller_price'], case['seller_item_description'])}, {"role":"Seller", "content":"Yes, I'm ready to play the game!"}]
        messages.extend(conversation)
    elif role == 'critic':
        
        dial = ''
        for turn in conversation:
            dial += '%s: %s ' % (turn['role'], turn['content'])

        messages = [{"role":"system", "content":"Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."}, {"role":"USER", "content":"Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15.\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal.\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % dial}] 
    

    return messages

def vicuna_prompt(messages, role):
    seps = [' ', '</s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'USER: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret

def llama2_prompt(messages, role):
    seps = [' ', ' </s><s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'USER: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + " " + message['content'] + seps[i % 2]
    ret += '%s' % role
    return ret

def qwen_prompt(messages, role):
    """Map PPDPP's message list onto ChatML roles for a chat-templated local model.

    Same role assignment as chatgpt_prompt -- the named speaker becomes the assistant,
    everyone else the user -- but the result is fed to the tokenizer's own chat template
    rather than a hand-assembled separator string.
    """
    out = [{'role': 'system', 'content': messages[0]['content']}]
    for m in messages[1:]:
        out.append({'role': 'assistant' if m['role'] == role else 'user',
                    'content': m['content']})
    return out


def chatgpt_prompt(messages, role):
    #print(messages)
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'] == role:
            new_messages.append({'role':'assistant', 'content':message['content']})
        elif message['role'] != role:
            new_messages.append({'role':'user', 'content':message['content']})
    return new_messages



