"""Conservative English/Russian extraction for explicit assertions.

This parser does not infer facts from questions, hypotheticals or reported speech.
Unsupported text belongs in episodic memory, not in guessed graph triples.
"""
import hashlib
import re

NAME = r"[A-Za-zА-Яа-яЁё][\w'’\-]*(?:\s+[A-Za-zА-Яа-яЁё][\w'’\-]*){0,3}?"
SUBJECT = rf"(?P<subject>{NAME})"


def statements(text):
    for sentence in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text.strip()):
        sentence = sentence.strip()
        if not sentence or sentence.endswith('?'):
            continue
        if re.search(r"\b(?:if|maybe|perhaps|might|could|would|если|возможно|вероятно)\b", sentence, re.I):
            continue
        yield sentence.rstrip('.!;').strip()


def _subject(value):
    return "user" if value.lower() in {"i", "user", "я", "пользователь"} else value.strip()


def _valid_subject(value):
    # Do not turn 'Alice said Bob works at Acme' into a fictional person.
    return not re.search(r"\b(?:said|says|think|thinks|that|сказал|сказала|думает|что|не|not)\b", value, re.I)


def _object_dates(value):
    start = end = None
    match = re.search(r"\s+(?:from|с)\s+(\d{4})\s+(?:to|until|по|до)\s+(\d{4})(?:\s+года?)?$", value, re.I)
    if match:
        start, end = f"{match[1]}-01-01", f"{match[2]}-12-31"
        value = value[:match.start()]
    else:
        match = re.search(r"\s+(?:since|in|с|в)\s+(\d{4})(?:\s+года?)?$", value, re.I)
        if match:
            start = f"{match[1]}-01-01"
            value = value[:match.start()]
    return value.strip(), start, end


def extract_assertions(text):
    facts = []
    patterns = [
        (rf"{SUBJECT}\s+(?:now\s+|currently\s+|теперь\s+|сейчас\s+)?(?:works?\s+(?:at|for)|worked\s+at|(?:am|is)\s+employed\s+(?:at|by)|работа(?:ет|ю|л|ла)\s+в)\s+(?:компании\s+)?(?P<object>.+)", "works_at", "company"),
        (rf"{SUBJECT}\s+(?:now\s+|сейчас\s+)?(?:lives?\s+in|lived\s+in|resides?\s+in|жив[её]т\s+в|живу\s+в|жил[аи]?\s+в)\s+(?P<object>.+)", "lives_in", "location"),
        (rf"{SUBJECT}\s+(?:moved\s+to|relocated\s+to|переехал[аи]?\s+в)\s+(?P<object>.+)", "lives_in", "location"),
        (rf"{SUBJECT}\s+(?:does\s+not\s+like|doesn't\s+like|do\s+not\s+like|don't\s+like|dislikes?|hates?|не\s+люб(?:ит|лю)|ненавидит)\s+(?P<object>.+)", "dislikes", "concept"),
        (rf"{SUBJECT}\s+(?:prefers?|предпочита(?:ет|ю))\s+(?P<object>.+)", "prefers", "concept"),
        (rf"{SUBJECT}\s+(?:absolutely\s+)?(?:likes?|loves?|enjoys?|люб(?:ит|лю))\s+(?P<object>.+)", "likes", "concept"),
    ]
    for sentence in statements(text):
        for pattern, predicate, kind in patterns:
            match = re.fullmatch(pattern, sentence, re.I)
            if not match or not _valid_subject(match['subject']):
                continue
            obj, start, end = _object_dates(match['object'])
            # Multiple clauses and negated/uncertain targets need a language model.
            if not obj or re.search(r"\b(?:but|and|that|но|и|что|not|не)\b", obj, re.I):
                continue
            if predicate == 'likes' and re.search(r"\bloves?\b", sentence, re.I):
                predicate = 'loves'
            elif predicate == 'dislikes' and re.search(r"\bhates?\b", sentence, re.I):
                predicate = 'hates'
            facts.append({"subject": _subject(match['subject']), "predicate": predicate,
                          "object": obj, "object_type": kind,
                          "valid_from": start, "valid_until": end})
            break
    return facts


def profile_payload(text):
    facts = []
    for sentence in statements(text):
        match = re.fullmatch(r"(?:my name is|меня зовут)\s+(.+)", sentence, re.I)
        if match:
            facts.append({"key": "name", "value": match[1], "category": "identity", "confidence": 0.8})
            continue
        match = re.fullmatch(r"(?:I am|I'm|user is|я)\s+(?:an?\s+)?(.+)", sentence, re.I)
        if match and not re.search(r"\b(?:not|не|prefer|предпочитаю|люблю|работаю|живу)\b", match[1], re.I):
            facts.append({"key": "role", "value": match[1], "category": "identity", "confidence": 0.7})
            continue
        for assertion in extract_assertions(sentence):
            subject, predicate, value = assertion['subject'], assertion['predicate'], assertion['object']
            prefix = "" if subject == 'user' else subject.lower().replace(' ', '_') + ':'
            if predicate in {'prefers', 'likes', 'dislikes', 'loves', 'hates'}:
                slot = 'preference:' + hashlib.sha256(value.lower().encode()).hexdigest()[:12]
                if re.search(r"\b(?:answers?|responses?|replies|communication)\b|ответ|общени", value, re.I):
                    slot = 'communication_style'
                facts.append({"key": prefix + slot, "value": f"{subject} {predicate} {value}",
                              "category": "preference", "confidence": 0.8})
            elif predicate in {'works_at', 'lives_in'} and assertion['valid_until'] is None:
                facts.append({"key": prefix + predicate, "value": f"{subject} {predicate} {value}",
                              "category": "identity", "confidence": 0.8})
    return {"facts": facts}
