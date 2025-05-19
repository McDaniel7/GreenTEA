#######################
import re, string


def _replace_date(text):
    text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', text)
    text = re.sub('\s(\d*\s\w*\s\d*)\s', ' DATE ', text)
    text = re.sub('\d{1}k\d{2}', ' DATE ', text)
    return text


def _replace_tracking(text):
    text = re.sub(
        r"1Z[A-H,J-N,P,R-Z,0-9]{16}|[kKJj]{1}[0-9]{10}|[0-9]{12}|[0-9]{15}|[0-9]{20,22}|"
        + r"[A-Z]{2}[0-9,A-Z]{9}US|[0-9]{10,11}|[0-9]{16}|[A-Z]{2}[0-9]{9}[A-Z]{2}",
        " tracking ",
        text,
    )
    return text


def _replace_orderid(text):
    text = re.sub(
        r"[0-9]{3}\-[0-9]{7}\-[0-9]{7}|[0-9]{17}|[0-9]{3} [0-9]{7} [0-9]{7}", " orderid ", text
    )
    return text


def _remove_number(word):
    return re.sub(r"\d+", "", word)


def _replace_days(text):
    weekdays = (
            r"mon|tue|wed|thu|fri|sat|sun|"
            + r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    )
    months = (
            r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            + r"january|february|march|april|may|june|july|august|september|october|november|december"
    )
    sep = r"[., :]?\s+"
    day = r"\d+"
    year = r"\d+"
    day_or_year = r"\d+(?:\w+)?"
    text = re.sub(
        rf"(?:{weekdays}){sep}(?:{day}{sep})?(?:{months}){sep}{day_or_year}(?:{sep}{year})?{sep}"
        + rf"([0-2]\d:[0-6]\d:[0-6]\d){sep}(?:am|pm)?",
        " date ",
        text,
    )
    return text


# def _expand_contractions(text, nlp_model):
#     lemmatized_text = ' '.join([token.lemma_ for token in nlp_model(text)])
#     return lemmatized_text


def _decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def _anno_cleanup(text):
    text = re.sub(
        r"(?<![a-zA-Z])v(?![a-zA-Z])",  # vc|vcx|vet|vt
        " with ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vo|w/o|w.o(?![a-zA-Z])",  # vc|vcx|vet|vt
        " without",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])cs(?![a-zA-Z])",  # vc|vcx|vet|vt
        " customer service ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])ref(?![a-zA-Z])",  # vc|vcx|vet|vt
        " referral ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])rlr(?![a-zA-Z])",  # vc|vcx|vet|vt
        " returnless refund rlr ",
        str(text),
    )
    ext = re.sub(
        r"(?<![a-zA-Z])nv(?![a-zA-Z])",  # vc|vcx|vet|vt
        " normal veteran ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vc(?![a-zA-Z])",  # vc|vcx|vet|vt
        " veteran customer ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vcx(?![a-zA-Z])",  # vc|vcx|vet|vt
        " veteran customer ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vet(?![a-zA-Z])",  # vc|vcx|vet|vt
        " veteran customer ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vt(?![a-zA-Z])",  # vc|vcx|vet|vt
        " veteran customer ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])c(?![a-zA-Z])",  # vc|vcx|vet|vt
        " customer ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])vcac(?![a-zA-Z])",  # vcac
        " veteran customer account compromise ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])n(?![a-zA-Z])",  # vc|vcx|vet|vt
        " and ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])od(?![a-zA-Z])",  # oh
        " order ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])oh(?![a-zA-Z])",  # oh
        " order history ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])hoh(?![a-zA-Z])",  # hoh
        " hihg order history ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])moh(?![a-zA-Z])",  # moh
        " medium order history ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ab(?![a-zA-Z])",  # ab
        " abuse ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])rels(?![a-zA-Z])",  # rel|rels
        " relations",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])rel(?![a-zA-Z])",  # rel|rels
        " relations",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tsds(?![a-zA-Z])",  # tds|tsd|tsds
        " tracking shows delivered",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tsd(?![a-zA-Z])",  # tds|tsd|tsds
        " tracking shows delivered",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tsdbnr(?![a-zA-Z])",  # tsdbnr
        " tracking shows delivered but not received ",
        str(text)
    )
    text = re.sub(
        r">>",
        " ",
        str(text)
    )
    text = re.sub(
        r">",
        " larger than ",
        str(text)
    )
    text = re.sub(
        r"<",
        " less than ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])prim(?![a-zA-Z])",  # prim|pri
        " primary ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])pri(?![a-zA-Z])",  # prim|pri
        " primary ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])cx(?![a-zA-Z])",  # cx
        " customer ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])nc(?![a-zA-Z])",  # nc|ncx
        " normal customer ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ncx(?![a-zA-Z])",  # nc|ncx
        " normal customer ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])rm(?![a-zA-Z])",  # s
        " risk mining ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])fr(?![a-zA-Z])",  # s
        " for ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])sol(?![a-zA-Z])",  # s
        " solicit ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])s(?![a-zA-Z])",  # s
        " solicit ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])w(?![a-zA-Z])",  # s
        " warned ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])pst(?![a-zA-Z])",  # pw
        " post ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])pw(?![a-zA-Z])",  # pw
        " post warning ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ps(?![a-zA-Z])",  # ps
        " post solicit ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])psw(?![a-zA-Z])",  # psw
        " post solicit warning ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])sw(?![a-zA-Z])",  # sw
        " solicit warning ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])nw(?![a-zA-Z])",  # nw
        " newly warned ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])pa(?![a-zA-Z])",  # pa
        " post action ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])pcoa(?![a-zA-Z])",  # pcoa
        " post closure open account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])cap(?![a-zA-Z])",  # cap
        " concession abuse prevention ",
        str(text),
    )
    text = re.sub(
        r"(?<![a-zA-Z])mod(?![a-zA-Z])",  # mod
        " moderate ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])a/c|acc(?![a-zA-Z])",  # a/c
        " account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])acc(?![a-zA-Z])",  # a/c
        " account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])acct(?![a-zA-Z])",  # a/c
        " account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ac(?![a-zA-Z])",  # a/c
        " account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])med(?![a-zA-Z])",  # med
        " medium ",
        str(text)
    )
    #     text = re.sub(
    #         r"(![a-zA-Z]).n(![a-zA-Z])", # .n
    #         " normal ",
    #         text
    #     )
    text = re.sub(
        r"(?<![a-zA-Z])n/a(?![a-zA-Z])",  # a/c
        " not available ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])q(?![a-zA-Z])",  # a/c
        " queued ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])sa(?![a-zA-Z])",  # sa
        " shipping address ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])cc(?![a-zA-Z])",  # cc
        " credit card ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ip(?![a-zA-Z])",  # ub|ip|fu
        " device ID ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ub(?![a-zA-Z])",  # ub|ip|fu
        " device ID ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])fu(?![a-zA-Z])",  # ub|ip|fu
        " device ID ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])axn(?![a-zA-Z])",  # axn
        " action ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tsh(?![a-zA-Z])",  # tsh|ths
        " thresholds ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])thes(?![a-zA-Z])",  # tsh|ths
        " thresholds ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])thresh(?![a-zA-Z])",  # tsh|ths
        " thresholds ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])thres(?![a-zA-Z])",  # tsh|ths
        " thresholds ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ths(?![a-zA-Z])",  # tsh|ths
        " thresholds ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tds(?![a-zA-Z])",  # tds
        " thresholds",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])th(?![a-zA-Z])",  # tds
        " thresholds",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])f(?![a-zA-Z])",  # tds
        " fraud",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ins(?![a-zA-Z])",  # ins
        " instance ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])ph(?![a-zA-Z])",  # ph
        " phone ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])mul(?![a-zA-Z])",  # mul
        " multiple",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])moa(?![a-zA-Z])",  # moa
        " mobile only account ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])tr(?![a-zA-Z])",  # tr
        " temporarily reinstate ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])hv(?![a-zA-Z])",  # hv
        " high value ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])es(?![a-zA-Z])",  # es
        " easy ship ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])rrnm(?![a-zA-Z])",  # rrnm
        " return reason not match ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])rrm(?![a-zA-Z])",  # rrm
        " return reason match ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])gud|gd(?![a-zA-Z])",  # gud
        " good ",
        str(text)
    )
    text = re.sub(
        r"(?<![a-zA-Z])rma(?![a-zA-Z])",  # gud
        " return mailing label ",
        str(text)
    )
    return text


def _remove_punctuation(word):
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    result = regex.sub(" ", str(word))
    return result


def _remove_whitespace(word):
    word = re.sub(r"\n|\r|,|\t", " ", str(word).replace("nextrow", " "))
    word = re.sub(r" +", " ", word)
    return word.strip()


def _remove_empty_anno(words):
    words = _remove_whitespace(words)
    if words == '':
        words = 'No comments'
    return words


def annotation_process(texts):
    clean_comments = []
    for row in texts.index:
        clean_text = _anno_cleanup(texts[row])
        clean_text = _remove_whitespace(_remove_punctuation(clean_text))
        clean_comments.append(clean_text)
    return clean_comments
