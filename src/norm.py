import re, unicodedata
CODE_MAP = [
    (r'ley federal del trabajo|(?<![a-z])lft(?![a-z])', 'LFT'),
    (r'constituci[oó]n pol[ií]tica|(?<![a-z])cpeum(?![a-z])', 'CPEUM'),
    (r'ley del seguro social|(?<![a-z])lss(?![a-z])', 'LSS'),
    (r'c[oó]digo civil federal|(?<![a-z])ccf(?![a-z])', 'CCF'),
    (r'c[oó]digo de comercio', 'CCom'),
    (r'ley general de responsabilidades|(?<![a-z])lgra(?![a-z])', 'LGRA'),
    (r'ley federal de procedimiento|(?<![a-z])lfpca(?![a-z])', 'LFPCA'),
    (r'ley de amparo', 'LAmparo'),
    (r'nom-?\s*\d{3}', 'NOM'),
    # --- extended after auditing the merged 90-case gold set: these codes appear
    # in the ORIGINAL 60 and were previously bucketed as UNK, which would have
    # collapsed distinct statutes into one token during article scoring.
    (r'ley federal para prevenir y eliminar la discriminaci[oó]n|(?<![a-z])lfped(?![a-z])', 'LFPED'),
    (r'ley de concursos mercantiles|(?<![a-z])lcm(?![a-z])', 'LCM'),
    (r'c[oó]digo fiscal de la federaci[oó]n|(?<![a-z])cff(?![a-z])', 'CFF'),
    (r'ley del impuesto sobre la renta|(?<![a-z])lisr(?![a-z])', 'LISR'),
    (r'ley del impuesto al valor agregado|(?<![a-z])liva(?![a-z])', 'LIVA'),
    (r'ley de migraci[oó]n', 'LMigracion'),
    (r'ley federal de los trabajadores al servicio del estado|(?<![a-z])lfts?e(?![a-z])', 'LFTSE'),
    (r'nmx-?\s*[a-z]', 'NMX'),
    (r'iso/?\s*iec|(?<![a-z])iso(?![a-z])', 'ISO'),
    (r'reforma constitucional', 'CPEUM'),
    (r'(?<![a-z])usmca(?![a-z])|t-?mec', 'USMCA'),
    (r'(?<![a-z])oit(?![a-z])', 'OIT'),
]
def _strip(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))
def normalize_articles(items):
    out=set()
    for raw in items or []:
        if not isinstance(raw,str) or not raw.strip(): continue
        s=_strip(raw.lower()).replace('\u2013','-').replace('\u2014','-')
        code=None
        for pat,name in CODE_MAP:
            if re.search(pat,s): code=name; break
        if code is None: code='UNK'
        if code in ('ISO','NMX'):
            m=re.search(r'(iso/?\s*iec\s*\d+(?::\d{4})?|nmx-?[a-z0-9-]+)', s)
            out.add(code+":"+re.sub(r'\s+','',m.group(1)) if m else code+":UNSPEC"); continue
        if code=='NOM':
            m=re.search(r'nom-?\s*(\d{3})-?\s*([a-z]+)?-?\s*(\d{4})?', s)
            if m: out.add("NOM:"+"-".join([g for g in m.groups() if g])); continue
        # Strip parenthesised years/notes first: '(2017)', '(IA responsable)'.
        # Without this a year is parsed as an article number (CPEUM:2017).
        body = re.sub(r'\([^)]*\)', ' ', s)
        body = re.sub(r'fracc?(ion|\.)?\s*[ivxlcdm]+', ' ', body)
        body = re.sub(r'p[aá]rrafo\s*\d+', ' ', body)
        for m in re.finditer(r'(\d{1,4})\s*-\s*(\d{1,4})(?![a-z0-9])', body):
            a,b=int(m.group(1)),int(m.group(2))
            if 0 < b-a < 30:
                for n in range(a,b+1): out.add(f"{code}:{n}")
        for m in re.finditer(r'(?:art[íi]?c?u?l?o?s?\.?\s*)?(\d{1,4})\s*(?:-|\s)?\s*([a-d])?(?![\d])', body):
            num=m.group(1); suf=m.group(2)
            out.add(f"{code}:{num}" + (f"-{suf.upper()}" if suf else ""))
    return out
