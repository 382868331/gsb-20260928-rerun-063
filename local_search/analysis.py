import re
from typing import Dict, List

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(value: object) -> List[str]:
    return TOKEN_RE.findall(str(value).lower())


def field_tokens(fields: Dict[str, object]) -> Dict[str, List[str]]:
    indexed = {}
    for name, value in fields.items():
        if value is not None:
            indexed[name] = tokenize(value)
    return indexed
