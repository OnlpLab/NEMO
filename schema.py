from pydantic import BaseModel
from typing import Optional, List
from enum import Enum, IntEnum


class ModelName(str, Enum):
    token_single = 'token-single'
    token_multi = "token-multi"
    morph = "morph"

class MultiModelName(str, Enum):
    token_multi = "token-multi"


class MorphModelName(str, Enum):
    morph = "morph"


class NEMOQuery(BaseModel):
    sentences: str
    tokenized: Optional[bool]= False

    class Config:
        schema_extra = {
            "example": {
                "sentences": "עשרות אנשים מגיעים מתאילנד לישראל.\nתופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת.",
                "tokenized": False,
            }
        }


class Verbosity(IntEnum):
    BASIC = 0
    INTERMID = 1
    SYNTAX = 2


#response models
class NCRFPreds(BaseModel):
    tokenized_text: List[str]
    ncrf_preds: List[str]


class Morpheme(BaseModel):
    form: str
    # token_id: int
    nemo_morph: Optional[str] = None
    nemo_multi_align_morph: Optional[str] = None
    id: Optional[int]
    lemma: Optional[str] = None
    #upostag: Optional[str] = None
    #xpostag: Optional[str] = None
    pos: Optional[str] = None
    feats: Optional[str] = None
    head: Optional[int] = None
    deprel: Optional[str] = None
    #deps: Optional[str] = None


class Token(BaseModel):
    text: str
    ncrf_preds: Optional[str] = None
    nemo_single: Optional[str] = None
    nemo_multi: Optional[str] = None
    nemo_multi_align_token: Optional[str] = None
    nemo_morph_align_token: Optional[str] = None
    # id: int
    morphs: Optional[List[Morpheme]] = []

    def __iter__(self):
        return self.morphs.__iter__()

    def __next__(self):
        return self.morphs.__next__()


class Doc(BaseModel):
    text: Optional[str] = None
    tokens: List[Token]
    #morphs: Optional[List[Morpheme]] = [] # better? and add token_id to Morpheme
    ma_lattice: Optional[str] = None
    pruned_lattice: Optional[str] = None
    md_lattice: Optional[str] = None
    dep_tree: Optional[str] = None
    
    def __iter__(self):
        return self.tokens.__iter__()

    def __next__(self):
        return self.tokens.__next__()

    @classmethod
    def iter_token_attrs(self, attr):
        for i, token in enumerate(self):
            yield i, getattr(token, attr)

    @classmethod
    def iter_morph_attrs(self, attr):
        for i, token in enumerate(self):
            for morph in token:
                yield i, getattr(morph, attr)
