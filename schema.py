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

#verbosity: 
#0 - tokens, morphemes (if morph endpoint) and final requested nemo preds
#       - morphological features: 'form', 'lemma', 'upostag', 'xpostag', 'feats'
#1 - adds intermediate nemo preds (for example when running morph_hybrid, 
#                                   you also get nemo_multi, nemo_multi_align_token, nemo_multi_align_morph)
#2 - adds syntactic tree info: 'head', 'deprel', 'deps'
#include_yap_output - adds yap raw output
# use response_model_exclude_unset


#response models
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
    # id: int
    morphs: Optional[List[Morpheme]] = []
    nemo_single: Optional[str] = None
    nemo_multi: Optional[str] = None
    nemo_multi_align_token: Optional[str] = None
    nemo_morph_align_token: Optional[str] = None


class Doc(BaseModel):
    text: Optional[str] = None
    tokens: List[Token]
    #morphs: Optional[List[Morpheme]] = [] # better? and add token_id to Morpheme
    ma_lattice: Optional[str] = None
    md_lattice: Optional[str] = None
    pruned_lattice: Optional[str] = None
    dep_tree: Optional[str] = None
    
    @classmethod
    def get_morphs(self):
        for i, token in enumerate(self.tokens):
            for morph in token.morphs:
                yield i, morph