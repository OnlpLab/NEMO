from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


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

#verbosity: 
#0 - tokens, morpheme forms (if morph endpoint) and final nemo preds
#1 - adds intermediate nemo preds
#2 - adds full morpho-syntactic preds for morphemes
#3 - adds yap raw output
# use response_model_exclude_unset

#response models
class NEMODoc(BaseModel):
    text: Optional[str] = None
    tokens: List[Token]
    #morphs: Optional[List[Morpheme]] = [] # better? and add token_id to Morpheme

    # verbose=3
    ma_lattice: Optional[str] = None
    md_lattice: Optional[str] = None
    pruned_lattice: Optional[str] = None
    dep_tree: Optional[str] = None
    
    @classmethod
    def get_morphs(self):
        for i, token in enumerate(self.tokens):
            for morph in token.morphs:
                yield i, morph


class Token(BaseModel):
    text: str
    # id: int
    morphs: Optional[List[Morpheme]] = []
    nemo_single: Optional[str] = None
    nemo_multi: Optional[str] = None
    nemo_multi_align_token: Optional[str] = None
    nemo_morph_align_token: Optional[str] = None


class Morpheme(BaseModel):
    form: str
    # token_id: int
    nemo_morph: Optional[str] = None
    nemo_multi_align_morph: Optional[str] = None
    # verbose=2
    id: Optional[int]
    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: Optional[str] = None
    head: Optional[str] = None
    deprel: Optional[str] = None
