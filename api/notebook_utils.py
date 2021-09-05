import iobes
import pandas as pd
from typing import Dict, Any, List, Optional, Union

def to_dict(span, text):
    return {
        'text': ' '.join(text[span[1]:span[2]]),
        'label': span[0],
        'start': span[1],
        'end': span[2]
    }


def iter_token_attrs(doc, attr):
    for i, token in enumerate(doc['tokens']):
        yield token[attr]


def iter_morph_attrs(doc, attr):
    for i, token in enumerate(doc['tokens']):
        for morph in token['morphs']:
            yield morph[attr]


def ents_to_df(docs):
    sc = []
    for i, doc in enumerate(docs):
        for level, scenarios in doc['ents'].items():
            for scenario, ents in scenarios.items():
                for ent in ents: 
                    sc.append({
                        'sent_id': i,
                        'text': ent['text'],
                        'label': ent['label'],
                        'level': level,
                        'scenario': scenario
                    })
    return pd.DataFrame(sc)

def escape_html(text: str) -> str:
    """Replace <, >, &, " with their HTML encoded representation. Intended to
    prevent HTML errors in rendered displaCy markup.
    text (str): The original text.
    RETURNS (str): Equivalent text to be safely used within HTML.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


DEFAULT_LANG = "he"
DEFAULT_DIR = "rtl"
DEFAULT_ENTITY_COLOR = "#ddd"


TPL_ENTS = """
<div class="entities" style="line-height: 2.5; direction: {dir}">{content}</div>
"""


TPL_ENT = """
<mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    {text}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
"""

TPL_ENT_RTL = """
<mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em">
    {text}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-right: 0.5rem">{label}</span>
</mark>
"""


TPL_PAGE = """
<!DOCTYPE html>
<html lang="{lang}">
    <head>
        <title>displaCy</title>
    </head>
    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: {dir}">{content}</body>
</html>
"""

DEFAULT_LABEL_COLORS = {
    "ORG": "#7aecec",
    "DUC": "#bfeeb7",
    "GPE": "#feca74",
    "LOC": "#ff9561",
    "PER": "#aa9cfc",
    "FAC": "#9cc9cc",
    "EVE": "#ffeb80",
    "LAW": "#ff8197",
    "ANG": "#ff8197",
    "WOA": "#f0d0ff",
}

class EntityRenderer:
    """Render named entities as HTML."""

    style = "ent"

    def __init__(self, options: Dict[str, Any] = {}) -> None:
        """Initialise dependency renderer.
        options (dict): Visualiser-specific options (colors, ents)
        """
        colors = dict(DEFAULT_LABEL_COLORS)
        colors.update(options.get("colors", {}))
        self.default_color = DEFAULT_ENTITY_COLOR
        self.colors = {label.upper(): color for label, color in colors.items()}
        self.ents = options.get("ents", None)
        if self.ents is not None:
            self.ents = [ent.upper() for ent in self.ents]
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG
        template = options.get("template")
        if template:
            self.ent_template = template
        else:
            if self.direction == "rtl":
                self.ent_template = TPL_ENT_RTL
            else:
                self.ent_template = TPL_ENT

    def render(
        self, res: dict, level: str, scenario: str
    ) -> str:
        """Render complete markup.
        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (str): Rendered HTML markup.
        """
        rendered = []
        for i, p in enumerate(res):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            text = list(iter_token_attrs(p, 'text')) if level=='token' else list(iter_morph_attrs(p, 'form'))
            rendered.append(self.render_ents(text, p['ents'][level][scenario], p.get("title")))
        else:
            markup = "".join(rendered)
        return markup

    def render_ents(
        self, text: str, spans: List[Dict[str, Any]], title: Optional[str]
    ) -> str:
        """Render entities in text.
        text (str): Original text.
        spans (list): Individual entity spans and their start, end and label.
        title (str / None): Document title set in Doc.user_data['title'].
        """
        markup = ""
        offset = 0
        for span in spans:
            label = span["label"]
            start = span["start"]
            end = span["end"]
            additional_params = span.get("params", {})
            prev_text = escape_html(' '.join(text[offset:start]))
            markup += prev_text
            entity = escape_html(' '.join(text[start:end]))

            if self.ents is None or label.upper() in self.ents:
                color = self.colors.get(label.upper(), self.default_color)
                ent_settings = {"label": label, "text": entity, "bg": color}
                ent_settings.update(additional_params)
                markup += self.ent_template.format(**ent_settings)
            else:
                markup += entity
            offset = end
        last_text = escape_html(' '.join(text[offset:]))
        markup += last_text
        markup = TPL_ENTS.format(content=markup, dir=self.direction)
        return markup

