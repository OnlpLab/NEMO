import iobes
import pandas as pd
from typing import Dict, Any, List, Optional, Union

def to_dict(span):
    return {
        'label': span[0],
        'start': span[1],
        'end': span[2]
    }

def get_spans(docs, label_key, text_key):
    spans = []
    for i, doc in enumerate(docs):
        labels = doc[label_key]
        text = doc[text_key]
        spans.append({
                        'text': text,
                        'ents': [to_dict(x) for x in iobes.parse_spans_iobes(labels)]
                    })
    return spans

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

def get_ents(docs, label_key, text_key):
    ents = []
    for i, doc in enumerate(docs):
        labels = doc[label_key]
        morphs = doc[text_key]
        spans = iobes.parse_spans_iobes(labels)
        for span in spans:
            text = ' '.join(morphs[span.start:span.end])
            ents.append({ 'sent_id': i+1, 
                          'text': text, 
                          'cat': span.type})
    return pd.DataFrame(ents)

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
        self, parsed: List[Dict[str, Any]], page: bool = False, minify: bool = False
    ) -> str:
        """Render complete markup.
        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (str): Rendered HTML markup.
        """
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            rendered.append(self.render_ents(p["text"], p["ents"], p.get("title")))
        if page:
            docs = "".join([TPL_FIGURE.format(content=doc) for doc in rendered])
            markup = TPL_PAGE.format(content=docs, lang=self.lang, dir=self.direction)
        else:
            markup = "".join(rendered)
        if minify:
            return minify_html(markup)
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

