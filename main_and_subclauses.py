import itertools as it
import typing as tp
import pandas as pd
import spacy
import numpy as np
import time


VERB_POS = {"VERB", "AUX"}
FINITE_VERB_TAGS = {"VVFIN", "VMFIN", "VAFIN"}


class Clause:
    def __init__(self, spans: tp.Iterable["spacy.tokens.Span"]):
        """Clause is a sequence of potentially divided spans.

        This class basically identifies a clause as subclause and
        provides a string representation of the clause without the
        commas stemming from interjecting subclauses.

        A clause can consist of multiple unconnected spans, because
        subclauses can divide the clause they are depending on. That's
        why a clause cannot just be constituted by a single span, but
        must be based on an iterable of spans.
        """

        self.spans = spans

    @property
    def __chain(self) -> tp.Iterable["spacy.tokens.Token"]:
        return [token for token in it.chain(*self.spans)]

    # We make this class an iterator over the tokens in order to
    #  mimic span behavior. This is what we need the following
    #  dunder methods for.
    def __getitem__(self, index: int) -> "spacy.tokens.Token":
        return self.__chain[index]
    

    def __iter__(self) -> tp.Iterator:
        self.n = 0
        return self

    def __next__(self) -> "spacy.tokens.Token":
        self.n += 1
        try:
            return self[self.n - 1]
        except IndexError:
            raise StopIteration

    def __repr__(self) -> str:
        return " ".join([span.text for span in self.inner_spans])

    @property
    def is_subclause(self) -> bool:
        """Clause is a subclause iff the finite verb is in last position."""
        return (
            self[-2].tag_ in FINITE_VERB_TAGS
            if self[-1].pos_ == "PUNCT"
            else self[-1].tag_ in FINITE_VERB_TAGS
        )

    @property
    def clause_type(self) -> str:
        return "SUB" if self.is_subclause else "MAIN"

    @property
    def inner_spans(self) -> tp.List["spacy.tokens.Span"]:
        """"Spans with punctuation tokens removed from span boundaries."""
        inner_spans = []
        for span in self.spans:
            if len(span) > 1:
                if span[0].pos_ == "PUNCT":
                    span = span[1:] 
                else:
                    span
                if span[-1].pos_ == "PUNCT":
                    span = span[:-1]  
                else:
                    span
            inner_spans.append(span)

        return inner_spans


class ClausedSentence(spacy.tokens.Span):
    """Span with extracted clause structure.

    This class is used to identify the positions of the finite verbs, to
    identify all the tokens that belong to the clause around each finite
    verb and to make a Clause object of each clause.
    """

    @property
    def __finite_verb_indices(self) -> tp.List[int]:
        return [token.i for token in self if token.tag_ in FINITE_VERB_TAGS]

    def progeny(
        self,
        index: int,
        stop_indices: tp.Optional[tp.List[int]] = None,
    ) -> tp.List["spacy.tokens.Token"]:
        """Walk through progeny tree until a stop index is met."""
        if stop_indices is None:
            stop_indices = []

        progeny = [index]  # consider a token its own child
        for child in self[index - self.start].children:
            if child.i in stop_indices or not (0 <= child.i < len(self)):
                continue
            progeny += [child.i] + self.progeny(child.i, stop_indices)

        return sorted(list(set(progeny)))


    @property
    def clauses(self) -> tp.Generator["Clause", None, None]:
        for verb_index in self.__finite_verb_indices:
            clause_tokens = [
                self[index]
                for index in self.progeny(
                    index=verb_index, stop_indices=self.__finite_verb_indices
                )
            ]

            spans = []

            # Create spans from range extraction of token indices
            for _, group in it.groupby(
                enumerate(clause_tokens),
                lambda index_token: index_token[0] - index_token[1].i,
            ):
                tokens = [item[1] for item in group]
                spans.append(self[tokens[0].i : tokens[-1].i + 1])

            yield Clause(spans)

def process_text(text):
    language_model = "de_core_news_lg"
    nlp = spacy.load(language_model) 
    doc = nlp(text)
    main = 0
    sub = 0
    try: 
        for sentence in doc.sents:
            claused_sentence = ClausedSentence(sentence.doc, 0, len(sentence.doc))
            clauses = list(claused_sentence.clauses)
            for clause in clauses:
                if clause.clause_type == "MAIN":
                    main += 1
                elif clause.clause_type == "SUB":
                    sub += 1
                        #print(f"{clause.clause_type}: {clause.inner_spans}")
    except IndexError: # Handle IndexError (list index out of range)
        return [np.nan, np.nan] 
    return [main, sub]

start_time = time.time()

def main_and_subclauses(text):
    text=text.strip()
    text=text.replace("\n", " ")
    nlp = spacy.load("de_core_news_lg") 
    nlp.add_pipe("emoji", first=True)
    doc = nlp(text) 
    main=0
    sub=0
    no_of_sents = 0
    for sentence in doc.sents:
        sentence=str(sentence)
        #if str(sentence)[0]=="#": #only important on Insta as there are hashtags at the end
            #break
        if sentence=="":  
            break
        
        if "https" in sentence: #only important on X as there is the link to the full article at the end
            for i in range(len(sentence)-1):
                if sentence[i] == "h" and sentence[i+1]=="t" and sentence[i+2]=="t" and sentence[i+3]=="t":
                    sentence = sentence[:i]
                    if sentence == " " or sentence == "":
                        break
                    else:
                        x = process_text(sentence)
                        main += x[0]
                        sub += x[1]
                        if sentence != "#":
                            no_of_sents += 1
                            print(sentence)
                        break
        
        else:
            x = process_text(sentence)
            main += x[0]
            sub += x[1]
            if sentence != "#": 
                no_of_sents += 1
                print(sentence)
    return [main, sub, no_of_sents]

#df = pd.read_csv("data_Instagram.csv", sep=";")
#df[["main_clauses", "sub_clauses", "no_of_sentences"]] = df["caption"].apply(lambda x: pd.Series(main_and_subclauses(x)))
#df.to_csv("data_Instagram_with_sentence_structure.csv", sep=";")

#df = pd.read_csv("data_X.csv", sep=";")
#df[["main_clauses", "sub_clauses", "no_of_sentences"]] = df["text"].apply(lambda x: pd.Series(main_and_subclauses(x)))
#df.to_csv("data_X_with_sentence_structure.csv", sep=";")

df = pd.read_csv("data_Instagram_final.csv", sep=";")
df[["main_clauses_image", "sub_clauses_image", "no_of_sentences_image"]] = df["Image Text"].apply(lambda x: pd.Series(main_and_subclauses(x)))
#df.to_csv("data_Instagram_images_with_sentence_structure.csv", sep=";")

end_time = time.time()
runtime = end_time - start_time
hours, remainder = divmod(runtime, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nTotal runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")