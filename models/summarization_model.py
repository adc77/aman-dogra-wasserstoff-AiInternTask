from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

class SummarizationModel:
    def __init__(self, language="english"):
        self.summarizer = LsaSummarizer(Stemmer(language))
        self.summarizer.stop_words = get_stop_words(language)

    def summarize_object(self, object_id, identification, extracted_text):
        # Combine all information into a single text
        full_text = f"Object ID: {object_id}\n"
        full_text += f"Identified as: {identification}\n"
        full_text += f"Extracted text: {extracted_text}\n"

        # Parse the text
        parser = PlaintextParser.from_string(full_text, Tokenizer("english"))

        # Generate summary
        summary = self.summarizer(parser.document, sentences_count=2)  # Adjust sentence count as needed
        
        return " ".join([str(sentence) for sentence in summary])

    def process_objects(self, identifications, text_data):
        summaries = {}
        for object_id in identifications.keys():
            identification = identifications[object_id]
            extracted_text = text_data.get(object_id, "")
            summary = self.summarize_object(object_id, identification, extracted_text)
            summaries[object_id] = summary
        return summaries