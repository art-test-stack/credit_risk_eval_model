import stanza
from stanza.server import CoreNLPClient


from typing import Any, List


# May use the package for finance :
# https://stanfordnlp.github.io/stanza/client_properties.html#:~:text=FINANCE_PROPS%20%3D%20%7B%0A%20%20%20%20%22depparse.model%22%3A%20%22/path/to/finance%2Dparser.gz%22%2C%0A%20%20%20%20%22ner.model%22%3A%20%22/path/to/finance%2Dner.ser.gz%22%0A%7D%0A%0Awith%20CoreNLPClient()%20as%20client%3A%0A%20%20%20%20bio_ann%20%3D%20client.annotate(bio_text%2C%20properties%3DBIOMEDICAL_PROPS)%0A%20%20%20%20finance_ann%20%3D%20client.annotate(finance_text%2C%20properties%3DFINANCE_PROPS)
# from stanza.server import CoreNLPClient
# FINANCE_PROPS = {
#     "depparse.model": "/path/to/finance-parser.gz",
#     "ner.model": "/path/to/finance-ner.ser.gz"
# }

# with CoreNLPClient() as client:
#     finance_ann = client.annotate(finance_text, properties=FINANCE_PROPS)


class CoreNLP:
    def __init__(
            self,
            annotators: List[str] = ['tokenize','ssplit','pos','lemma'],
            memory: str = '6G',
            endpoint: str = "http://localhost:8000",
        ) -> None:
        # download English model
        stanza.download('en')
        self.client = CoreNLPClient(
            annotators=annotators,
            # annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
            # timeout=30000,
            memory=memory,
            endpoint=endpoint,
        )

    def __call__(self, text) -> Any:
        return self.forward()
    
    def forward(self, text) -> List[str]:
        with self.client as client:
            ann = client.annotate(text)

        return [tok.value for sentence in ann.sentence for tok in sentence.token]